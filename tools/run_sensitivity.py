#!/usr/bin/env python3
"""Run one-factor-at-a-time sensitivity sweeps for active model arguments.

Every value starts a fresh training run. The dataset YAML supplies all fixed
settings, the selected argument is overridden on the command line, and the
validation split selects the checkpoint. A JSON and CSV summary are updated
after every run so interrupted sweeps remain inspectable.
"""

import argparse
import csv
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

FLOAT_PARAMETERS = {
    "alpha": [0.10, 0.25, 0.40, 0.55, 0.70, 0.85],
    "rho": [0.20, 0.25, 0.30, 0.35, 0.40, 0.55],
    "n": [0.25, 0.50, 0.75, 1.0],
    "neighbor_weight": [0.0, 0.00025, 0.0005, 0.001, 0.002],
    "lambda_": [0.0, 0.25, 0.50, 0.75],
    "temperature": [0.005, 0.01, 0.02, 0.05],
}
INTEGER_PARAMETERS = {
    "K": [2, 4, 6, 8],
    "m": [25, 50, 100, 200],
    "memory_size": None,  # Default values are derived from the configured bank capacity.
}
PARAMETERS = {**FLOAT_PARAMETERS, **INTEGER_PARAMETERS}
PARAMETER_ALIASES = {
    "mask_pro": "rho",
    "soft_weight": "alpha",
    "num_prototypes": "K",
    "memory_ratio": "n",
    "neighbor_k": "m",
    "soft_label_alpha": "lambda_",
}

CSV_FIELDS = [
    "dataset", "parameter", "requested_value", "effective_value", "seed",
    "validation_mR", "best_epoch", "memory_samples_per_rank",
    "effective_m", "return_code", "run_dir", "train_log",
]
AGGREGATE_FIELDS = [
    "dataset", "parameter", "requested_value", "effective_value",
    "num_seeds", "mean_validation_mR", "std_validation_mR", "seeds",
]


def load_config(path):
    """Load a YAML config and recursively merge its base config."""
    import yaml

    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping: {}".format(path))
    config = normalize_config_keys(config)
    base = config.pop("base_config", None)
    if base:
        merged = load_config(path.parent / base)
        merged.update(config)
        return merged
    return config


def normalize_config_keys(config):
    """Translate legacy config names to the parameter symbols in the figures."""
    config = dict(config)
    if "beta" in config:
        config.setdefault("adam_beta2", config.pop("beta"))
        if "alpha" in config:
            config.setdefault("adam_beta1", config.pop("alpha"))
    aliases = {
        "mask_pro": "rho",
        "soft_weight": "alpha",
        "num_prototypes": "K",
        "memory_ratio": "n",
        "neighbor_k": "m",
        "soft_label_alpha": "lambda_",
        "lambda": "lambda_",
    }
    for old_name, new_name in aliases.items():
        if old_name in config:
            config.setdefault(new_name, config.pop(old_name))
    if "loss_names" in config:
        config["loss_names"] = "+".join(
            "matching_label" if name.strip() == "id" else name.strip()
            for name in str(config["loss_names"]).split("+")
        )
    return config


def training_pair_count(config):
    """Count training captions, matching RSTPReid._process_anno()."""
    root = Path(config["root_dir"])
    if not root.is_absolute():
        root = ROOT / root
    annotation_path = root / "json" / (config["dataset_name"] + ".json")
    if not annotation_path.is_file():
        return None, annotation_path
    records = json.loads(annotation_path.read_text(encoding="utf-8"))
    count = 0
    for record in records:
        split = str(record.get("split", "")).strip().lower()
        if split == "train":
            captions = record.get("captions", [])
            if not isinstance(captions, list):
                raise ValueError("captions must be a list in {}".format(annotation_path))
            count += len(captions)
    if count <= 0:
        raise ValueError("No training captions found in {}".format(annotation_path))
    return count, annotation_path


def samples_per_rank(num_pairs, nproc_per_node):
    if num_pairs is None:
        return None
    # DistributedSampler uses ceil(N / world_size) examples on each rank.
    return int(math.ceil(num_pairs / float(nproc_per_node)))


def memory_capacity(config, pair_count, nproc_per_node, parameter=None, value=None):
    """Return the actual per-rank memory bank size for a proposed run."""
    per_rank = samples_per_rank(pair_count, nproc_per_node)
    if per_rank is None:
        return None
    ratio = float(config.get("n", 1.0))
    cap = int(config.get("memory_size", -1))
    if parameter == "n":
        ratio = float(value)
    elif parameter == "memory_size":
        cap = int(value)
    capacity = max(1, int(round(per_rank * ratio)))
    if cap > 0:
        capacity = min(capacity, cap)
    return capacity


def default_values(parameter, config, pair_count, nproc_per_node):
    if parameter == "memory_size":
        capacity = memory_capacity(config, pair_count, nproc_per_node)
        if capacity is None:
            raise FileNotFoundError(
                "The training annotation JSON is required to derive memory_size "
                "values; pass --values with explicit positive integer caps."
            )
        # The full-capacity point is the configured baseline, so add three
        # smaller caps to give distinct effective bank sizes.
        values = [max(1, int(round(capacity * fraction)))
                  for fraction in (0.25, 0.5, 0.75)]
    else:
        values = list(PARAMETERS[parameter])
    if parameter in config and config[parameter] is not None:
        values.append(config[parameter])
    return unique_values(values)


def unique_values(values):
    result = []
    for value in values:
        if not any(value == previous for previous in result):
            result.append(value)
    return result


def parse_values(parameter, raw_values):
    if parameter in INTEGER_PARAMETERS:
        try:
            values = [int(value) for value in raw_values]
        except ValueError as exc:
            raise ValueError("{} requires integer values".format(parameter)) from exc
    else:
        try:
            values = [float(value) for value in raw_values]
        except ValueError as exc:
            raise ValueError("{} requires numeric values".format(parameter)) from exc
    return unique_values(values)


def validate_values(parameter, values):
    for value in values:
        if parameter == "lambda_" and not 0.0 <= value <= 1.0:
            raise ValueError("{} values must be in [0, 1]".format(parameter))
        if parameter == "alpha" and not 0.0 <= value <= 1.0:
            raise ValueError("alpha values must be in [0, 1]")
        if parameter in {"rho", "n"} and not 0.0 < value <= 1.0:
            raise ValueError("{} values must be in (0, 1]".format(parameter))
        if parameter in {"K", "m"} and value <= 0:
            raise ValueError("{} values must be positive".format(parameter))
        if parameter == "memory_size" and (value == 0 or value < -1):
            raise ValueError("memory_size values must be -1 or positive")
        if parameter == "neighbor_weight" and value < 0.0:
            raise ValueError("{} values must be non-negative".format(parameter))
        if parameter == "temperature" and value <= 0.0:
            raise ValueError("temperature values must be positive")


def validate_active(parameter, config):
    losses = set(str(config.get("loss_names", "")).split("+"))
    soft_mix = float(config.get("lambda_", 0.5))
    if parameter in {
        "K", "n", "memory_size", "m",
        "neighbor_weight", "lambda_", "alpha",
    } and not {"sdm", "matching_label"}.issubset(losses):
        raise ValueError(
            "{} requires both sdm and matching_label in loss_names".format(parameter)
        )
    if parameter == "rho" and (not config.get("MLM", True) or "mlm" not in losses):
        raise ValueError("rho only affects the active MLM training path")
    if parameter == "temperature" and not ({"sdm", "itc"} & losses):
        raise ValueError("temperature requires sdm or itc in loss_names")
    if parameter in {
        "n", "memory_size", "m", "neighbor_weight",
        "lambda_", "alpha",
    } and int(config.get("num_epoch", 30)) < 2:
        raise ValueError("{} affects neighbor targets from epoch 2 onward".format(parameter))
    if parameter in {
        "n", "memory_size", "m", "neighbor_weight", "alpha",
    } and soft_mix == 0.0:
        raise ValueError("{} is inactive because lambda_ is zero".format(parameter))
    if parameter == "rho" and float(config.get("mlm_loss_weight", 1.0)) == 0.0:
        raise ValueError("rho is inactive because mlm_loss_weight is zero")


def format_value(value):
    if isinstance(value, int):
        return str(value)
    return format(float(value), ".12g")


def value_slug(value):
    return format_value(value).replace("-", "m").replace(".", "p")


def write_json(path, value):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def write_csv(path, rows, fields=CSV_FIELDS):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def aggregate_rows(rows):
    groups = {}
    for row in rows:
        if row["validation_mR"] == "":
            continue
        key = (row["dataset"], row["parameter"], row["requested_value"], row["effective_value"])
        groups.setdefault(key, []).append((int(row["seed"]), float(row["validation_mR"])))
    aggregated = []
    for key, results in groups.items():
        scores = [score for _, score in results]
        aggregated.append({
            "dataset": key[0],
            "parameter": key[1],
            "requested_value": key[2],
            "effective_value": key[3],
            "num_seeds": len(scores),
            "mean_validation_mR": "{:.6f}".format(statistics.mean(scores)),
            "std_validation_mR": "{:.6f}".format(
                statistics.pstdev(scores) if len(scores) > 1 else 0.0
            ),
            "seeds": ",".join(str(seed) for seed, _ in results),
        })
    return aggregated


def build_command(args, config_path, parameter, value, seed, run_root):
    train_script = ROOT / "train.py"
    if args.nproc_per_node > 1:
        command = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node", str(args.nproc_per_node), str(train_script),
        ]
    else:
        command = [sys.executable, str(train_script)]
    run_name = "sens_{}_{}_s{}".format(parameter, value_slug(value), seed)
    command.extend([
        "--config-file", str(config_path),
        "--name", run_name,
        "--output_dir", str(run_root / "train_output"),
        "--seed", str(seed),
        "--val_dataset", "val",
        "--" + parameter, format_value(value),
    ])
    return command


def find_train_log(run_root):
    matches = list((run_root / "train_output").rglob("train_log.txt"))
    if len(matches) != 1:
        return None
    return matches[0]


def read_validation_result(log_path, stdout_path):
    pattern = re.compile(
        r"best validation mR:\s*([0-9]+(?:\.[0-9]+)?)\s+at epoch\s+(\d+)"
    )
    text_parts = []
    if log_path is not None and log_path.is_file():
        text_parts.append(log_path.read_text(encoding="utf-8", errors="replace"))
    if stdout_path.is_file():
        text_parts.append(stdout_path.read_text(encoding="utf-8", errors="replace"))
    matches = [match for text in text_parts for match in pattern.finditer(text)]
    if not matches:
        return None, None
    match = matches[-1]
    return float(match.group(1)), int(match.group(2))


def make_parser(datasets):
    parser = argparse.ArgumentParser(
        description="Run fresh, validation-selected, one-factor-at-a-time sweeps."
    )
    parser.add_argument("--dataset", required=True, choices=datasets)
    parser.add_argument(
        "--parameter", required=True,
        choices=sorted(PARAMETERS) + sorted(PARAMETER_ALIASES) + ["all"],
        help="one active training argument, or all registered arguments",
    )
    parser.add_argument(
        "--values", nargs="+", default=None,
        help="optional values for one parameter; defaults are supplied per argument",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--output-root", default="outputs/sensitivity")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    dataset_names = sorted(path.stem for path in CONFIG_DIR.glob("*.yaml")
                           if path.stem != "base")
    args = make_parser(dataset_names).parse_args(argv)
    args.parameter = PARAMETER_ALIASES.get(args.parameter, args.parameter)
    if args.nproc_per_node <= 0:
        raise ValueError("nproc-per-node must be positive")
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if args.parameter == "all" and args.values:
        raise ValueError("--values can only be used with one --parameter")

    config_path = CONFIG_DIR / (args.dataset + ".yaml")
    config = load_config(config_path)
    if config.get("sampler", "random") != "random":
        raise ValueError(
            "The sensitivity runner currently requires sampler=random so its "
            "per-rank memory-capacity calculations match the training loader."
        )

    pair_count, annotation_path = training_pair_count(config)
    if args.parameter in {"n", "memory_size", "m", "all"} and pair_count is None:
        raise FileNotFoundError(
            "Cannot resolve the actual memory/neighbor capacity because this file "
            "is missing: {}".format(annotation_path)
        )

    selected_parameters = list(PARAMETERS) if args.parameter == "all" else [args.parameter]
    sessions = []
    for parameter in selected_parameters:
        validate_active(parameter, config)
        if args.values is not None:
            values = parse_values(parameter, args.values)
        else:
            values = default_values(
                parameter, config, pair_count, args.nproc_per_node
            )
        if not values:
            raise ValueError("No sweep values were supplied for {}".format(parameter))
        validate_values(parameter, values)

        effective_memories = [
            memory_capacity(config, pair_count, args.nproc_per_node, parameter, value)
            for value in values
        ]
        if parameter in {"n", "memory_size"}:
            observed = [capacity for capacity in effective_memories if capacity is not None]
            if len(set(observed)) != len(observed):
                raise ValueError(
                    "The requested {} values collapse to duplicate effective memory "
                    "sizes under the other configured memory limit: {}. Adjust the "
                    "values or the dataset config so every point changes the bank."
                    .format(parameter, observed)
                )
        if parameter == "m" and pair_count is not None:
            capacity = memory_capacity(config, pair_count, args.nproc_per_node)
            clipped = [value for value in values if value > capacity]
            if clipped:
                raise ValueError(
                    "m values {} exceed the configured effective memory "
                    "capacity {} and would be clipped by the model. Reduce the values "
                    "or increase the configured memory capacity.".format(clipped, capacity)
                )
        if parameter in {"n", "memory_size"} and pair_count is not None:
            min_capacity = min(effective_memories)
            configured_neighbors = int(config.get("m", 100))
            if configured_neighbors > min_capacity:
                raise ValueError(
                    "This memory sweep would also change the model's effective "
                    "m: the smallest bank has {} samples, while the config "
                    "requests {} neighbors. Lower m in the dataset config "
                    "or choose larger memory values before running the sweep."
                    .format(min_capacity, configured_neighbors)
                )
        sessions.append((parameter, values, effective_memories))

    if args.dry_run:
        for parameter, values, effective_memories in sessions:
            print("\n{}: {}".format(parameter, values))
            for value, effective_memory in zip(values, effective_memories):
                for seed in args.seeds:
                    run_root = Path(args.output_root) / args.dataset / "DRY_RUN" / parameter / value_slug(value) / ("seed_{}".format(seed))
                    if not run_root.is_absolute():
                        run_root = ROOT / run_root
                    command = build_command(
                        args, config_path, parameter, value, seed, run_root,
                    )
                    print("  value={} effective_memory={} seed={}".format(
                        format_value(value), effective_memory, seed
                    ))
                    print("  " + shlex.join(command))
        return 0

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    session_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session_dir = output_root / args.dataset / session_stamp
    session_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "dataset": args.dataset,
        "config": str(config_path),
        "validation_split": "val",
        "seeds": args.seeds,
        "nproc_per_node": args.nproc_per_node,
        "training_pairs_per_rank": samples_per_rank(pair_count, args.nproc_per_node),
        "runs": [],
    }
    json_path = session_dir / "summary.json"
    csv_path = session_dir / "summary.csv"
    aggregate_path = session_dir / "summary_by_value.csv"
    write_json(json_path, summary)
    write_csv(csv_path, [])
    write_csv(aggregate_path, [], AGGREGATE_FIELDS)

    rows = []
    for parameter, values, effective_memories in sessions:
        for value, effective_memory in zip(values, effective_memories):
            for seed in args.seeds:
                run_root = session_dir / "runs" / parameter / value_slug(value) / ("seed_{}".format(seed))
                run_root.mkdir(parents=True, exist_ok=False)
                command = build_command(
                    args, config_path, parameter, value, seed, run_root
                )
                stdout_path = run_root / "stdout.log"
                print("RUN {}={} seed={} -> {}".format(
                    parameter, format_value(value), seed, run_root
                ), flush=True)
                with stdout_path.open("w", encoding="utf-8") as stdout_handle:
                    completed = subprocess.run(
                        command, cwd=str(ROOT), stdout=stdout_handle,
                        stderr=subprocess.STDOUT, check=False,
                    )

                train_log = find_train_log(run_root)
                validation_mr, best_epoch = read_validation_result(
                    train_log, stdout_path
                )
                effective_m = None
                if parameter == "m":
                    effective_m = min(int(value), effective_memory)
                elif "m" in config and effective_memory is not None:
                    effective_m = min(int(config["m"]), effective_memory)
                effective_value = effective_m if parameter == "m" else value
                row = {
                    "dataset": args.dataset,
                    "parameter": parameter,
                    "requested_value": format_value(value),
                    "effective_value": format_value(effective_value) if effective_value is not None else "",
                    "seed": seed,
                    "validation_mR": "" if validation_mr is None else "{:.6f}".format(validation_mr),
                    "best_epoch": "" if best_epoch is None else best_epoch,
                    "memory_samples_per_rank": "" if effective_memory is None else effective_memory,
                    "effective_m": "" if effective_m is None else effective_m,
                    "return_code": completed.returncode,
                    "run_dir": str(run_root),
                    "train_log": "" if train_log is None else str(train_log),
                }
                rows.append(row)
                summary["runs"].append({
                    **row,
                    "validation_mR": validation_mr,
                    "command": command,
                    "stdout_log": str(stdout_path),
                })
                write_json(json_path, summary)
                write_csv(csv_path, rows)
                write_csv(aggregate_path, aggregate_rows(rows), AGGREGATE_FIELDS)
                if completed.returncode != 0 or validation_mr is None:
                    raise RuntimeError(
                        "Run failed or did not report best validation mR. "
                        "Inspect {} and {}".format(stdout_path, train_log)
                    )
                print("DONE validation mR={:.4f} best_epoch={}".format(
                    validation_mr, best_epoch
                ), flush=True)

    print("Sensitivity summary: {}".format(json_path))
    print("CSV summary: {}".format(csv_path))
    print("Aggregated summary: {}".format(aggregate_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
