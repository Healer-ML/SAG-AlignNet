"""Command-line and YAML configuration for SAG-AlignNet."""

import argparse
from pathlib import Path

import yaml


def _parser():
    parser = argparse.ArgumentParser(description="SAG-AlignNet")
    parser.add_argument("--config-file", "--config_file", dest="config_file", default="")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="sag-alignnet")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=100, type=int)
    parser.add_argument("--eval_period", default=1, type=int)
    parser.add_argument("--val_dataset", default="val", choices=("val", "test"))
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--resume_ckpt_file", default="")
    parser.add_argument("--checkpoint", default="", help="model checkpoint for evaluation")
    parser.add_argument("--split", default="", choices=("", "val", "test"))
    parser.add_argument("--eval_output", default="", help="optional JSON file for evaluation metrics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--deterministic", dest="deterministic", default=True, action="store_true")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")

    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--img_aug", dest="img_aug", default=True, action="store_true")
    parser.add_argument("--no-img-aug", dest="img_aug", action="store_false")
    parser.add_argument("--cmt_depth", type=int, default=4)
    parser.add_argument("--masked_token_rate", type=float, default=0.8)
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1)
    parser.add_argument("--lr_factor", type=float, default=5.0)
    parser.add_argument("--MLM", dest="MLM", default=True, action="store_true")
    parser.add_argument("--disable-mlm", dest="MLM", action="store_false")

    parser.add_argument("--loss_names", default="itc+sdm+matching_label+mlm+adv")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0)
    parser.add_argument("--rho", "--mask_pro", dest="rho", type=float, default=0.25,
                        help="token masking ratio (Fig. 8)")
    parser.add_argument(
        "--mask_strategy", default="random",
        choices=("random", "noun", "noun_adj", "agriculture_random"),
    )
    parser.add_argument("--alpha", "--soft_weight", dest="alpha", type=float, default=0.1,
                        help="weight of the current matching label in soft targets (Fig. 8)")
    parser.add_argument("--K", "--num_prototypes", dest="K", type=int, default=6,
                        help="number of learnable prototypes (Fig. 9)")
    parser.add_argument("--memory_size", type=int, default=-1)
    parser.add_argument("--n", "--memory_ratio", dest="n", type=float, default=1.0,
                        help="fraction of current training samples retained in the memory bank (Fig. 9)")
    parser.add_argument("--m", "--neighbor_k", dest="m", type=int, default=100,
                        help="number of nearest memory samples used for neighbor targets (Fig. 9)")
    parser.add_argument("--neighbor_weight", type=float, default=0.001,
                        help="weight added for each retrieved neighbor label")
    parser.add_argument("--lambda", "--lambda_", "--soft_label_alpha", dest="lambda_",
                        type=float, default=0.5,
                        help="hard/soft matching-label loss mixing coefficient (Fig. 9)")

    parser.add_argument("--img_size", nargs=2, type=int, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    parser.add_argument("--optimizer", default="Adam", choices=("SGD", "Adam", "AdamW"))
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4.4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.0)
    parser.add_argument("--adam-beta1", "--adam_beta1", "--beta1", dest="adam_beta1",
                        type=float, default=0.9)
    parser.add_argument("--adam-beta2", "--adam_beta2", "--beta2", "--beta",
                        dest="adam_beta2", type=float, default=0.999)

    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--grad_clip_max_norm", type=float, default=5.0,
                        help="maximum gradient norm; set <= 0 to disable clipping")
    parser.add_argument("--milestones", nargs="+", type=int, default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=8)
    parser.add_argument("--warmup_method", default="linear", choices=("constant", "linear"))
    parser.add_argument("--lrscheduler", default="cosine", choices=("step", "exp", "poly", "cosine", "linear"))
    parser.add_argument("--target_lr", type=float, default=0.0)
    parser.add_argument("--power", type=float, default=0.9)

    parser.add_argument("--dataset_name", default="CCPRD")
    parser.add_argument("--sampler", default="random", choices=("random", "identity"))
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="data/AgriSearch/CCPRD")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval-only", dest="training", default=True, action="store_false")
    parser.add_argument(
        "--mechanism_variant", default="native",
        help="native or comma-separated cprod=plot|prior,masalign=irra|rasa",
    )
    return parser


def _read_config(path):
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping: {}".format(path))
    config = _normalize_config_keys(config)
    base = config.pop("base_config", None)
    if base:
        merged = _read_config(path.parent / base)
        merged.update(config)
        config = merged
    return config


def _normalize_config_keys(config):
    """Translate legacy YAML names to the symbols used in the paper figures."""
    config = dict(config)
    # Older YAML files used alpha/beta for Adam and soft_weight for Fig. 8's
    # alpha. Interpret that pair before translating soft_weight.
    if "beta" in config:
        config.setdefault("adam_beta2", config["beta"])
        config.pop("beta")
        if "alpha" in config:
            config.setdefault("adam_beta1", config["alpha"])
            config.pop("alpha")

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
            config.setdefault(new_name, config[old_name])
            config.pop(old_name)
    if "loss_names" in config:
        config["loss_names"] = _normalize_loss_names(config["loss_names"])
    return config


def _normalize_loss_names(value):
    """Use descriptive task names while accepting the former `id` token."""
    names = [name.strip() for name in str(value).split("+") if name.strip()]
    return "+".join("matching_label" if name == "id" else name for name in names)


def get_args(argv=None):
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config-file", "--config_file", dest="config_file", default="")
    probe, _ = config_probe.parse_known_args(argv)

    parser = _parser()
    config = _read_config(probe.config_file) if probe.config_file else {}
    known = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError("Unknown config key(s): {}".format(", ".join(unknown)))
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    args.loss_names = _normalize_loss_names(args.loss_names)
    args.img_size = tuple(args.img_size)
    args.milestones = tuple(args.milestones)
    if args.split:
        args.val_dataset = args.split
    if not 0.0 < args.rho <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if args.K <= 0 or args.m <= 0:
        raise ValueError("K and m must be positive")
    if not 0.0 < args.n <= 1.0:
        raise ValueError("n must be in (0, 1]")
    if not 0.0 <= args.lambda_ <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    if args.neighbor_weight < 0.0:
        raise ValueError("neighbor_weight must be non-negative")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return args
