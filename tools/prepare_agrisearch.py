#!/usr/bin/env python3
"""Prepare AgriSearch's five annotation sets without copying raw images.

By default this normalizes split aliases, copies JSON annotations, and creates
an ``img`` symlink in the repository's canonical data layout. With
``--generate-splits``, whole image-ID groups are deterministically assigned to
train/val/test and IDs are re-numbered so training IDs remain contiguous.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


DATASETS = {
    "FRS-CMR": ("farm", [("farm",)]),
    "CroDis4CMR": ("crop", [("data_cddm", "crop"), ("data_CDDM", "crop")]),
    "CCPRD": ("agr", [("agr",)]),
    "GjPest4CMR": ("wp", [("wp",)]),
    "FruDis4CMR": ("fruit", [("data_cddm", "fruit"), ("data_CDDM", "fruit")]),
}

SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
}


def find_source_dataset(source_root, candidates, internal_name):
    for components in candidates:
        root = source_root.joinpath(*components)
        annotation = root / "json" / (internal_name + ".json")
        image_dir = next((root / name for name in ("img", "imgs") if (root / name).is_dir()), None)
        if annotation.is_file() and image_dir is not None:
            return root, image_dir, annotation
    checked = [str(source_root.joinpath(*parts)) for parts in candidates]
    raise FileNotFoundError(
        "Could not find {}/json/{}.json and img|imgs under: {}".format(
            " or ".join(checked), internal_name, ", ".join(checked)
        )
    )


def normalize_image_path(value):
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("img_path must be a relative path without '..': {!r}".format(value))
    parts = list(path.parts)
    if parts and parts[0].lower() in {"img", "imgs"}:
        parts = parts[1:]
    if not parts:
        raise ValueError("img_path is empty")
    return PurePosixPath(*parts).as_posix()


def normalized_records(annotation_path, image_dir, generate_splits, seed, ratios, check_images):
    records = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a non-empty JSON list: {}".format(annotation_path))

    prepared = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Record {} is not a JSON object".format(index))
        for key in ("id", "img_path", "captions"):
            if key not in record:
                raise ValueError("Record {} is missing {!r}".format(index, key))
        try:
            identity = int(record["id"])
        except (TypeError, ValueError):
            raise ValueError("Record {} has a non-integer id".format(index))
        if identity < 0:
            raise ValueError("Record {} has a negative id".format(index))
        image_path = normalize_image_path(record["img_path"])
        captions = record["captions"]
        if not isinstance(captions, list) or not captions or any(not isinstance(text, str) or not text.strip() for text in captions):
            raise ValueError("Record {} must have a non-empty list of caption strings".format(index))
        item = dict(record)
        item["id"] = identity
        item["img_path"] = image_path
        item["captions"] = captions
        if not generate_splits:
            raw_split = str(record.get("split", "")).strip().lower()
            if raw_split not in SPLIT_ALIASES:
                raise ValueError("Record {} has missing/unknown split {!r}".format(index, raw_split))
            item["split"] = SPLIT_ALIASES[raw_split]
        if check_images and not (image_dir / image_path).is_file():
            raise FileNotFoundError("Image referenced by record {} is missing: {}".format(index, image_dir / image_path))
        prepared.append(item)

    if generate_splits:
        rng = random.Random(seed)
        grouped = defaultdict(list)
        for item in prepared:
            grouped[item["id"]].append(item)
        identity_groups = list(grouped.items())
        rng.shuffle(identity_groups)
        n_groups = len(identity_groups)
        n_train = max(1, int(n_groups * ratios[0]))
        n_val = max(1, int(n_groups * ratios[1]))
        if n_train + n_val >= n_groups:
            raise ValueError("Ratios leave no identities for the test split")
        train_ids = set(identity for identity, _ in identity_groups[:n_train])
        val_ids = set(identity for identity, _ in identity_groups[n_train:n_train + n_val])
        for item in prepared:
            item["split"] = "train" if item["id"] in train_ids else ("val" if item["id"] in val_ids else "test")
        # Keep the model's training classifier labels contiguous while retaining
        # one globally consistent ID per source identity across all splits.
        source_ids_by_split = {
            split: sorted({item["id"] for item in prepared if item["split"] == split})
            for split in ("train", "val", "test")
        }
        new_id = {}
        next_id = 0
        for split in ("train", "val", "test"):
            for source_id in source_ids_by_split[split]:
                new_id[source_id] = next_id
                next_id += 1
        for item in prepared:
            item["id"] = new_id[item["id"]]

    train_ids = sorted({item["id"] for item in prepared if item["split"] == "train"})
    if train_ids and train_ids != list(range(len(train_ids))):
        raise ValueError(
            "Training IDs must be zero-based and contiguous; first IDs are {}. "
            "Use --generate-splits only when you intend to create a new split.".format(train_ids[:10])
        )
    if not all(any(item["split"] == split for item in prepared) for split in ("train", "val", "test")):
        raise ValueError("All train, val/dev, and test splits must contain at least one record")
    return prepared


def link_image_directory(source, destination):
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        raise FileExistsError("Existing image symlink points elsewhere: {}".format(destination))
    if destination.exists():
        raise FileExistsError("Image destination already exists and is not a symlink: {}".format(destination))
    try:
        destination.symlink_to(source, target_is_directory=True)
    except OSError as error:
        raise OSError(
            "Cannot create image symlink {} -> {}. Run on a filesystem that supports "
            "directory symlinks, or create the canonical layout manually.".format(destination, source)
        ) from error


def write_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Root containing farm, agr, wp, and data_cddm")
    parser.add_argument("--output-root", type=Path, default=Path("data/AgriSearch"))
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--check-images", action="store_true", help="verify every annotation image path exists")
    parser.add_argument("--generate-splits", action="store_true", help="create new deterministic image-ID-disjoint splits")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()

    if args.generate_splits and abs(sum(args.ratios) - 1.0) > 1e-8:
        parser.error("--ratios must sum to 1")
    if any(value <= 0 for value in args.ratios):
        parser.error("all --ratios must be positive")
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not source_root.is_dir():
        parser.error("source root does not exist: {}".format(source_root))

    print("Dataset         Records  Captions  Train IDs  Train/Val/Test")
    print("-" * 70)
    for benchmark_name in args.datasets:
        internal_name, candidates = DATASETS[benchmark_name]
        _, source_images, source_json = find_source_dataset(source_root, candidates, internal_name)
        records = normalized_records(
            source_json, source_images, args.generate_splits, args.seed, args.ratios, args.check_images
        )
        target_root = output_root / benchmark_name
        link_image_directory(source_images, target_root / "img")
        write_json(target_root / "json" / (internal_name + ".json"), records)
        split_counts = Counter(item["split"] for item in records)
        train_ids = len({item["id"] for item in records if item["split"] == "train"})
        caption_count = sum(len(item["captions"]) for item in records)
        split_text = "{}/{}/{}".format(split_counts["train"], split_counts["val"], split_counts["test"])
        print("{:<15} {:>7} {:>9} {:>10}  {}".format(
            benchmark_name, len(records), caption_count, train_ids, split_text
        ))
        print("  annotations: {}".format(target_root / "json" / (internal_name + ".json")))
        print("  image link:  {} -> {}".format(target_root / "img", source_images.resolve()))


if __name__ == "__main__":
    main()
