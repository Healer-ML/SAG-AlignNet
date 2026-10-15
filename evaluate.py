"""Evaluate an inference checkpoint on a validation or held-out test split."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch

from datasets import build_dataloader
from model import build_model
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from utils.metrics import Evaluator
from utils.options import get_args


def main(argv=None):
    args = get_args(argv)
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for evaluation")
    if not args.split:
        raise SystemExit("--split val or --split test is required")
    args.training = False
    logger = setup_logger("SAG-AlignNet", save_dir=args.output_dir, if_train=False)
    logger.info("dataset=%s split=%s checkpoint=%s", args.dataset_name, args.val_dataset, args.checkpoint)

    image_loader, text_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    Checkpointer(model).load(args.checkpoint)
    model.to(args.device)
    results = Evaluator(image_loader, text_loader).eval(model.eval(), logger=logger)
    report = {
        "method": "SAG-AlignNet",
        "dataset": args.dataset_name,
        "split": args.val_dataset,
        "checkpoint": os.path.abspath(args.checkpoint),
        "metrics": results,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.eval_output:
        output_path = os.path.abspath(args.eval_output)
        output_parent = os.path.dirname(output_path)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
