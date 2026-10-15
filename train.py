"""Train SAG-AlignNet on one AgriSearch dataset split."""

import os
import os.path as op
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch

from datasets import build_dataloader
from model import build_model
from processor.processor import do_train
from solver import build_lr_scheduler, build_optimizer
from utils.checkpoint import Checkpointer
from utils.comm import get_rank, synchronize
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from utils.metrics import Evaluator
from utils.options import get_args


def set_seed(seed=1, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def main(argv=None):
    args = get_args(argv)
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.distributed = world_size > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        device = "cuda:{}".format(args.local_rank)
    else:
        device = args.device
    set_seed(args.seed + get_rank(), deterministic=args.deterministic)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for a smoke run")

    run_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, "{}_{}".format(run_time, args.name))
    logger = setup_logger(
        "SAG-AlignNet", save_dir=args.output_dir, if_train=True,
        distributed_rank=get_rank(),
    )
    logger.info("world_size=%d device=%s seed=%d", world_size, device, args.seed)
    logger.info("%s", str(args).replace(",", "\n"))
    save_train_configs(args.output_dir, args)

    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    logger.info("Total parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            broadcast_buffers=False,
        )

    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
    else:
        start_epoch = 1
    do_train(
        start_epoch, args, model, train_loader, evaluator,
        optimizer, scheduler, checkpointer,
    )


if __name__ == "__main__":
    main()
