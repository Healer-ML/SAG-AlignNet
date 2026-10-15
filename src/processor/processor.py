import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = args.device
    if args.memory_size == 0 or args.memory_size < -1:
        raise ValueError("memory_size must be -1 or a positive integer")
    if not 0.0 < args.n <= 1.0:
        raise ValueError("n must be in (0, 1]")
    if args.distributed and device.startswith("cuda"):
        device = "cuda:{}".format(args.local_rank)
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("SAG-AlignNet.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "adv_loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "xld_loss": AverageMeter(),
        "hc_loss": AverageMeter(),
        "prior_proto_loss": AverageMeter(),
        "rasa_pitm_loss": AverageMeter(),
        "rasa_prd_loss": AverageMeter(),
        "rasa_mrtd_loss": AverageMeter(),
        #"sdmcross_loss": AverageMeter(),
        "itm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "mlm2_loss": AverageMeter(),
        "mlmpre_loss": AverageMeter(),
        "mlmerror_loss": AverageMeter(),
        #"help_loss":AverageMeter(),
        #"mco_loss":AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()#,
        #"help_acc":AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir) if get_rank() == 0 else None

    best_top1 = 0.0
    store1 = []################
    store2 = []################
    idstore = []###############
    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        sampler = getattr(train_loader, "sampler", None)
        if not hasattr(sampler, "set_epoch"):
            sampler = getattr(getattr(train_loader, "batch_sampler", None), "sampler", sampler)
        if args.distributed and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for meter in meters.values():
            meter.reset()
        model.train()
        store1_tem = []################
        store2_tem = []################
        idstore_tem = []###############
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            #ret = model(batch,epoch,store1,store2,idstore)
            ret,ifeats,tfeats = model(batch,epoch,store1,store2,idstore)#########################################
            # A memory bank is a target cache, so it must not retain old autograd graphs.
            store1_tem.append(ifeats.detach())################
            store2_tem.append(tfeats.detach())################
            idstore_tem.append(batch['pids'].detach())###############

            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['adv_loss'].update(ret.get('adv_loss', 0), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['xld_loss'].update(ret.get('xld_loss', 0), batch_size)
            meters['hc_loss'].update(ret.get('hc_loss', 0), batch_size)
            meters['prior_proto_loss'].update(ret.get('prior_proto_loss', 0), batch_size)
            meters['rasa_pitm_loss'].update(ret.get('rasa_pitm_loss', 0), batch_size)
            meters['rasa_prd_loss'].update(ret.get('rasa_prd_loss', 0), batch_size)
            meters['rasa_mrtd_loss'].update(ret.get('rasa_mrtd_loss', 0), batch_size)
            meters['itm_loss'].update(ret.get('itm_loss', 0), batch_size)
            #meters['sdmcross_loss'].update(ret.get('sdmcross_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            meters['mlm2_loss'].update(ret.get('mlm2_loss', 0), batch_size)
            meters['mlmpre_loss'].update(ret.get('mlmpre_loss', 0), batch_size)
            meters['mlmerror_loss'].update(ret.get('mlmerror_loss', 0), batch_size)
            #meters['help_loss'].update(ret.get('help_loss', 0), batch_size)
            #meters['mco_loss'].update(ret.get('mco_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)
            #meters['help_acc'].update(ret.get('help_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            if args.grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip_max_norm
                )
            optimizer.step()
            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        memory_i = torch.cat(store1_tem, dim=0)
        memory_t = torch.cat(store2_tem, dim=0)
        memory_id = torch.cat(idstore_tem, dim=0)
        memory_limit = max(1, int(round(memory_i.size(0) * args.n)))
        if args.memory_size > 0:
            memory_limit = min(memory_limit, args.memory_size)
        if memory_limit < memory_i.size(0):
            # Uniformly sample once per epoch; the existing global seed makes this reproducible.
            indices = torch.randperm(memory_i.size(0), device=memory_i.device)[:memory_limit]
            memory_i = memory_i[indices]
            memory_t = memory_t[indices]
            memory_id = memory_id[indices]
        store1 = [memory_i]################
        store2 = [memory_t]################
        idstore = [memory_id]###############
        effective_m = min(args.m, memory_i.size(0))
        logger.info(
            "Memory bank updated: %d samples (n=%.4f, cap=%d); "
            "m requested=%d effective=%d neighbor_weight=%.6g",
            memory_i.size(0), args.n, args.memory_size,
            args.m, effective_m, args.neighbor_weight,
        )
        if tb_writer is not None:
            tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
            tb_writer.add_scalar('temperature', ret['temperature'], epoch)
            for k, v in meters.items():
                if v.avg > 0:
                    tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    metrics = evaluator.eval(model.module.eval(), logger=logger)
                else:
                    metrics = evaluator.eval(model.eval(), logger=logger)
                validation_mr = metrics["mR"]

                torch.cuda.empty_cache()
                if best_top1 < validation_mr:
                    best_top1 = validation_mr
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best validation mR: {best_top1:.4f} at epoch {arguments['epoch']}")
    if tb_writer is not None:
        tb_writer.close()


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("SAG-AlignNet.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    return evaluator.eval(model.eval(), logger=logger)
