import copy
import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss


# ********************************************************
# 弃用！双优化器存在梯度问题，改为默认train_engine.py合并loss优化！
# ********************************************************
def train_one_epoch(
    train_loader,
    model,
    optimizer_ssa,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    model_ema=None,
    clip_grad_l2norm=-1,
    logging_interval=200,
    scaler=None,
    scaler_ssa=None,
):
    """Training the model for one epoch"""

    logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    num_iters = len(train_loader)
    use_amp = False if scaler is None else True

    model.train()
    for iter_idx, data_dict in enumerate(train_loader):

        # current learning rate
        curr_backbone_lr = None
        if hasattr(model.module, "backbone"):  # if backbone exists
            if model.module.backbone.freeze_backbone == False:  # not frozen
                curr_backbone_lr = scheduler.get_last_lr()[1]   # SSA若不在主优化器,0为不学的SSA, 需调为1--即adapter的学习率
        curr_det_lr = scheduler.get_last_lr()[-1]

        n_class = len(train_loader.dataset.class_map)  # 生成类别矩阵用
        # dection for/backward
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            losses = model(**data_dict, n_class=n_class, return_loss=True)
        loss_det = losses.get("cost", None)

        # detection backward
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss_det).backward()
            if clip_grad_l2norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_det.backward()
            if clip_grad_l2norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
            optimizer.step()

        # 缓存主干生成的中间变量供 SSA 使用
        backbone_wrapper = model.module.backbone
        caption = getattr(backbone_wrapper, "cached_caption", None)
        t2 = getattr(backbone_wrapper, "cached_t2", None)
        label_features = getattr(backbone_wrapper, "cached_label_features", None)
        frame_label_ids = getattr(backbone_wrapper, "cached_frame_label_ids", None)

        # SSA 独立 forward
        optimizer_ssa.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            _, SSA_loss = backbone_wrapper.SSA(caption, t2, label_features, frame_label_ids)

        SSA_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.module.backbone.SSA.parameters(), clip_grad_l2norm)
        optimizer_ssa.step()

        # # SSA backward
        # maybe_ddp_model = model
        # from torch.nn.parallel import DistributedDataParallel as DDP
        # is_ddp = isinstance(maybe_ddp_model, DDP)
        #
        # if is_ddp:
        #     ctx = maybe_ddp_model.no_sync()
        # else:
        #     # 非 DDP 时 no-op 上下文
        #     from contextlib import nullcontext
        #     ctx = nullcontext()
        #
        # with ctx:
        #     if use_amp:
        #         scaler_ssa.scale(SSA_loss).backward()
        #         if clip_grad_l2norm > 0:
        #             scaler_ssa.unscale_(optimizer_ssa)
        #             torch.nn.utils.clip_grad_norm_(model.module.backbone.SSA.parameters(), clip_grad_l2norm)
        #         scaler_ssa.step(optimizer_ssa)
        #     else:
        #         SSA_loss.backward()
        #         if clip_grad_l2norm > 0:
        #             torch.nn.utils.clip_grad_norm_(model.module.backbone.SSA.parameters(), clip_grad_l2norm)
        #         optimizer_ssa.step()
            # print(f"[DEBUG] SSA_loss.grad_fn={SSA_loss.grad_fn}, loss_det.grad_fn={loss_det.grad_fn}")

        # update ema
        if model_ema is not None:
            model_ema.update(model)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # printing each logging_interval
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or ((iter_idx + 1) == num_iters):
            # print to terminal
            block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(curr_epoch, iter_idx, num_iters - 1)
            block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
            block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
            block4 = "lr_det={:.1e}".format(curr_det_lr)
            if curr_backbone_lr is not None:
                block4 = "lr_backbone={:.1e}".format(curr_backbone_lr) + "  " + block4
            block5 = "mem={:.0f}MB".format(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
            logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))

    # # forward pass
        # with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):        # amp:混合精度训练，加速训练节省显存
        #     losses = model(**data_dict, n_class=n_class, return_loss=True)
        #
        # optimizer.zero_grad()
        # optimizer_ssa.zero_grad(set_to_none=True)
        # # compute the gradients
        # #SSA
        # if "infonce_loss" in losses:
        #     loss_ssa = losses["infonce_loss"]
        #
        #     if use_amp:
        #         scaler.scale(loss_ssa).backward(retain_graph=True)
        #     else:
        #         loss_ssa.backward(retain_graph=True)
        #
        # # 原检测优化器
        # if use_amp:
        #     scaler.scale(losses["cost"]).backward()
        # else:
        #     losses["cost"].backward()
        #
        # if clip_grad_l2norm > 0.0:
        #     if loss_ssa is not None and optimizer_ssa is not None:
        #         if use_amp: scaler.unscale_(optimizer_ssa)
        #         torch.nn.utils.clip_grad_norm_(model.module.backbone.SSA.parameters(), clip_grad_l2norm)
        #
        #     if use_amp: scaler.unscale_(optimizer)
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
        #
        # if use_amp:
        #     # === AMP 模式 ===
        #     if optimizer_ssa is not None and loss_ssa is not None:
        #         scaler.step(optimizer_ssa)
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        #     # === FP32 模式 ===
        #     if optimizer_ssa is not None and loss_ssa is not None:
        #         optimizer_ssa.step()
        #     optimizer.step()
        #
        # # === 调试 align 梯度 ===
        # if hasattr(model.module.backbone, "SSA"):
        #     for n, p in model.module.backbone.SSA.align.named_parameters():
        #         if p.grad is not None:
        #             print(
        #                 f"[Iter {iter_idx:05d}] align grad mean={p.grad.abs().mean().item():.6e}, "
        #                 f"max={p.grad.abs().max().item():.6e}"
        #             )
        #         else:
        #             print(f"[Iter {iter_idx:05d}] align grad None")
        #
        # # 检查SSA梯度和GIT梯度，后续可注释
        # if (iter_idx + 1) % 200 == 0 and hasattr(model.module.backbone, "SSA"):
        #     ssa_grad = torch.nn.utils.clip_grad_norm_(
        #         model.module.backbone.SSA.parameters(), 1.0
        #     )
        #     # git_grad = sum(
        #     #     p.grad.abs().sum()
        #     #     for p in model.module.backbone.git_backbone.parameters()
        #     #     if p.grad is not None
        #     # )
        #     print(f"[DEBUG] iter {iter_idx + 1}: SSA grad={ssa_grad:.3f}")
        #
        # # update scheduler
        # scheduler.step()
        #
        # # update ema
        # if model_ema is not None:
        #     model_ema.update(model)
        #
        # # track all losses
        # losses = reduce_loss(losses)  # only for log
        # for key, value in losses.items():
        #     if key not in losses_tracker:
        #         losses_tracker[key] = AverageMeter()
        #     losses_tracker[key].update(value.item())
        #
        # # printing each logging_interval
        # if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or ((iter_idx + 1) == num_iters):
        #     # print to terminal
        #     block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(curr_epoch, iter_idx, num_iters - 1)
        #     block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
        #
        #     infonce_avg = losses_tracker["infonce_loss"].avg
        #     block2 += "  SSA={:.4f}".format(infonce_avg)
        #     # infonce_avg = losses_tracker["infonce_loss"].avg if "infonce_loss" in losses_tracker else 0.0  # for debug
        #     # origin_loss_val = losses_tracker["cost"].avg - infonce_avg
        #     # block2 += "  origin_Loss={:.4f}".format(origin_loss_val)
        #
        #     block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
        #     block4 = "lr_det={:.1e}".format(curr_det_lr)
        #     if curr_backbone_lr is not None:
        #         block4 = "lr_backbone={:.1e}".format(curr_backbone_lr) + "  " + block4
        #     block5 = "mem={:.0f}MB".format(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
        #     logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))


def val_one_epoch(
    val_loader,
    model,
    logger,
    rank,
    curr_epoch,
    model_ema=None,
    use_amp=False,
):
    """Validating the model for one epoch: compute the loss"""

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    logger.info("[Val]: Epoch {:d} Loss".format(curr_epoch))
    losses_tracker = {}

    model.eval()
    for data_dict in tqdm.tqdm(val_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                losses = model(**data_dict, return_loss=True)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

    # print to terminal
    block1 = "[Val]: [{:03d}]".format(curr_epoch)
    block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
    block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
    logger.info("  ".join([block1, block2, "  ".join(block3)]))

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)
    return losses_tracker["cost"].avg
