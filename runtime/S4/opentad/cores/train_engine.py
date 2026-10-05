import copy
import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss

def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    model_ema=None,
    clip_grad_l2norm=-1,
    logging_interval=200,
    scaler=None,
):
    """Training the model for one epoch"""

    logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    num_iters = len(train_loader)
    use_amp = False if scaler is None else True

    model.train()

    for iter_idx, data_dict in enumerate(train_loader):
        group_lrs = scheduler.get_last_lr()

        n_class = len(train_loader.dataset.class_map)  # 生成类别矩阵用

        # import torch.distributed as dist  # ！
        # rank = dist.get_rank() if dist.is_initialized() else 0  # ！ debug 开始
        # names = data_dict.get("video_name", None)
        # if names is None and "metas" in data_dict:
        #     names = [m.get("video_name", "?") for m in data_dict["metas"]]
        # # frame_label_ids 是你传给 SSA 的那个 [B,T] 标注
        # if "frame_label_ids" in data_dict:
        #     act_sum = int((data_dict["frame_label_ids"] > 0).sum())
        # else:
        #     act_sum = -1  # 没在 data_dict 就先占位
        # print(f"[Rank {rank}] videos={names} | act_in_rank_batch={act_sum}")  # ！ debug 结束  定位真实错误样本！

        # forward pass
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):        # amp:混合精度训练，加速训练节省显存
            losses = model(**data_dict, n_class=n_class, return_loss=True, curr_epoch=curr_epoch)

        # # === Gradient Contribution 监控 (放在 backward 前) ===
        # ssa = getattr(model.module.backbone, "SSA", None)
        # if ssa is not None and hasattr(ssa, "loss_infonce"):
        #     loss_mod = ssa.loss_infonce
        #     loss_infonce = getattr(loss_mod, "last_infonce", None)
        #     loss_bg = getattr(loss_mod, "last_bg", None)
        #
        #     if (loss_infonce is not None) and (loss_bg is not None):
        #         # 你要看哪部分的参数：全 SSA、或只 linear_v / text_proj
        #         ssa_params_all = [p for p in ssa.parameters() if p.requires_grad]
        #         ssa_params_lv = [p for n, p in ssa.named_parameters() if p.requires_grad and "linear_v" in n]
        #         ssa_params_tp = [p for n, p in ssa.named_parameters() if p.requires_grad and "text_proj" in n]
        #
        #         def grad_l2_norm(loss_term, params):
        #             if len(params) == 0: return 0.0
        #             grads = torch.autograd.grad(loss_term, params, retain_graph=True, allow_unused=True)
        #             sq = [(g.float().norm() ** 2) for g in grads if g is not None]
        #             return float(torch.sqrt(torch.stack(sq).sum()).item()) if len(sq) else 0.0
        #
        #         # 计算两部分对梯度的贡献（注意此处不乘0.5、不乘lambda_bg，先看原始强度）
        #         g_inf_all = grad_l2_norm(loss_infonce, ssa_params_all)
        #         g_bg_all = grad_l2_norm(loss_bg, ssa_params_all)
        #
        #         g_inf_lv = grad_l2_norm(loss_infonce, ssa_params_lv)
        #         g_bg_lv = grad_l2_norm(loss_bg, ssa_params_lv)
        #
        #         g_inf_tp = grad_l2_norm(loss_infonce, ssa_params_tp)
        #         g_bg_tp = grad_l2_norm(loss_bg, ssa_params_tp)
        #
        #         if (iter_idx % logging_interval == 0) or (iter_idx + 1 == num_iters):
        #             print(f"[Grad || SSA]  inf/all={g_inf_all:.3e}   bg/all={g_bg_all:.3e}")
        #             print(f"[Grad || linV] inf={g_inf_lv:.3e}       bg={g_bg_lv:.3e}")
        #             print(f"[Grad || tProj]inf={g_inf_tp:.3e}       bg={g_bg_tp:.3e}")
        # # =======================================================

        # 原检测优化器
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(losses["cost"]).backward()
        else:
            losses["cost"].backward()

        # gradient clipping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            if use_amp:
                scaler.unscale_(optimizer)
                # print("3") #!
                # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
                # torch.cuda.synchronize()
                # print(f"[Rank {rank}] after clip, total_norm={float(total_norm):.6e}")
                # print("4") #!
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

        # update parameters
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # update scheduler
        scheduler.step()

        # update ema
        if model_ema is not None:
            model_ema.update(model)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # printing each logging_interval,for debug 打印统计信息
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or ((iter_idx + 1) == num_iters):
            # print to terminal
            block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(curr_epoch, iter_idx, num_iters - 1)
            block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)

            infonce_avg = losses_tracker["infonce_loss"].avg if "infonce_loss" in losses_tracker else 0.0  # for debug
            origin_loss_val = losses_tracker["cost"].avg - infonce_avg
            block2 += "  origin_Loss={:.4f}".format(origin_loss_val)

            block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
            block4 = "  ".join(
                "lr_{}={:.1e}".format(group.get("group_name", index), lr)
                for index, (group, lr) in enumerate(zip(optimizer.param_groups, group_lrs))
            )
            block5 = "mem={:.0f}MB".format(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
            logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))


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
