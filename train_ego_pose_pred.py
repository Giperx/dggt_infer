import os
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf

from dggt.models.vggt import VGGT
from dggt.utils.pose_encoding import encode_pose, decode_pose
from dggt.utils.geometry import se3_to_relative_pose_error, calculate_auc_np
from datasets.nuscenes.nuscenes_temporal_dataset import NuScenesTemporalMultiCamDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_ego_pose_pred_nuscenes.yaml')
    return parser.parse_args()


def load_ckpt_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        for key in ['state_dict', 'model', 'model_state_dict']:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    raise ValueError(f'Unsupported checkpoint format: {ckpt_path}')


def strip_prefix_from_state_dict(state_dict, prefix: str):
    new_state_dict = {}
    if prefix is None or prefix == '':
        return dict(state_dict)
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def strip_leading_dot_from_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[1:] if k.startswith('.') else k] = v
    return new_state_dict


def _filter_state_dict_by_shape(module, state_dict):
    module_state = module.state_dict()
    filtered = {}
    shape_mismatches = []

    for k, v in state_dict.items():
        if k not in module_state:
            continue
        if module_state[k].shape != v.shape:
            shape_mismatches.append((k, tuple(v.shape), tuple(module_state[k].shape)))
            continue
        filtered[k] = v

    return filtered, shape_mismatches


def _select_best_state_dict_for_module(module, raw_state_dict, user_prefix: str = ''):
    # Try several common checkpoint key layouts and select the one with the fewest mismatches.
    raw_no_dot = strip_leading_dot_from_state_dict(raw_state_dict)
    candidates = [
        ('as_is', dict(raw_state_dict)),
        ('strip_leading_dot', raw_no_dot),
        ('strip_user_prefix', strip_prefix_from_state_dict(raw_state_dict, user_prefix)),
        ('strip_user_prefix_after_dot', strip_prefix_from_state_dict(raw_no_dot, user_prefix)),
        ('strip_module', strip_prefix_from_state_dict(raw_state_dict, 'module.')),
        ('strip_module_after_dot', strip_prefix_from_state_dict(raw_no_dot, 'module.')),
        ('strip_ego_pose_head', strip_prefix_from_state_dict(raw_state_dict, 'ego_pose_head.')),
        ('strip_ego_pose_head_after_dot', strip_prefix_from_state_dict(raw_no_dot, 'ego_pose_head.')),
        ('strip_module_ego_pose_head', strip_prefix_from_state_dict(raw_state_dict, 'module.ego_pose_head.')),
        ('strip_module_ego_pose_head_after_dot', strip_prefix_from_state_dict(raw_no_dot, 'module.ego_pose_head.')),
    ]

    best_name = None
    best_state = None
    best_missing = None
    best_unexpected = None
    best_shape_mismatches = None
    best_score = None

    for name, cand in candidates:
        filtered_cand, shape_mismatches = _filter_state_dict_by_shape(module, cand)
        missing, unexpected = module.load_state_dict(filtered_cand, strict=False)
        score = len(missing) + len(unexpected)
        if best_score is None or score < best_score:
            best_name = name
            best_state = filtered_cand
            best_missing = missing
            best_unexpected = unexpected
            best_shape_mismatches = shape_mismatches
            best_score = score

    return best_state, best_name, best_missing, best_unexpected, best_shape_mismatches


def _log_state_dict_mismatch(log_fn, tag: str, missing_keys, unexpected_keys):
    if len(missing_keys) == 0 and len(unexpected_keys) == 0:
        log_fn(f"{tag} state_dict matched perfectly.")
        return

    log_fn(f"{tag} missing keys ({len(missing_keys)}):")
    for k in missing_keys:
        log_fn(f"  MISSING: {k}")

    log_fn(f"{tag} unexpected keys ({len(unexpected_keys)}):")
    for k in unexpected_keys:
        log_fn(f"  UNEXPECTED: {k}")


def _log_shape_mismatches(log_fn, tag: str, shape_mismatches, max_lines: int = 30):
    if shape_mismatches is None or len(shape_mismatches) == 0:
        return
    log_fn(f"{tag} shape mismatches ({len(shape_mismatches)}):")
    for idx, (k, ckpt_shape, model_shape) in enumerate(shape_mismatches):
        if idx >= max_lines:
            remaining = len(shape_mismatches) - max_lines
            if remaining > 0:
                log_fn(f"  ... ({remaining} more shape mismatches omitted)")
            break
        log_fn(f"  SHAPE_MISMATCH: {k} ckpt={ckpt_shape} model={model_shape}")


def _get_autocast_dtype(dtype_name: str):
    if dtype_name == 'float16':
        return torch.float16
    if dtype_name == 'bfloat16':
        return torch.bfloat16
    return torch.float16


# def resolve_model_module_name(cfg_name: str) -> str:
#     # Config keeps the user-facing name "dynamic_head", model module is instance_head.
#     if cfg_name == 'dynamic_head':
#         return 'instance_head'
#     return cfg_name


def load_init_weights(model: VGGT, cfg, local_rank: int, log_fn):
    init_cfg = cfg.model_init

    agg_state = load_ckpt_state_dict(init_cfg.aggregator_ckpt)
    ego_ckpt_path = init_cfg.get('ego_pose_head_ckpt', init_cfg.get('dynamic_head_ckpt', None))
    if ego_ckpt_path is None:
        raise ValueError('model_init.ego_pose_head_ckpt is required for ego pose training.')
    ego_state = load_ckpt_state_dict(ego_ckpt_path)

    agg_state = strip_prefix_from_state_dict(agg_state, init_cfg.aggregator_prefix)
    ego_prefix = init_cfg.get('ego_pose_head_prefix', init_cfg.get('dynamic_head_prefix', ''))

    missing_agg, unexpected_agg = model.aggregator.load_state_dict(agg_state, strict=False)
    ego_state_selected, ego_load_mode, missing_ego, unexpected_ego, ego_shape_mismatches = _select_best_state_dict_for_module(
        model.ego_pose_head,
        ego_state,
        user_prefix=ego_prefix,
    )
    # Re-apply selected state dict so final loaded weights are explicit and deterministic.
    missing_ego, unexpected_ego = model.ego_pose_head.load_state_dict(ego_state_selected, strict=False)

    if init_cfg.aggregator_to_bfloat16:
        model.aggregator = model.aggregator.to(torch.bfloat16)

    if local_rank == 0:
        log_fn(f"Loaded aggregator ckpt: {init_cfg.aggregator_ckpt}")
        log_fn(f"aggregator missing={len(missing_agg)}, unexpected={len(unexpected_agg)}")
        log_fn(f"Loaded ego_pose_head ckpt: {ego_ckpt_path}")
        log_fn(f"ego_pose_head load mode: {ego_load_mode}")
        log_fn(f"ego_pose_head missing={len(missing_ego)}, unexpected={len(unexpected_ego)}")
        _log_state_dict_mismatch(log_fn, 'ego_pose_head', missing_ego, unexpected_ego)
        _log_shape_mismatches(log_fn, 'ego_pose_head', ego_shape_mismatches)

    del agg_state, ego_state


def build_optimizer(model, cfg):
    trainable_modules = list(cfg.train.trainable_modules)
    param_groups = []

    for module_name_cfg in trainable_modules:
        # module_name = resolve_model_module_name(module_name_cfg)
        module = getattr(model.module, module_name_cfg)
        lr_map = cfg.train.lr_per_module
        lr = lr_map.get(module_name_cfg, cfg.train.lr)
        param_groups.append({'params': module.parameters(), 'lr': float(lr)})

    if len(param_groups) == 0:
        raise ValueError('No trainable modules configured.')

    return AdamW(param_groups, weight_decay=float(cfg.train.weight_decay))


def check_and_fix_inf_nan(x: torch.Tensor, name: str):
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def ego_pose_loss_single(pred_pose_enc, gt_pose_enc, T_loss_type='l1', R_loss_type='l1'):
    if T_loss_type == 'l1':
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
    elif T_loss_type == 'l2':
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, p=2, keepdim=True)
    elif 'huber' in T_loss_type:
        beta = float(T_loss_type.split('_')[1])
        loss_T = F.smooth_l1_loss(pred_pose_enc[..., :3], gt_pose_enc[..., :3], reduction='none', beta=beta)
    else:
        raise ValueError(f'Unknown T_loss_type: {T_loss_type}')

    if R_loss_type == 'l1':
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
    elif R_loss_type == 'l2':
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1, p=2)
    else:
        raise ValueError(f'Unknown R_loss_type: {R_loss_type}')

    loss_T = check_and_fix_inf_nan(loss_T, 'loss_T').clamp(max=100).mean()
    loss_R = check_and_fix_inf_nan(loss_R, 'loss_R').mean()
    return loss_T, loss_R


def compute_ego_pose_loss(
    pred_pose_encodings,
    gt_ego_pose,
    T_loss_type='l1',
    R_loss_type='l1',
    gamma=0.6,
    pose_encoding_type='absT_quaR',
    prefix='absolute_',
):
    gt_pose_encoding = encode_pose(gt_ego_pose, pose_encoding_type=pose_encoding_type)
    n_stages = len(pred_pose_encodings)

    total_loss_T = 0.0
    total_loss_R = 0.0

    for stage_idx, pred_pose_stage in enumerate(pred_pose_encodings):
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        loss_T_stage, loss_R_stage = ego_pose_loss_single(
            pred_pose_stage,
            gt_pose_encoding,
            T_loss_type=T_loss_type,
            R_loss_type=R_loss_type,
        )
        total_loss_T = total_loss_T + loss_T_stage * stage_weight
        total_loss_R = total_loss_R + loss_R_stage * stage_weight

    avg_loss_T = total_loss_T / max(n_stages, 1)
    avg_loss_R = total_loss_R / max(n_stages, 1)
    return {
        f'loss_{prefix}ego_pose_T': avg_loss_T,
        f'loss_{prefix}ego_pose_R': avg_loss_R,
    }


def compute_absolute_ego_pose_loss(predictions, gt_ego_pose, cfg):
    pred_pose_stages = [pose_enc[:, 1:] for pose_enc in predictions['absolute_ego_pose_enc_list']]
    gt_pose = gt_ego_pose[:, 1:]

    loss_dict = compute_ego_pose_loss(
        pred_pose_encodings=pred_pose_stages,
        gt_ego_pose=gt_pose,
        prefix='absolute_',
        T_loss_type=str(cfg.loss.get('T_loss_type', 'l1')),
        R_loss_type=str(cfg.loss.get('R_loss_type', 'l1')),
        gamma=float(cfg.loss.get('stage_gamma', 0.6)),
        pose_encoding_type=str(cfg.loss.get('pose_encoding_type', 'absT_quaR')),
    )

    total_loss = (
        float(cfg.loss.get('absolute_ego_pose_weight', 1.0))
        * (loss_dict['loss_absolute_ego_pose_T'] + loss_dict['loss_absolute_ego_pose_R'])
    )
    return loss_dict, total_loss


def compute_pose_metrics(pred_ego_pose_enc, gt_ego_pose):
    r_error = []
    t_error = []

    batch_size = pred_ego_pose_enc.shape[0]
    for batch_idx in range(batch_size):
        pred_se3, _ = decode_pose(
            pred_ego_pose_enc[batch_idx : batch_idx + 1],
            pose_encoding_type='absT_quaR',
            build_intrinsics=False,
        )
        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
            pred_se3[0],
            gt_ego_pose[batch_idx],
            3,
        )
        r_error.extend(rel_rangle_deg.detach().cpu().numpy().reshape(-1).tolist())
        t_error.extend(rel_tangle_deg.detach().cpu().numpy().reshape(-1).tolist())

    r_error = np.asarray(r_error, dtype=np.float32)
    t_error = np.asarray(t_error, dtype=np.float32)

    aucs = {}
    for threshold in [5, 10, 20, 30]:
        auc_value, _ = calculate_auc_np(r_error, t_error, max_threshold=threshold)
        aucs[threshold] = float(auc_value)

    return {
        'r_error': r_error,
        't_error': t_error,
        'rot_mean': float(rel_rangle_deg.mean().item()),
        'trans_mean': float(rel_tangle_deg.mean().item()),
        'aucs': aucs,
    }


def main(args):
    cfg = OmegaConf.load(args.config)

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    base_log_dir = cfg.log.log_dir
    run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if local_rank == 0 else None
    obj_list = [run_name]
    dist.broadcast_object_list(obj_list, src=0)
    run_name = obj_list[0]

    log_dir = os.path.join(base_log_dir, run_name)
    ckpt_dir = os.path.join(log_dir, 'ckpt')
    tb_dir = os.path.join(log_dir, cfg.log.tensorboard_subdir)
    debug_vis_dir = os.path.join(log_dir, 'debug_vis')
    log_txt_path = os.path.join(log_dir, 'train.log')

    writer = None
    log_file = None
    if local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(debug_vis_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        OmegaConf.save(cfg, os.path.join(log_dir, 'config_resolved.yaml'))
        log_file = open(log_txt_path, 'a', encoding='utf-8')

    def log_fn(msg: str):
        if local_rank == 0:
            print(msg)
            if log_file is not None:
                log_file.write(msg + '\n')
                log_file.flush()

    dataset = NuScenesTemporalMultiCamDataset(
        root_dir=cfg.data.image_dir,
        scene_list_file=cfg.data.train_scene_list,
        sequence_length=int(cfg.data.sequence_length),
        num_cameras_per_frame=int(cfg.data.num_cameras_per_frame),
        max_interval=int(cfg.data.max_interval),
        target_size=int(cfg.data.target_size),
        is_val=False,
        multiframe_forward_order=bool(cfg.data.get('multiframe_forward_order', False)),
    )
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        sampler=sampler,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
    )

    val_dataset = NuScenesTemporalMultiCamDataset(
        root_dir=cfg.data.image_dir,
        scene_list_file=cfg.data.val_scene_list,
        sequence_length=int(cfg.data.sequence_length),
        num_cameras_per_frame=int(cfg.data.num_cameras_per_frame),
        max_interval=int(cfg.data.max_interval),
        target_size=int(cfg.data.target_size),
        is_val=True,
        multiframe_forward_order=bool(cfg.data.get('multiframe_forward_order', False)),
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(cfg.train.batch_size),
        sampler=val_sampler,
        num_workers=int(cfg.train.val_num_workers),
        pin_memory=True,
    )

    max_epoch = int(cfg.train.max_epoch)
    steps_per_epoch = len(dataloader)
    max_steps = max_epoch * steps_per_epoch
    progress_steps_per_epoch = int(cfg.train.get('progress_steps_per_epoch', steps_per_epoch))
    progress_total_steps = max_epoch * progress_steps_per_epoch

    train_stats = dataset.get_dataset_stats()
    val_stats = val_dataset.get_dataset_stats()

    if local_rank == 0:
        log_fn('===== Dataset/Training Summary =====')
        log_fn(f"train scenes={train_stats['num_scenes']}, train sequences={train_stats['num_sequences']}")
        log_fn(f"val scenes={val_stats['num_scenes']} (fixed first sequence, interval=3)")
        log_fn(f"world_size={world_size}, batch_size_per_gpu={int(cfg.train.batch_size)}")
        log_fn(f"steps_per_epoch={steps_per_epoch}, max_epoch={max_epoch}, max_steps={max_steps}")
        log_fn(f"progress_steps_per_epoch={progress_steps_per_epoch}, progress_total_steps={progress_total_steps}")

    val_every_steps = int(cfg.train.get('val_every_steps', int(cfg.train.val_every) * max(steps_per_epoch, 1)))
    save_ckpt_every_steps = int(
        cfg.train.get('save_ckpt_every_steps', int(cfg.train.save_ckpt_every) * max(steps_per_epoch, 1))
    )
    cal_flag = bool(cfg.log.get('cal_flag', False))
    debug_cal_every_steps = int(cfg.log.get('debug_cal_every_steps', 0))
    if local_rank == 0:
        log_fn(
            f"val_every_steps={val_every_steps}, save_ckpt_every_steps={save_ckpt_every_steps}, "
            f"cal_flag={cal_flag}, debug_cal_every_steps={debug_cal_every_steps}"
        )

    model = VGGT(useDynamicHead=False, useCameraHead=cfg.model_init.use_camera_head_for_ego_pose).to(device)
    load_init_weights(model, cfg, local_rank, log_fn)

    model.train()
    model = DDP(model, device_ids=[local_rank])
    model._set_static_graph()

    for param in model.module.parameters():
        param.requires_grad = False
    for module_name_cfg in cfg.train.trainable_modules:
        # module_name = resolve_model_module_name(module_name_cfg)
        for param in getattr(model.module, module_name_cfg).parameters():
            param.requires_grad = True

    optimizer = build_optimizer(model, cfg)

    warmup_steps = int(cfg.train.warmup_iterations)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / max(warmup_steps, 1), 1.0) * 0.5 * (
            1 + torch.cos(torch.tensor(torch.pi * step / max(max_steps, 1)))
        ),
    )

    amp_enabled = bool(cfg.train.use_amp)
    amp_dtype = _get_autocast_dtype(cfg.train.amp_dtype)

    global_step = 0
    last_ckpt_global_step = -1
    pbar = tqdm(total=progress_total_steps, dynamic_ncols=True, desc='train_steps') if local_rank == 0 else None

    def run_validation(cur_global_step: int, cur_epoch_idx: int):
        model.eval()
        val_sampler.set_epoch(cur_epoch_idx)
        val_loss_sum_local = torch.zeros(1, device=device)
        val_loss_T_sum_local = torch.zeros(1, device=device)
        val_loss_R_sum_local = torch.zeros(1, device=device)
        val_count_local = torch.zeros(1, device=device)
        val_pose_metric_result = None
        should_cal_pose_metric = cal_flag and (
            cur_global_step == 0 or (debug_cal_every_steps > 0 and cur_global_step % debug_cal_every_steps == 0)
        )

        with torch.no_grad():
            for batch_idx, val_batch in enumerate(val_dataloader):
                val_images = val_batch['images'].to(device)
                val_gt_ego_pose = val_batch['ego_n_to_ego_first'].to(device)

                val_predictions = model(val_images)
                val_loss_dict, val_loss = compute_absolute_ego_pose_loss(val_predictions, val_gt_ego_pose, cfg)

                val_loss_sum_local += val_loss.detach()
                val_loss_T_sum_local += val_loss_dict['loss_absolute_ego_pose_T'].detach()
                val_loss_R_sum_local += val_loss_dict['loss_absolute_ego_pose_R'].detach()
                val_count_local += 1

                if local_rank == 0 and should_cal_pose_metric and batch_idx == 0:
                    val_pose_metric_result = compute_pose_metrics(
                        val_predictions['absolute_ego_pose_enc'].detach().float(),
                        val_gt_ego_pose.detach().float(),
                    )

        dist.all_reduce(val_loss_sum_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_T_sum_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_R_sum_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count_local, op=dist.ReduceOp.SUM)
        avg_val_loss = (val_loss_sum_local / torch.clamp(val_count_local, min=1.0)).item()
        avg_val_loss_T = (val_loss_T_sum_local / torch.clamp(val_count_local, min=1.0)).item()
        avg_val_loss_R = (val_loss_R_sum_local / torch.clamp(val_count_local, min=1.0)).item()

        if local_rank == 0:
            lr_now = float(scheduler.get_last_lr()[0])
            if should_cal_pose_metric and val_pose_metric_result is not None:
                log_fn(
                    f"[epoch={cur_epoch_idx}/{max_epoch}] [step={cur_global_step}/{max_steps}] "
                    f"Val Loss: {avg_val_loss:.4f} (T={avg_val_loss_T:.4f}, R={avg_val_loss_R:.4f}) | "
                    f"Rot Error: {val_pose_metric_result['rot_mean']:.4f} | Trans Error: {val_pose_metric_result['trans_mean']:.4f} | "
                    f"AUC@5/10/20/30: {val_pose_metric_result['aucs'][5]:.4f} / {val_pose_metric_result['aucs'][10]:.4f} / {val_pose_metric_result['aucs'][20]:.4f} / {val_pose_metric_result['aucs'][30]:.4f} | "
                    f"LR: {lr_now:.6e}"
                )
            else:
                log_fn(
                    f"[epoch={cur_epoch_idx}/{max_epoch}] [step={cur_global_step}/{max_steps}] "
                    f"Val Loss: {avg_val_loss:.4f} (T={avg_val_loss_T:.4f}, R={avg_val_loss_R:.4f}) | LR: {lr_now:.6e}"
                )
            if writer is not None:
                writer.add_scalar('loss/val', avg_val_loss, cur_global_step)
                writer.add_scalar('loss/val_T', avg_val_loss_T, cur_global_step)
                writer.add_scalar('loss/val_R', avg_val_loss_R, cur_global_step)
                if should_cal_pose_metric and val_pose_metric_result is not None:
                    writer.add_scalar('metric/val_rot_error', val_pose_metric_result['rot_mean'], cur_global_step)
                    writer.add_scalar('metric/val_trans_error', val_pose_metric_result['trans_mean'], cur_global_step)
                    for threshold, auc_value in val_pose_metric_result['aucs'].items():
                        writer.add_scalar(f'metric/val_auc_{threshold}', auc_value, cur_global_step)

        model.train()

    run_val_at_start = bool(cfg.train.get('val_at_start', True))
    if run_val_at_start and val_every_steps > 0 and global_step % val_every_steps == 0:
        run_validation(global_step, 0)

    for epoch_idx in range(max_epoch):
        sampler.set_epoch(epoch_idx)

        train_loss_sum_local = torch.zeros(1, device=device)
        train_loss_T_sum_local = torch.zeros(1, device=device)
        train_loss_R_sum_local = torch.zeros(1, device=device)
        train_count_local = torch.zeros(1, device=device)

        for batch in dataloader:
            images = batch['images'].to(device)
            gt_ego_pose = batch['ego_n_to_ego_first'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                predictions = model(images)
                loss_dict, loss = compute_absolute_ego_pose_loss(predictions, gt_ego_pose, cfg)

            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            train_loss_sum_local += loss.detach()
            train_loss_T_sum_local += loss_dict['loss_absolute_ego_pose_T'].detach()
            train_loss_R_sum_local += loss_dict['loss_absolute_ego_pose_R'].detach()
            train_count_local += 1

            if pbar is not None and pbar.n < progress_total_steps:
                pbar.update(1)

            should_run_val = (val_every_steps > 0 and global_step % val_every_steps == 0)
            if should_run_val:
                run_validation(global_step, epoch_idx)

            should_save_ckpt = (
                local_rank == 0
                and save_ckpt_every_steps > 0
                and global_step % save_ckpt_every_steps == 0
                and global_step != last_ckpt_global_step
            )
            if should_save_ckpt:
                ckpt_path = os.path.join(ckpt_dir, f'model_step_{global_step}.pt')
                torch.save(model.module.state_dict(), ckpt_path)
                log_fn(f"[Checkpoint] Saved model at global_step {global_step} to {ckpt_path}")
                last_ckpt_global_step = global_step

        dist.all_reduce(train_loss_sum_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss_T_sum_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss_R_sum_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_count_local, op=dist.ReduceOp.SUM)
        avg_train_loss = (train_loss_sum_local / torch.clamp(train_count_local, min=1.0)).item()
        avg_train_loss_T = (train_loss_T_sum_local / torch.clamp(train_count_local, min=1.0)).item()
        avg_train_loss_R = (train_loss_R_sum_local / torch.clamp(train_count_local, min=1.0)).item()

        if local_rank == 0:
            lr_now = float(scheduler.get_last_lr()[0])
            log_fn(
                f"[epoch={epoch_idx}/{max_epoch}] [step={global_step}/{max_steps}] "
                f"Train Loss: {avg_train_loss:.4f} (T={avg_train_loss_T:.4f}, R={avg_train_loss_R:.4f}) | LR: {lr_now:.6e}"
            )
            if writer is not None:
                writer.add_scalar('loss/train', avg_train_loss, global_step)
                writer.add_scalar('loss/train_T', avg_train_loss_T, global_step)
                writer.add_scalar('loss/train_R', avg_train_loss_R, global_step)
                writer.add_scalar('lr', lr_now, global_step)

    if local_rank == 0:
        final_ckpt = os.path.join(ckpt_dir, f'model_final_step_{global_step}.pt')
        torch.save(model.module.state_dict(), final_ckpt)
        log_fn(f"[Checkpoint] Saved final model to {final_ckpt}")
        if pbar is not None:
            pbar.close()

    if writer is not None:
        writer.close()
    if log_file is not None:
        log_file.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
