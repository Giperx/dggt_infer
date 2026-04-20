import os
import argparse
from datetime import datetime

import torch
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf

from dggt.models.vggt import VGGT
from datasets.nuscenes.nuscenes_temporal_dataset import NuScenesTemporalMultiCamDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_dynamic_nuscenes.yaml')
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
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


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
    dyn_state = load_ckpt_state_dict(init_cfg.dynamic_head_ckpt)

    agg_state = strip_prefix_from_state_dict(agg_state, init_cfg.aggregator_prefix)
    dyn_state = strip_prefix_from_state_dict(dyn_state, init_cfg.dynamic_head_prefix)

    missing_agg, unexpected_agg = model.aggregator.load_state_dict(agg_state, strict=False)
    missing_dyn, unexpected_dyn = model.dynamic_head.load_state_dict(dyn_state, strict=False)

    if init_cfg.aggregator_to_bfloat16:
        model.aggregator = model.aggregator.to(torch.bfloat16)

    if local_rank == 0:
        log_fn(f"Loaded aggregator ckpt: {init_cfg.aggregator_ckpt}")
        log_fn(f"aggregator missing={len(missing_agg)}, unexpected={len(unexpected_agg)}")
        log_fn(f"Loaded dynamic_head ckpt: {init_cfg.dynamic_head_ckpt}")
        log_fn(f"dynamic_head(instance_head) missing={len(missing_dyn)}, unexpected={len(unexpected_dyn)}")

    del agg_state, dyn_state


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


def save_debug_visualizations(global_step: int, epoch_idx: int, batch, images, dynamic_masks, dy_map, debug_vis_root: str):
    # One folder per visualization trigger (epoch + global step).
    epoch_vis_dir = os.path.join(debug_vis_root, f"epoch_{epoch_idx:06d}_step_{global_step:08d}")
    os.makedirs(epoch_vis_dir, exist_ok=True)

    batch_size = images.shape[0]
    dy_map_sigmoid = torch.sigmoid(dy_map)

    scene_names = batch.get('scene_name', ['unknown'] * batch_size)
    view_camera_ids = batch.get('view_camera_ids', None)

    for b in range(batch_size):
        scene_name = scene_names[b] if isinstance(scene_names, list) else str(scene_names)
        num_views = images.shape[1]

        for v in range(num_views):
            gt_image = images[b, v].detach().cpu().clamp(0, 1)
            gt_mask = dynamic_masks[b, v].detach().cpu().clamp(0, 1)
            pred_mask = dy_map_sigmoid[b, v].detach().cpu().clamp(0, 1)

            gt_mask_rgb = gt_mask.unsqueeze(0).repeat(3, 1, 1)
            pred_mask_rgb = pred_mask.unsqueeze(0).repeat(3, 1, 1)
            combined = torch.cat([gt_image, gt_mask_rgb, pred_mask_rgb], dim=-1)

            cam_id = int(view_camera_ids[b, v].item()) if view_camera_ids is not None else v
            filename = f"{global_step}_{scene_name}_{cam_id}_v{v}.png"
            T.ToPILImage()(combined).save(os.path.join(epoch_vis_dir, filename))


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
    debug_vis_every_steps = int(cfg.log.get('debug_vis_every_steps', int(cfg.log.get('debug_vis_every', 0))))
    vis_flag = bool(cfg.log.get('vis_flag', False))

    if local_rank == 0:
        log_fn(f"val_every_steps={val_every_steps}, save_ckpt_every_steps={save_ckpt_every_steps}, debug_vis_every_steps={debug_vis_every_steps}, vis_flag={vis_flag}")

    model = VGGT().to(device)
    load_init_weights(model, cfg, local_rank, log_fn)

    model.train()
    model = DDP(model, device_ids=[local_rank])
    model._set_static_graph()

    binary_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

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
        val_count_local = torch.zeros(1, device=device)

        with torch.no_grad():
            for val_batch in val_dataloader:
                val_images = val_batch['images'].to(device)
                if 'dynamic_mask' in val_batch:
                    val_dynamic_masks = val_batch['dynamic_mask'].to(device)[:, :, 0, :, :]
                else:
                    val_dynamic_masks = None

                val_predictions = model(val_images)
                val_dy_map = val_predictions['dynamic_conf'].squeeze(-1)

                if val_dynamic_masks is not None:
                    val_dynamic_loss = binary_loss_fn(val_dy_map[0], val_dynamic_masks[0].float())
                else:
                    val_dynamic_loss = binary_loss_fn(val_dy_map[0], torch.zeros_like(val_dy_map[0]))

                val_loss_sum_local += val_dynamic_loss.detach()
                val_count_local += 1

                if (
                    local_rank == 0
                    and vis_flag
                    and debug_vis_every_steps > 0
                    and cur_global_step % debug_vis_every_steps == 0
                    and val_dynamic_masks is not None
                ):
                    save_debug_visualizations(
                        cur_global_step,
                        cur_epoch_idx,
                        val_batch,
                        val_images,
                        val_dynamic_masks,
                        val_dy_map,
                        debug_vis_dir,
                    )

        dist.all_reduce(val_loss_sum_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count_local, op=dist.ReduceOp.SUM)
        avg_val_loss = (val_loss_sum_local / torch.clamp(val_count_local, min=1.0)).item()

        if local_rank == 0:
            lr_now = float(scheduler.get_last_lr()[0])
            log_fn(f"[epoch={cur_epoch_idx}/{max_epoch}] [step={cur_global_step}/{max_steps}] Val Loss: {avg_val_loss:.4f} | LR: {lr_now:.6e}")
            if writer is not None:
                writer.add_scalar('loss/val', avg_val_loss, cur_global_step)

        model.train()

    run_val_at_start = bool(cfg.train.get('val_at_start', True))
    if run_val_at_start and val_every_steps > 0 and global_step % val_every_steps == 0:
        run_validation(global_step, 0)

    for epoch_idx in range(max_epoch):
        sampler.set_epoch(epoch_idx)

        train_loss_sum_local = torch.zeros(1, device=device)
        train_count_local = torch.zeros(1, device=device)

        for batch in dataloader:
            images = batch['images'].to(device)
            if 'dynamic_mask' in batch:
                dynamic_masks = batch['dynamic_mask'].to(device)[:, :, 0, :, :]
            else:
                dynamic_masks = None

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                predictions = model(images)
                dy_map = predictions['dynamic_conf'].squeeze(-1)

                if dynamic_masks is not None:
                    dynamic_loss = binary_loss_fn(dy_map[0], dynamic_masks[0].float())
                else:
                    dynamic_loss = binary_loss_fn(dy_map[0], torch.zeros_like(dy_map[0]))

                loss = float(cfg.loss.dynamic_weight) * dynamic_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            train_loss_sum_local += loss.detach()
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
        dist.all_reduce(train_count_local, op=dist.ReduceOp.SUM)
        avg_train_loss = (train_loss_sum_local / torch.clamp(train_count_local, min=1.0)).item()

        if local_rank == 0:
            lr_now = float(scheduler.get_last_lr()[0])
            log_fn(f"[epoch={epoch_idx}/{max_epoch}] [step={global_step}/{max_steps}] Train Loss: {avg_train_loss:.4f} | LR: {lr_now:.6e}")
            if writer is not None:
                writer.add_scalar('loss/train', avg_train_loss, global_step)
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
