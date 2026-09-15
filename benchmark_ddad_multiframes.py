"""
DDAD multi-frame inference timing / benchmark script for DGGT.

Based on inference_ddad_multiframes.py. Strips out all image-saving
logic and instead measures the inference latency from `model(images)` through
`process_images_with_difix` using torch.cuda.Event.

Ego car mask logic (DDAD):
  - DDAD's cams (cam5/cam4/cam3) ALL capture the ego-car region. Each cam
    has its own mask; DDAD uses per-scene paths at
    `<scene_dir>/ego_car_masks/<cam_id>.jpg`.
  - For every non-current-cam5 view, load that cam's ego car mask and use
    it to suppress Gaussians that fall in the ego-car region.
  - The current frame's cam5 (the rendered view) is forced to all-ones
    (keep ALL Gaussians) for downstream metric integrity.

Usage:
    # default warmup=50, measure=50
    python benchmark_ddad_multiframes.py

    # override
    WARMUP=20 MEASURE=100 python benchmark_ddad_multiframes.py

Prints mean / median / min / max / stdev latency in ms and mean FPS.
"""

import sys
_local = [p for p in sys.path if ".local" in p]
if _local:
    sys.path = [p for p in sys.path if ".local" not in p] + _local

import os
import time
import statistics
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TV
from PIL import Image
from tqdm import tqdm
from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_split_gs
from gsplat.rendering import rasterization
from third_party.difix.infer import process_images_with_difix

# ── Global config ──
DATA_DIR = "data/datasets/ddad_process/valid"
SCENE_LIST = "data/datasets/ddad_process/valid/valid.txt"
CKPT_PATH = "pretrained/model_latest_nuscenes.pt"
TARGET_W = 518
TARGET_H = 322
CAM_IDS = [5, 4, 3]
WIDE_FACTOR = 3
DIFIX_CKPT = "pretrained/model_difix.pkl"
SIMPLE_MERGE = True
USE_EGO_CAR_MASK = True
EGO_CAR_OPACITY_LOGIT = -10.0
MULTIFRAMES = 3

# ── Benchmark config (env-overridable) ──
WARMUP = int(os.environ.get("WARMUP", "50"))
MEASURE = int(os.environ.get("MEASURE", "50"))


def alpha_t(t, t0, alpha, gamma0=1, gamma1=0.1):
    """Time-decayed opacity: farther-in-time Gaussians fade out."""
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / (gamma0**2 + 1e-6)
    conf = torch.exp(sigma * (t0 - t) ** 2)
    alpha_ = alpha * conf
    return alpha_.float()


def load_image(path, target_h, target_w):
    img = Image.open(path).convert("RGB")
    img = img.resize((target_w, target_h), Image.BILINEAR)
    tensor = TV.ToTensor()(img)
    return tensor


def load_sky_mask(path, target_h, target_w):
    mask = Image.open(path).convert("L")
    mask = mask.resize((target_w, target_h), Image.NEAREST)
    mask_np = np.array(mask)
    bg_mask = torch.from_numpy(mask_np == 0)
    return bg_mask


def load_ego_car_mask(path, target_h, target_w):
    mask = Image.open(path).convert("L")
    mask = mask.resize((target_w, target_h), Image.NEAREST)
    mask_np = np.array(mask)
    keep_mask = torch.from_numpy(mask_np >= 128)
    return keep_mask


def run_inference(model, difix_model, images, bg_masks, ego_car_masks, timestamps,
                  S, T, ST, device, dtype):
    """Run the full inference pipeline:
        model(images) → decode extrinsics/intrinsics → depth/point/gs/dy
        → static/dynamic separation → ego car suppression → dynamic GS
        → modify intrinsics → alpha_t → rasterize → sky model
        → alpha composite → .detach().cpu().clamp(0,1) → process_images_with_difix

    No image saving. Returns (rendered, (wide_W, H)).
    """
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)
        H, W = images.shape[-2:]

        # Decode camera parameters
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"], (H, W)
        )
        extrinsic = extrinsics[0]  # [T*S, 3, 4]
        bottom = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device)
            .view(1, 1, 4)
            .expand(extrinsic.shape[0], 1, 4)
        )
        extrinsic = torch.cat([extrinsic, bottom], dim=1)  # [T*S, 4, 4]
        intrinsic = intrinsics[0]  # [T*S, 3, 3]

        # Depth -> 3D point map
        depth_map = predictions["depth"][0]  # [T*S, H, W, 1]
        point_map = unproject_depth_map_to_point_map(
            depth_map, extrinsics[0], intrinsics[0]
        )[None, ...]  # [1, T*S, H, W, 3]
        point_map = torch.from_numpy(point_map).to(device).float()

        gs_map = predictions["gs_map"]    # [1, T*S, H, W, 11]
        gs_conf = predictions["gs_conf"]  # [1, T*S, H, W, 1]
        dy_map = predictions["dynamic_conf"].squeeze(-1)  # [1, T*S, H, W]

        # ── Static / Dynamic separation ──
        if SIMPLE_MERGE:
            static_mask = bg_masks
            static_points = point_map[static_mask].reshape(-1, 3)
            static_rgbs, static_opacity, static_scales, static_rotations = \
                get_split_gs(gs_map, static_mask)
            static_gs_conf = gs_conf[static_mask]
            view_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
            gs_timestamps = timestamps[view_idx]
        else:
            static_mask = bg_masks & (dy_map < 0.5)
            static_points = point_map[static_mask].reshape(-1, 3)
            static_rgbs, static_opacity, static_scales, static_rotations = \
                get_split_gs(gs_map, static_mask)
            gs_dynamic_list = dy_map[static_mask].sigmoid()
            static_opacity = static_opacity * (1 - gs_dynamic_list)
            static_gs_conf = gs_conf[static_mask]
            view_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
            gs_timestamps = timestamps[view_idx]

        # ── Ego car mask suppression ──
        # DDAD's cams (cam5/cam4/cam3) all see the ego car. Apply each cam's
        # mask to Gaussians from every view EXCEPT the current frame's cam5
        # (the rendered view, which is all-1 to keep all GS for downstream
        # metric integrity).
        if USE_EGO_CAR_MASK:
            nonzero_indices = torch.nonzero(static_mask, as_tuple=False)
            view_indices = nonzero_indices[:, 1]
            h_indices = nonzero_indices[:, 2]
            w_indices = nonzero_indices[:, 3]
            current_cam5_view = T * S - S
            not_current_cam5 = view_indices != current_cam5_view
            if not_current_cam5.any():
                # view_index = t_idx * S + cam_idx  →  cam_idx = view_index % S
                cam_indices_in_list = view_indices[not_current_cam5] % S
                ego_keep = ego_car_masks[
                    0, cam_indices_in_list,
                    h_indices[not_current_cam5], w_indices[not_current_cam5]
                ]
                suppress = not_current_cam5.clone()
                suppress[not_current_cam5] = ~ego_keep
                static_opacity[suppress] = EGO_CAR_OPACITY_LOGIT

        # Dynamic Gaussians (per view)
        dynamic_points, dynamic_rgbs, dynamic_opacitys = [], [], []
        dynamic_scales, dynamic_rotations = [], []
        if not SIMPLE_MERGE:
            for i in range(ST):
                bg_mask_i = bg_masks[:, i]
                dynamic_point = point_map[:, i][bg_mask_i].reshape(-1, 3)
                dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = \
                    get_split_gs(gs_map[:, i], bg_mask_i)
                gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid()
                dynamic_opacity = dynamic_opacity * gs_dynamic_list_i
                dynamic_points.append(dynamic_point)
                dynamic_rgbs.append(dynamic_rgb)
                dynamic_opacitys.append(dynamic_opacity)
                dynamic_scales.append(dynamic_scale)
                dynamic_rotations.append(dynamic_rotation)

        # ── Modify intrinsics for wide-FOV rendering ──
        wide_W = W * WIDE_FACTOR
        intrinsic_wide = intrinsic.clone()
        render_view_idx = T * S - S
        intrinsic_wide[render_view_idx, 0, 2] = wide_W / 2.0

        # ── Rendering: only current frame's cam5 ──
        t0 = timestamps[render_view_idx]
        static_opacity_ = alpha_t(
            gs_timestamps, t0, static_opacity, gamma0=static_gs_conf
        )
        static_gs_list = [
            static_points, static_rgbs, static_opacity_,
            static_scales, static_rotations
        ]
        if SIMPLE_MERGE:
            world_points, rgbs, opacity, scales, rotation = static_gs_list
        else:
            if dynamic_points[render_view_idx].shape[0] > 0:
                world_points, rgbs, opacity, scales, rotation = concat_list(
                    static_gs_list,
                    [
                        dynamic_points[render_view_idx],
                        dynamic_rgbs[render_view_idx],
                        dynamic_opacitys[render_view_idx],
                        dynamic_scales[render_view_idx],
                        dynamic_rotations[render_view_idx],
                    ],
                )
            else:
                world_points, rgbs, opacity, scales, rotation = static_gs_list

        renders, alphas, _ = rasterization(
            means=world_points,
            quats=rotation,
            scales=scales,
            opacities=opacity,
            colors=rgbs,
            viewmats=extrinsic[render_view_idx : render_view_idx + 1],
            Ks=intrinsic_wide[render_view_idx : render_view_idx + 1],
            width=wide_W,
            height=H,
            render_mode="RGB+ED",
        )
        renders = renders[..., :-1]

        # Sky background
        bg_render = model.sky_model(
            images, extrinsic, intrinsic_wide,
            render_width=wide_W, render_height=H,
            sample_intrinsics=intrinsic,
        )
        bg_render = (
            (bg_render - bg_render.min())
            / (bg_render.max() - bg_render.min() + 1e-8)
        )

        # Alpha composite
        renders = alphas * renders + (1 - alphas) * bg_render
        rendered_image = renders.permute(0, 3, 1, 2)

    # Outside autocast: prepare for difix (CPU transfer)
    rendered = rendered_image[0].detach().cpu().clamp(0, 1)
    if DIFIX_CKPT:
        rendered = process_images_with_difix(rendered, DIFIX_CKPT, model=difix_model)
    return rendered, (wide_W, H)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmark.")

    # Read scene list (use first scene only)
    with open(SCENE_LIST) as f:
        scene_names = [line.strip() for line in f if line.strip()]
    scene_name = scene_names[0]
    print(f"Scene: {scene_name}")

    device = "cuda"
    dtype = torch.float32
    cam_ids = CAM_IDS
    S = len(cam_ids)
    T = MULTIFRAMES
    ST = S * T

    # Load model
    print(f"Loading model from {CKPT_PATH} ...")
    model = VGGT().to(device)
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    print("Model loaded.")

    # Load Difix model
    difix_model = None
    if DIFIX_CKPT:
        print(f"Loading Difix model from {DIFIX_CKPT} ...")
        from third_party.difix.src.model import Difix
        difix_model = Difix(
            pretrained_path=DIFIX_CKPT,
            timestep=199,
            mv_unet=False
        )
        difix_model.set_eval()
        print("Difix model loaded.")

    # Pick first frame with a valid T-frame window
    scene_dir = os.path.join(DATA_DIR, scene_name)
    images_dir = os.path.join(scene_dir, "images")
    masks_dir = os.path.join(scene_dir, "sky_masks")

    all_files = sorted(os.listdir(images_dir))
    frame_ids = sorted(set(f.split("_")[0] for f in all_files if f.endswith(".jpg")))
    if len(frame_ids) < T:
        raise RuntimeError(
            f"Scene '{scene_name}' has only {len(frame_ids)} frames, need at least {T}."
        )
    fi = T - 1
    frame_id = frame_ids[fi]

    # ── Load images, sky masks, ego car masks for T frames × S cameras ──
    images_list = []       # length T*S
    bg_masks_list = []     # length T*S
    ego_car_masks_list = [] # length T*S
    for t_idx in range(T):
        fid = frame_ids[fi - (T - 1 - t_idx)]
        for cam_idx, cam_id in enumerate(cam_ids):
            img_path = os.path.join(images_dir, f"{fid}_{cam_id}.jpg")
            mask_path = os.path.join(masks_dir, f"{fid}_{cam_id}.png")
            images_list.append(load_image(img_path, TARGET_H, TARGET_W))
            if os.path.exists(mask_path):
                bg_masks_list.append(load_sky_mask(mask_path, TARGET_H, TARGET_W))
            else:
                bg_masks_list.append(torch.ones(TARGET_H, TARGET_W, dtype=torch.bool))

            # DDAD's cams all see the ego car, so load cam_id's mask (per-scene
            # path) for every non-current-cam5 view (cam5 in older frames +
            # cam4 + cam3 across all frames). The current frame's cam5 (the
            # rendered view) stays all-1 to keep all GS for downstream metric
            # integrity.
            is_current_cam5 = (t_idx == T - 1) and (cam_id == 5)
            if USE_EGO_CAR_MASK and not is_current_cam5:
                ego_mask_path = os.path.join(
                    scene_dir, "ego_car_masks", f"{cam_id}.jpg"
                )
                if os.path.exists(ego_mask_path):
                    ego_car_masks_list.append(
                        load_ego_car_mask(ego_mask_path, TARGET_H, TARGET_W)
                    )
                else:
                    print(f"  [WARN] Ego car mask not found: {ego_mask_path}, keeping all")
                    ego_car_masks_list.append(
                        torch.ones(TARGET_H, TARGET_W, dtype=torch.bool)
                    )
            else:
                # current frame cam5 or feature disabled: keep all
                ego_car_masks_list.append(
                    torch.ones(TARGET_H, TARGET_W, dtype=torch.bool)
                )

    images = torch.stack(images_list).unsqueeze(0).to(device)
    bg_masks = torch.stack(bg_masks_list).unsqueeze(0).to(device)
    if USE_EGO_CAR_MASK:
        ego_car_masks = torch.stack(ego_car_masks_list).unsqueeze(0).to(device)
    timestamps = torch.arange(T, device=device).repeat_interleave(S).float()

    print(f"Frame {frame_id}: images={tuple(images.shape)}, bg_masks={tuple(bg_masks.shape)}")
    print(f"  Timestamps: {timestamps.tolist()}")
    print(f"  Warmup: {WARMUP}, Measure: {MEASURE}, Difix: {'on' if DIFIX_CKPT else 'off'}")

    # ── Warmup ──
    print(f"\n[1/2] Warmup ({WARMUP} iters)...")
    for _ in tqdm(range(WARMUP), desc="Warmup", leave=False):
        run_inference(model, difix_model, images, bg_masks, ego_car_masks,
                      timestamps, S, T, ST, device, dtype)

    # ── Measure ──
    print(f"[2/2] Measure ({MEASURE} iters)...")
    total_times = []
    output_shape = None
    for _ in tqdm(range(MEASURE), desc="Measure", leave=False):
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        total_start.record()
        _, shape = run_inference(model, difix_model, images, bg_masks, ego_car_masks,
                                  timestamps, S, T, ST, device, dtype)
        total_end.record()
        torch.cuda.synchronize()
        total_times.append(total_start.elapsed_time(total_end))
        output_shape = shape

    wide_W, H = output_shape

    # ── Report ──
    mean_ms = statistics.mean(total_times)
    median_ms = statistics.median(total_times)
    min_ms = min(total_times)
    max_ms = max(total_times)
    stdev_ms = statistics.stdev(total_times) if MEASURE > 1 else 0.0
    fps = 1000.0 / mean_ms

    print(f"\n{'=' * 64}")
    print(f" Benchmark Results — DDAD multi-frame inference")
    print(f"{'=' * 64}")
    print(f"  Scene:        {scene_name}")
    print(f"  Frame:        {frame_id}")
    print(f"  Input shape:  {tuple(images.shape)}  (T*S, C, H, W)")
    print(f"  Output:       1 x 3 x {H} x {wide_W}  (cam5 wide-FOV)")
    print(f"  Difix:        {'enabled' if DIFIX_CKPT else 'disabled'}")
    print(f"  Warmup iters: {WARMUP}")
    print(f"  Measure iters:{MEASURE}")
    print(f"{'─' * 64}")
    print(f"  Latency (ms): mean={mean_ms:8.2f}  median={median_ms:8.2f}  "
          f"min={min_ms:8.2f}  max={max_ms:8.2f}  stdev={stdev_ms:6.2f}")
    print(f"  Throughput:   {fps:8.2f} FPS")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
