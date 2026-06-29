"""
nuScenes inference script for DGGT.
Loads 3 cameras (cam5, cam4, cam3) per frame, renders wide-FOV Gaussian Splatting images.
Mode 2 style: static/dynamic separation, sky background compositing.
Renders at 3x width for panoramic view, with optional Difix3D enhancement.
"""

import sys
# Prioritize conda env site-packages over user-site .local to avoid
# version conflicts (e.g. old Flask/Jinja2 in .local override newer conda versions)
_local = [p for p in sys.path if ".local" in p]
if _local:
    sys.path = [p for p in sys.path if ".local" not in p] + _local

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_split_gs
from gsplat.rendering import rasterization

# ── Global config ──
DATA_DIR = "data/datasets/nuscenes/processed_10Hz/trainval"
SCENE_LIST = "data/datasets/nuscenes/processed_10Hz_v2/nuScenes_Val2.txt"
CKPT_PATH = "pretrained/model_latest_waymo.pt" #model_latest_waymo model_latest_nuscenes
OUTPUT_PATH = "outputs_waymo_pt/nuscenes_inference"
TARGET_W = 518
TARGET_H = 294
CAM_IDS = [5, 4, 3]            # Input cameras for model (3 views)
RENDER_CAM_LIST = [5]          # Only save these cameras
WIDE_FACTOR = 3                # Render width = original_width * WIDE_FACTOR
MAX_FRAMES = -1                # Max frames per scene (-1 = all)
DIFIX_CKPT = "pretrained/model_difix.pkl"  # Set to path to enable Difix3D enhancement
DRY_RUN = False                # True = test data loading only, no model inference
BACKGROUND_COLOR = [1.0, 1.0, 1.0]  # White background for rendering
SIMPLE_MERGE = True           # True = merge all Gaussians with predicted opacity (no dyn weighting)


def alpha_t(t, t0, alpha, gamma0=1, gamma1=0.1):
    """Time-decayed opacity: farther-in-time Gaussians fade out."""
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / (gamma0**2 + 1e-6)
    conf = torch.exp(sigma * (t0 - t) ** 2)
    alpha_ = alpha * conf
    return alpha_.float()


def load_image(path, target_h, target_w):
    """Load image, resize, normalize to [0,1] float tensor [3, H, W]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((target_w, target_h), Image.BILINEAR)
    tensor = T.ToTensor()(img)  # [3, H, W], float32 [0,1]
    return tensor


def load_sky_mask(path, target_h, target_w):
    """Load sky mask, resize, return bool tensor [H, W]. True = not sky (foreground)."""
    mask = Image.open(path).convert("L")
    mask = mask.resize((target_w, target_h), Image.NEAREST)
    mask_np = np.array(mask)
    # 255 = sky, 0 = not sky
    # bg_mask: True where NOT sky
    bg_mask = torch.from_numpy(mask_np == 0)  # [H, W], bool
    return bg_mask


def main():
    # Read scene list
    with open(SCENE_LIST) as f:
        scene_names = [line.strip() for line in f if line.strip()]
    print(f"Scenes to process: {scene_names}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    cam_ids = CAM_IDS
    S = len(cam_ids)

    # Load model
    if not DRY_RUN:
        print(f"Loading model from {CKPT_PATH} ...")
        model = VGGT().to(device)
        checkpoint = torch.load(CKPT_PATH, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        print("Model loaded.")

    # Load Difix model once (if enabled)
    difix_model = None
    if DIFIX_CKPT:
        from third_party.difix.infer import process_images_with_difix
        print(f"Loading Difix model from {DIFIX_CKPT} ...")
        from third_party.difix.src.model import Difix
        # difix_model = load_difix_model(DIFIX_CKPT)
        difix_model = Difix(
            pretrained_path=DIFIX_CKPT,
            timestep=199,
            mv_unet=False
        )
        difix_model.set_eval()
        print("Difix model loaded.")

    scene_bar = tqdm(scene_names, desc="Scenes", unit="scene")
    for scene_name in scene_bar:
        scene_dir = os.path.join(DATA_DIR, scene_name)
        if not os.path.isdir(scene_dir):
            scene_bar.write(f"[WARN] Scene directory not found: {scene_dir}, skipping.")
            continue

        images_dir = os.path.join(scene_dir, "images")
        masks_dir = os.path.join(scene_dir, "sky_masks")

        # Determine frames
        all_files = sorted(os.listdir(images_dir))
        frame_ids = sorted(set(f.split("_")[0] for f in all_files if f.endswith(".jpg")))
        if MAX_FRAMES > 0:
            frame_ids = frame_ids[:MAX_FRAMES]

        # Output directories
        rgb_dir = os.path.join(OUTPUT_PATH, scene_name, "rgb")
        before_rgb_dir = os.path.join(OUTPUT_PATH, scene_name, "before_rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        if DIFIX_CKPT:
            os.makedirs(before_rgb_dir, exist_ok=True)

        frame_bar = tqdm(frame_ids, desc=f"Scene {scene_name}", unit="frame", leave=False)
        for fi, frame_id in enumerate(frame_bar):
            start_time = time.time()

            # Load images and sky masks for this frame, all cameras
            images_list = []
            bg_masks_list = []
            valid = True
            for cam_id in cam_ids:
                img_path = os.path.join(images_dir, f"{frame_id}_{cam_id}.jpg")
                mask_path = os.path.join(masks_dir, f"{frame_id}_{cam_id}.png")
                if not os.path.exists(img_path):
                    print(f"  [SKIP] Missing image: {img_path}")
                    valid = False
                    break
                images_list.append(load_image(img_path, TARGET_H, TARGET_W))
                if os.path.exists(mask_path):
                    bg_masks_list.append(load_sky_mask(mask_path, TARGET_H, TARGET_W))
                else:
                    bg_masks_list.append(torch.ones(TARGET_H, TARGET_W, dtype=torch.bool))

            if not valid:
                continue

            # Stack: images [1, S, 3, H, W], bg_masks [1, S, H, W]
            images = torch.stack(images_list).unsqueeze(0).to(device)
            bg_masks = torch.stack(bg_masks_list).unsqueeze(0).to(device)
            timestamps = torch.zeros(S, device=device)

            if DRY_RUN:
                print(f"  Frame {frame_id}: images={images.shape}, bg_masks={bg_masks.shape}")
                continue

            # ── Model forward ──
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                H, W = images.shape[-2:]

                # Decode camera parameters
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))
                extrinsic = extrinsics[0]  # [S, 3, 4]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device) \
                    .view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)  # [S, 4, 4]
                intrinsic = intrinsics[0]  # [S, 3, 3]

                # Depth -> 3D point map
                depth_map = predictions["depth"][0]  # [S, H, W, 1]
                point_map = unproject_depth_map_to_point_map(
                    depth_map, extrinsics[0], intrinsics[0]
                )[None, ...]  # [1, S, H, W, 3]
                point_map = torch.from_numpy(point_map).to(device).float()

                gs_map = predictions["gs_map"]   # [1, S, H, W, 11]
                gs_conf = predictions["gs_conf"] # [1, S, H, W, 1]
                dy_map = predictions["dynamic_conf"].squeeze(-1)  # [1, S, H, W]

                # ── Static / Dynamic separation (always) ──
                if SIMPLE_MERGE:
                    # Simple merge: use all Gaussians with predicted opacity (no dyn weighting)
                    static_mask = bg_masks
                    static_points = point_map[static_mask].reshape(-1, 3)
                    static_rgbs, static_opacity, static_scales, static_rotations = \
                        get_split_gs(gs_map, static_mask)
                    static_gs_conf = gs_conf[static_mask]
                    frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
                    gs_timestamps = timestamps[frame_idx]
                else:
                    # Original: static/dynamic separation with dyn weighting
                    static_mask = bg_masks & (dy_map < 0.5)
                    static_points = point_map[static_mask].reshape(-1, 3)
                    static_rgbs, static_opacity, static_scales, static_rotations = \
                        get_split_gs(gs_map, static_mask)
                    gs_dynamic_list = dy_map[static_mask].sigmoid()
                    static_opacity = static_opacity * (1 - gs_dynamic_list)
                    static_gs_conf = gs_conf[static_mask]
                    frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
                    gs_timestamps = timestamps[frame_idx]

                # Dynamic Gaussians (per view)
                dynamic_points, dynamic_rgbs, dynamic_opacitys = [], [], []
                dynamic_scales, dynamic_rotations = [], []
                if not SIMPLE_MERGE:
                    for i in range(S):
                        bg_mask_i = bg_masks[:, i]  # [1, H, W]
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
                # Model outputs pixel-space intrinsics: fx, fy, cx=W/2, cy=H/2
                # For wide-FOV: scale fx by 1/WIDE_FACTOR, shift cx to wide center
                intrinsic_wide = intrinsic.clone()
                # intrinsic_wide[:, 0, 0] /= WIDE_FACTOR    # fx: 400 -> 133.3
                intrinsic_wide[:, 0, 2] = wide_W / 2.0    # cx: 259 -> 777 (center of wide image)
                
                # fy and cy unchanged (height doesn't change)

                # ── Per-view rendering (at wide resolution) ──
                chunked_renders, chunked_alphas = [], []
                for idx in range(S):
                    t0 = timestamps[idx]
                    static_opacity_ = alpha_t(
                        gs_timestamps, t0, static_opacity, gamma0=static_gs_conf
                    )
                    static_gs_list = [
                        static_points, static_rgbs, static_opacity_,
                        static_scales, static_rotations
                    ]
                    if SIMPLE_MERGE:
                        # Simple merge: use only static Gaussians (all points)
                        world_points, rgbs, opacity, scales, rotation = static_gs_list
                    else:
                        # Original: combine static and dynamic Gaussians
                        if dynamic_points[idx].shape[0] > 0:
                            world_points, rgbs, opacity, scales, rotation = concat_list(
                                static_gs_list,
                                [
                                    dynamic_points[idx], dynamic_rgbs[idx],
                                    dynamic_opacitys[idx], dynamic_scales[idx],
                                    dynamic_rotations[idx]
                                ],
                            )
                        else:
                            world_points, rgbs, opacity, scales, rotation = static_gs_list

                    renders_chunk, alphas_chunk, _ = rasterization(
                        means=world_points,
                        quats=rotation,
                        scales=scales,
                        opacities=opacity,
                        colors=rgbs,
                        viewmats=extrinsic[idx : idx + 1],
                        Ks=intrinsic_wide[idx : idx + 1],
                        width=wide_W,
                        height=H,
                        render_mode="RGB+ED",
                    )
                    chunked_renders.append(renders_chunk)
                    chunked_alphas.append(alphas_chunk)

                renders = torch.cat(chunked_renders, dim=0)  # [S, H, wide_W, 4]
                renders = renders[..., :-1]  # drop depth, keep RGB
                alphas = torch.cat(chunked_alphas, dim=0)    # [S, H, wide_W, 1]

                # Sky background (at wide resolution)
                # sample with original intrinsics (matches 518-wide input images)
                # render with wide intrinsics (matches 1554-wide output)
                bg_render = model.sky_model(images, extrinsic, intrinsic_wide,
                                            render_width=wide_W, render_height=H,
                                            sample_intrinsics=intrinsic)
                bg_render = (bg_render - bg_render.min()) / (bg_render.max() - bg_render.min() + 1e-8)

                # Alpha composite
                renders = alphas * renders + (1 - alphas) * bg_render
                rendered_image = renders.permute(0, 3, 1, 2)  # [S, 3, H, wide_W]

            # ── Save results (only RENDER_CAM_LIST cameras) ──
            elapsed = time.time() - start_time
            frame_bar.set_postfix({"time": f"{elapsed:.1f}s"})
            for i, cam_id in enumerate(cam_ids):
                if cam_id not in RENDER_CAM_LIST:
                    continue
                rendered = rendered_image[i].detach().cpu().clamp(0, 1)

                if DIFIX_CKPT:
                    # Save pre-difix to before_rgb/
                    before_path = os.path.join(before_rgb_dir, f"{frame_id}_{cam_id}_wide.jpg")
                    T.ToPILImage()(rendered).save(before_path, quality=95)
                    # Apply Difix3D enhancement (reuse pre-loaded model)
                    rendered = process_images_with_difix(rendered, DIFIX_CKPT, model=difix_model)

                # Save to rgb/
                save_path = os.path.join(rgb_dir, f"{frame_id}_{cam_id}_wide.jpg")
                T.ToPILImage()(rendered.clamp(0, 1)).save(save_path, quality=95)

            # if fi % 10 == 0 or fi == len(frame_ids) - 1:
            #     print(f"  Frame {fi+1}/{len(frame_ids)} ({frame_id}): "
            #           f"wide {wide_W}x{H}, {elapsed:.2f}s")

    scene_bar.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
