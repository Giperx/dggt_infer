"""
WideDrive multi-frame inference script for DGGT.

Based on inference_widedrive.py, but loads MULTIFRAMES consecutive frames per
inference step so the model sees temporal context and the alpha_t time-decay
actually has meaning.

Input layout (MULTIFRAMES=3, CAM_IDS=[5,4,3]):
  Frame fi-2: cam5, cam4, cam3   (oldest,   t=0)
  Frame fi-1: cam5, cam4, cam3   (middle,   t=1)
  Frame fi  : cam5, cam4, cam3   (current/latest, t=2)  ← rendered

  Total S*T = 9 views fed to model.
  Only the current frame's cam5 (index T*S - S = 6) is rendered and saved.
  Camera 2 is the wide GT panorama and is not a model input.
  sky_masks/ is empty. WideDrive has no ego-car masks, so USE_EGO_CAR_MASK
  stays off and every view keeps all Gaussians.

Output:
  Per scene, images start from frame 002 (MULTIFRAMES-1) instead of 000.
  rgb/{frame}_5_wide.jpg
  before_rgb/{frame}_5_wide.jpg   # only when Difix is enabled
  mask/{frame}_5_wide.png         # alpha > 0.5, used by metrics
"""

import sys
_local = [p for p in sys.path if ".local" in p]
if _local:
    sys.path = [p for p in sys.path if ".local" not in p] + _local

import os
import time
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

# ── Global config ──
DATA_DIR = "data/datasets/WideDrive_processed/WideDriveVal"
SCENE_LIST = "data/datasets/WideDrive_processed/WideDriveVal/val.txt"
CKPT_PATH = "pretrained/model_latest_nuscenes.pt"
OUTPUT_PATH = "outputs_widedrive/widedrive_multiframes_inference"
TARGET_W = 518
TARGET_H = 294
CAM_IDS = [5, 4, 3]            # Input cameras per frame (3 views)
RENDER_CAM_LIST = [5]          # Only save these cameras
WIDE_FACTOR = 3                # Render width = original_width * WIDE_FACTOR
MAX_FRAMES = -1                # Max frames per scene (-1 = all)
DIFIX_CKPT = "pretrained/model_difix.pkl"  # Set to path to enable Difix3D enhancement
DRY_RUN = False                # True = test data loading only, no model inference
BACKGROUND_COLOR = [1.0, 1.0, 1.0]
SIMPLE_MERGE = True            # True = merge all Gaussians with predicted opacity (no dyn weighting)
USE_EGO_CAR_MASK = False       # WideDrive has no ego-car masks
NUSCENES_EGO_CAR_MASK = ""     # unused while USE_EGO_CAR_MASK is False
EGO_CAR_OPACITY_LOGIT = -10.0  # Logit value for suppressed Gaussians (sigmoid ≈ 0.00005)
MULTIFRAMES = 3                # Number of consecutive frames per inference step


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
    tensor = TV.ToTensor()(img)
    return tensor


def load_sky_mask(path, target_h, target_w):
    """Load sky mask, resize, return bool tensor [H, W]. True = not sky (foreground)."""
    mask = Image.open(path).convert("L")
    mask = mask.resize((target_w, target_h), Image.NEAREST)
    mask_np = np.array(mask)
    bg_mask = torch.from_numpy(mask_np == 0)  # [H, W], bool
    return bg_mask


def load_ego_car_mask(path, target_h, target_w):
    """Load ego car mask, resize, return bool tensor [H, W]. True = keep, False = suppress."""
    mask = Image.open(path).convert("L")
    mask = mask.resize((target_w, target_h), Image.NEAREST)
    mask_np = np.array(mask)
    # White (>=128) = keep, Black (<128) = ego car region to suppress
    keep_mask = torch.from_numpy(mask_np >= 128)  # [H, W], bool
    return keep_mask


def save_render_mask(alpha, path, threshold=0.5):
    """Save alpha > threshold as a uint8 mask. White = valid render."""
    if alpha.dim() == 3:
        alpha = alpha[..., 0]
    mask = (alpha.detach().float().cpu() > threshold).numpy().astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(path)


def main():
    # Read scene list
    with open(SCENE_LIST) as f:
        scene_names = [line.strip() for line in f if line.strip()]
    print(f"Scenes to process: {scene_names}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    cam_ids = CAM_IDS
    S = len(cam_ids)            # cameras per frame
    T = MULTIFRAMES             # frames per inference step
    ST = S * T                  # total views per inference step

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
        mask_dir = os.path.join(OUTPUT_PATH, scene_name, "mask")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        if DIFIX_CKPT:
            os.makedirs(before_rgb_dir, exist_ok=True)

        # We need MULTIFRAMES consecutive frames; iteration starts at index T-1
        # so we can look back T-1 frames. Output frame_id = frame_ids[fi].
        render_frame_ids = frame_ids[T - 1:]  # e.g. for T=3: starts at frame_ids[2]

        frame_bar = tqdm(
            enumerate(render_frame_ids),
            total=len(render_frame_ids),
            desc=f"Scene {scene_name}",
            unit="frame",
            leave=False,
        )
        for ri, frame_id in frame_bar:
            start_time = time.time()
            fi = ri + (T - 1)  # index into frame_ids for the current (latest) frame

            # ── Load images, sky masks, ego car masks for T frames × S cameras ──
            images_list = []       # length T*S
            bg_masks_list = []     # length T*S
            ego_car_masks_list = [] # length T*S
            valid = True
            for t_idx in range(T):
                fid = frame_ids[fi - (T - 1 - t_idx)]  # oldest → newest
                for cam_idx, cam_id in enumerate(cam_ids):
                    img_path = os.path.join(images_dir, f"{fid}_{cam_id}.jpg")
                    mask_path = os.path.join(masks_dir, f"{fid}_{cam_id}.png")
                    if not os.path.exists(img_path):
                        print(f"  [SKIP] Missing image: {img_path}")
                        valid = False
                        break
                    images_list.append(load_image(img_path, TARGET_H, TARGET_W))
                    if os.path.exists(mask_path):
                        bg_masks_list.append(load_sky_mask(mask_path, TARGET_H, TARGET_W))
                    else:
                        bg_masks_list.append(torch.ones(TARGET_H, TARGET_W, dtype=torch.bool))

                    # Ego car mask: cam5 (nuScenes CAM_BACK) is the only view that
                    # captures the ego car. Load the shared CAM_BACK_mask for cam5
                    # views of older frames; the current frame's cam5 (the rendered
                    # view) and all non-cam5 views stay all-ones (keep all GS).
                    is_current_cam5 = (t_idx == T - 1) and (cam_id == 5)
                    if USE_EGO_CAR_MASK and cam_id == 5 and not is_current_cam5:
                        if os.path.exists(NUSCENES_EGO_CAR_MASK):
                            ego_car_masks_list.append(
                                load_ego_car_mask(NUSCENES_EGO_CAR_MASK, TARGET_H, TARGET_W)
                            )
                        else:
                            print(f"  [WARN] Ego car mask not found: {NUSCENES_EGO_CAR_MASK}, keeping all")
                            ego_car_masks_list.append(
                                torch.ones(TARGET_H, TARGET_W, dtype=torch.bool)
                            )
                    else:
                        # current frame cam5, or non-cam5 view, or feature disabled
                        ego_car_masks_list.append(
                            torch.ones(TARGET_H, TARGET_W, dtype=torch.bool)
                        )
                if not valid:
                    break
            if not valid:
                continue

            # Stack: images [1, T*S, 3, H, W], bg_masks [1, T*S, H, W]
            images = torch.stack(images_list).unsqueeze(0).to(device)
            bg_masks = torch.stack(bg_masks_list).unsqueeze(0).to(device)
            if USE_EGO_CAR_MASK:
                ego_car_masks = torch.stack(ego_car_masks_list).unsqueeze(0).to(device)

            # Timestamps: each frame group has a distinct t, views within same frame share t
            # oldest frame t=0, ... newest frame t=T-1
            # [0,0,0, 1,1,1, 2,2,2] for T=3, S=3
            timestamps = torch.arange(T, device=device).repeat_interleave(S).float()

            if DRY_RUN:
                print(f"  Frame {frame_id}: images={images.shape}, bg_masks={bg_masks.shape}")
                print(f"  Timestamps: {timestamps.tolist()}")
                continue

            # ── Model forward ──
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
                    static_mask = bg_masks  # [1, T*S, H, W]
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
                # nuScenes: only cam5 (CAM_BACK) views of older frames see the ego
                # car. Suppress Gaussians from those views that fall in the mask's
                # ego-car region. The current frame's cam5 (rendered view) keeps
                # all Gaussians. For CAM_IDS=[5,4,3], cam_idx==0 means cam5.
                if USE_EGO_CAR_MASK:
                    nonzero_indices = torch.nonzero(static_mask, as_tuple=False)  # [N, 4]: (batch, view, h, w)
                    view_indices = nonzero_indices[:, 1]  # view index (0..T*S-1) per Gaussian
                    h_indices = nonzero_indices[:, 2]
                    w_indices = nonzero_indices[:, 3]

                    # cam5 views except the current frame's cam5 (which is all-1)
                    is_cam5_view = (view_indices % S) == 0
                    current_cam5_view = T * S - S
                    is_non_current_cam5 = is_cam5_view & (view_indices != current_cam5_view)
                    if is_non_current_cam5.any():
                        # cam_idx within CAM_IDS = view_index % S (== 0 for cam5)
                        cam_indices_in_list = view_indices[is_non_current_cam5] % S
                        ego_keep = ego_car_masks[
                            0,
                            cam_indices_in_list,
                            h_indices[is_non_current_cam5],
                            w_indices[is_non_current_cam5],
                        ]
                        # ego_keep=False means ego car region → suppress
                        suppress = is_non_current_cam5.clone()
                        suppress[is_non_current_cam5] = ~ego_keep
                        static_opacity[suppress] = EGO_CAR_OPACITY_LOGIT

                # Dynamic Gaussians (per view)
                dynamic_points, dynamic_rgbs, dynamic_opacitys = [], [], []
                dynamic_scales, dynamic_rotations = [], []
                if not SIMPLE_MERGE:
                    for i in range(ST):
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
                # Only the current frame's cam5 (last view) is rendered at wide resolution
                wide_W = W * WIDE_FACTOR
                intrinsic_wide = intrinsic.clone()
                # Keep original intrinsics for sky model sampling
                # Only the current frame's cam5 gets wide cx
                render_view_idx = T * S - S  # = (T-1)*S = index of current frame's cam5
                intrinsic_wide[render_view_idx, 0, 2] = wide_W / 2.0  # cx at wide center

                # ── Rendering: only current frame's cam5 ──
                t0 = timestamps[render_view_idx]  # current frame's timestamp
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
                renders = renders[..., :-1]  # drop depth, keep RGB  [1, H, wide_W, 3]

                # Sky background (at wide resolution)
                # Pass all T*S images/views for best sky prediction
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
                rendered_image = renders.permute(0, 3, 1, 2)  # [1, 3, H, wide_W]

            # ── Save results ──
            elapsed = time.time() - start_time
            frame_bar.set_postfix({"time": f"{elapsed:.1f}s"})

            cam_id = 5  # current frame's cam5
            rendered = rendered_image[0].detach().cpu().clamp(0, 1)
            save_render_mask(
                alphas[0], os.path.join(mask_dir, f"{frame_id}_{cam_id}_wide.png")
            )

            if DIFIX_CKPT:
                before_path = os.path.join(
                    before_rgb_dir, f"{frame_id}_{cam_id}_wide.jpg"
                )
                TV.ToPILImage()(rendered).save(before_path, quality=95)
                rendered = process_images_with_difix(
                    rendered, DIFIX_CKPT, model=difix_model
                )

            save_path = os.path.join(rgb_dir, f"{frame_id}_{cam_id}_wide.jpg")
            TV.ToPILImage()(rendered.clamp(0, 1)).save(save_path, quality=95)

    scene_bar.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
