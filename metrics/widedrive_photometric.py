"""Dense PSNR, MAE, RMSE, SSIM, and LPIPS on WideDrive wide images."""
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from skimage.exposure import match_histograms

from widedrive_common import region_bounds

REGION_NAMES = ("Full", "Left", "Center", "Right")
VARIANTS = ("unmasked", "masked")
METRIC_NAMES = ("psnr", "mae", "rmse", "ssim", "lpips")


def _gaussian_kernel(win_size, sigma, channels, device):
    coords = torch.arange(win_size, dtype=torch.float32, device=device) - win_size // 2
    weights = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    weights = weights / weights.sum()
    kernel = (weights[:, None] * weights[None, :]).expand(channels, 1, win_size, win_size).contiguous()
    return kernel


def dense_ssim_map(rendered, gt, device, win_size=11, data_range=1.0):
    """Window SSIM map, [H, W], averaged over RGB."""
    left = torch.from_numpy(np.ascontiguousarray(rendered)).float().permute(2, 0, 1).unsqueeze(0).to(device)
    right = torch.from_numpy(np.ascontiguousarray(gt)).float().permute(2, 0, 1).unsqueeze(0).to(device)
    channels = left.shape[1]
    kernel = _gaussian_kernel(win_size, 1.5, channels, device)
    pad = win_size // 2
    mu_x = F.conv2d(left, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(right, kernel, padding=pad, groups=channels)
    sigma_x = F.conv2d(left ** 2, kernel, padding=pad, groups=channels) - mu_x ** 2
    sigma_y = F.conv2d(right ** 2, kernel, padding=pad, groups=channels) - mu_y ** 2
    sigma_xy = F.conv2d(left * right, kernel, padding=pad, groups=channels) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean(dim=1).squeeze(0).detach().cpu().numpy()


def scalar_errors(rendered, gt, mask):
    if mask.sum() == 0:
        return None
    diff = rendered - gt
    mae = np.abs(diff)[mask].mean()
    mse = np.square(diff)[mask].mean()
    rmse = float(np.sqrt(mse))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
    return {"mae": float(mae), "rmse": rmse, "psnr": psnr, "n_pixels": int(mask.sum())}


def match_region(rendered, gt, mask):
    """Histogram-match rendered pixels selected by mask. Other pixels stay unchanged."""
    if mask.sum() == 0:
        return None
    matched = rendered.copy()
    pred = rendered[mask]
    ref = gt[mask]
    aligned = match_histograms(pred[None, ...], ref[None, ...], channel_axis=-1)[0]
    matched[mask] = np.clip(aligned, 0.0, 1.0)
    return matched


def load_lpips(device):
    import lpips

    model = lpips.LPIPS(net="alex", spatial=True).to(device)
    model.eval()
    return model


def spatial_lpips(lpips_fn, rendered, gt, device):
    pred = torch.from_numpy(np.ascontiguousarray(rendered)).float().permute(2, 0, 1).unsqueeze(0)
    ref = torch.from_numpy(np.ascontiguousarray(gt)).float().permute(2, 0, 1).unsqueeze(0)
    pred = pred.to(device) * 2.0 - 1.0
    ref = ref.to(device) * 2.0 - 1.0
    with torch.no_grad():
        lpips_map = lpips_fn(pred, ref)
    lpips_map = lpips_map.detach().float().cpu().numpy()
    return np.squeeze(lpips_map)


def _resize_map(lpips_map, shape_hw):
    if lpips_map.shape == tuple(shape_hw):
        return lpips_map
    import torch.nn.functional as torch_F

    tensor = torch.from_numpy(lpips_map).float()[None, None]
    resized = torch_F.interpolate(tensor, size=shape_hw, mode="bilinear", align_corners=False)
    return resized[0, 0].numpy()


def evaluate_regions(rendered, gt, render_mask, device, lpips_fn, histogram_match=False):
    """Score full image and L/Center/R, each with and without the render mask."""
    height, width = rendered.shape[:2]
    bounds = {"Full": (0, width), **region_bounds(width)}
    results = {}
    for name, (x0, x1) in bounds.items():
        crop_render = rendered[:, x0:x1]
        crop_gt = gt[:, x0:x1]
        crop_mask = None if render_mask is None else render_mask[:, x0:x1]
        for variant in VARIANTS:
            if variant == "masked":
                if crop_mask is None:
                    continue
                valid = crop_mask
            else:
                valid = np.ones(crop_render.shape[:2], dtype=bool)
            pred = crop_render
            if histogram_match:
                pred = match_region(crop_render, crop_gt, valid)
                if pred is None:
                    continue
            errors = scalar_errors(pred, crop_gt, valid)
            if errors is None:
                continue
            ssim_map = dense_ssim_map(pred, crop_gt, device)
            errors["ssim"] = float(ssim_map[valid].mean()) if valid.any() else 0.0
            lpips_map = spatial_lpips(lpips_fn, pred, crop_gt, device)
            lpips_map = _resize_map(lpips_map, valid.shape)
            errors["lpips"] = float(lpips_map[valid].mean()) if valid.any() else 0.0
            results[f"{name}_{variant}"] = errors
    return results


def empty_buckets():
    return {f"{region}_{variant}": [] for region in REGION_NAMES for variant in VARIANTS}


def add_frame(buckets, frame_metrics):
    for key, value in frame_metrics.items():
        buckets[key].append(value)


def _fmt_metric(name, value):
    if name in ("mae", "rmse"):
        return f"{value * 255.0:.2f}"
    if name == "psnr":
        return f"{value:.2f}"
    return f"{value:.4f}"


def write_report(path, title, meta_lines, scene_buckets, global_buckets, skipped):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write(f"=== {title} ===\n")
        handle.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for line in meta_lines:
            handle.write(line.rstrip() + "\n")
        handle.write("MAE and RMSE are reported on the 0-255 scale. PSNR uses [0, 1] images.\n")
        handle.write("Masked metrics use only the rendered alpha mask. GT is dense and has no mask.\n")
        if skipped:
            handle.write(f"Skipped masked frames: {skipped['bad_mask']}. Missing GT frames: {skipped['missing_gt']}.\n")
        handle.write("\n" + "=" * 80 + "\nSummary\n" + "=" * 80 + "\n")
        _write_bucket(handle, global_buckets)
        handle.write("\n" + "=" * 80 + "\nPer-scene\n" + "=" * 80 + "\n")
        for scene in sorted(scene_buckets):
            handle.write(f"\nScene {scene}:\n")
            _write_bucket(handle, scene_buckets[scene], indent="  ")
    return path


def _write_bucket(handle, buckets, indent=""):
    for region in REGION_NAMES:
        for variant in VARIANTS:
            key = f"{region}_{variant}"
            rows = buckets.get(key, [])
            if not rows:
                handle.write(f"{indent}{key:20s}: no frames\n")
                continue
            parts = [f"n={len(rows)}"]
            for name in METRIC_NAMES:
                values = [row[name] for row in rows if np.isfinite(row[name])]
                if values:
                    parts.append(f"{name.upper()}={_fmt_metric(name, float(np.mean(values)))}")
            handle.write(f"{indent}{key:20s}: " + ", ".join(parts) + "\n")
