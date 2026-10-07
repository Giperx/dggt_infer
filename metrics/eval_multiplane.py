"""Score multiplane stitches against sparseMultiplaneImages3 ground truth.

Renders are ``{frame}_5_wide.jpg`` from inference_multiplane.py. Those cameras
are predicted. GT is ``{frame}_5_multiplane_wide.png``, a 1554-wide stitch.
Left, center, and right are equal thirds. CBSR and PD are not computed.

SSIM follows the sparse-dataset rule, not WideDrive. Left and right are sparse
luminance SSIM on the GT mask. Center is an 11x11 window SSIM on that mask.
With histogram matching, the SSIM averaged into mean L+R and mean L+R+Center is
the per-pixel HM-SSIM of every region, including center. The window SSIM stays
a separate mixed line.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from tqdm import tqdm

from widedrive_common import load_scene_ids
from widedrive_photometric import (
    dense_ssim_map,
    load_lpips,
    match_region,
    scalar_errors,
    spatial_lpips,
    _resize_map,
)

PRESETS = {
    "nuscenes": {
        "gt_root": "data/datasets/nuscenes/sparseMultiplaneImages3_1554x294",
        "val_list": "data/datasets/nuscenes/processed_10Hz_v2/nuScenes_Val.txt",
        "single": "outputs_nuscenes_pt/nuscenes_multiplane_inference",
        "multiframes": "outputs_nuscenes_pt/nuscenes_multiplane_multiframes",
    },
    "ddad": {
        "gt_root": "data/datasets/ddad_process/sparseMultiplaneImages3_1554x322",
        "val_list": "data/datasets/ddad_process/valid/valid.txt",
        "single": "outputs_nuscenes_pt/ddad_multiplane_inference",
        "multiframes": "outputs_nuscenes_pt/ddad_multiplane_multiframes",
    },
    "lyft1920": {
        "gt_root": "data/datasets/lyft/1920_sparseMultiplaneWideFOVImages3",
        "val_list": "data/datasets/lyft/lyft_val1920_3cams/lyft_val1920.txt",
        "single": "outputs_nuscenes_pt/lyft_multiplane/lyft1920_inference",
        "multiframes": "outputs_nuscenes_pt/lyft_multiplane_multiframes/lyft1920_multiframes_inference",
    },
    "lyft1224": {
        "gt_root": "data/datasets/lyft/1224_sparseMultiplaneWideFOVImages3",
        "val_list": "data/datasets/lyft/lyft_val1224_3cams/lyft_val1224.txt",
        "single": "outputs_nuscenes_pt/lyft_multiplane/lyft1224_inference",
        "multiframes": "outputs_nuscenes_pt/lyft_multiplane_multiframes/lyft1224_multiframes_inference",
    },
}
REGIONS = ("Left", "Center", "Right")
NAMES = ("psnr", "mae", "rmse", "ssim", "lpips")


def parse_args():
    parser = argparse.ArgumentParser(description="Multiplane sparse GT metrics")
    parser.add_argument("--dataset", required=True, choices=sorted(PRESETS) + ["lyft"])
    parser.add_argument("--mode", default="multiframes", choices=("single", "multiframes"))
    parser.add_argument("--render-root", default=None)
    parser.add_argument("--gt-root", default=None)
    parser.add_argument("--val-list", default=None)
    parser.add_argument("--image-dir", default="rgb")
    parser.add_argument("--histogram-match", action="store_true")
    return parser.parse_args()


def _find_frame(folder, frame_id, suffix):
    if not os.path.isdir(folder):
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(folder, f"{frame_id}{suffix}{ext}")
        if os.path.exists(path):
            return path
    return None


def _load_rgb(path, size_wh=None):
    image = Image.open(path).convert("RGB")
    if size_wh is not None and image.size != size_wh:
        image = image.resize(size_wh, Image.BICUBIC)
    return np.asarray(image).astype(np.float64) / 255.0


def _load_mask(path, shape_hw):
    if path is None or not os.path.exists(path):
        return None
    mask = np.asarray(Image.open(path).convert("L"))
    if mask.shape != tuple(shape_hw):
        mask = np.asarray(Image.open(path).convert("L").resize((shape_hw[1], shape_hw[0]), Image.NEAREST))
    return mask > 127


def sparse_ssim(rendered, gt, mask, k1=0.01):
    """Per-pixel luminance SSIM. No sliding window. Equivalent in 0-1 or 0-255."""
    if int(mask.sum()) == 0:
        return 0.0
    pred = rendered[mask] * 255.0
    ref = gt[mask] * 255.0
    c1 = (k1 * 255.0) ** 2
    term = (2.0 * pred * ref + c1) / (pred ** 2 + ref ** 2 + c1)
    return float(term.mean())


def _fmt(name, value):
    if name in ("mae", "rmse"):
        return f"{value * 255.0:.2f}"
    if name == "psnr":
        return f"{value:.2f}"
    return f"{value:.4f}"


def _mean_line(rows, keys):
    parts = []
    for name in NAMES:
        values = []
        for key in keys:
            values.extend(row[key][name] for row in rows if key in row and np.isfinite(row[key][name]))
        if values:
            parts.append(f"{name.upper()}={_fmt(name, float(np.mean(values)))}")
    return ", ".join(parts) if parts else "no frames"


def score_tree(render_root, gt_root, val_list, image_dir, device, lpips_fn, histogram_match):
    scenes, missing = load_scene_ids(render_root, val_list)
    rows = []
    skipped_gt = 0
    for scene in scenes:
        rgb_dir = os.path.join(render_root, scene, image_dir)
        if not os.path.isdir(rgb_dir):
            continue
        frames = sorted(
            name.split("_")[0]
            for name in os.listdir(rgb_dir)
            if name.endswith("_5_wide.jpg") or name.endswith("_5_wide.png")
        )
        for frame in frames:
            render_path = _find_frame(rgb_dir, frame, "_5_wide")
            gt_path = _find_frame(os.path.join(gt_root, scene, "rgb"), frame, "_5_multiplane_wide")
            if render_path is None or gt_path is None:
                skipped_gt += 1
                continue
            rendered = _load_rgb(render_path)
            gt = _load_rgb(gt_path, (rendered.shape[1], rendered.shape[0]))
            gt_mask = _load_mask(
                _find_frame(os.path.join(gt_root, scene, "mask"), frame, "_5_multiplane_wide"),
                rendered.shape[:2],
            )
            if gt_mask is None:
                gt_mask = np.ones(rendered.shape[:2], dtype=bool)
            third = rendered.shape[1] // 3
            bounds = {"Left": (0, third), "Center": (third, third * 2), "Right": (third * 2, rendered.shape[1])}
            scored = {"scene": scene, "frame": frame}
            for name, (x0, x1) in bounds.items():
                valid = gt_mask[:, x0:x1]
                pred = rendered[:, x0:x1]
                ref = gt[:, x0:x1]
                if histogram_match:
                    matched = match_region(pred, ref, valid)
                    if matched is None:
                        continue
                    pred = matched
                errors = scalar_errors(pred, ref, valid)
                if errors is None:
                    continue
                if histogram_match or name != "Center":
                    errors["ssim"] = sparse_ssim(pred, ref, valid)
                else:
                    ssim_map = dense_ssim_map(pred, ref, device)
                    errors["ssim"] = float(ssim_map[valid].mean()) if valid.any() else 0.0
                if histogram_match and name == "Center":
                    ssim_map = dense_ssim_map(pred, ref, device)
                    errors["dense_ssim"] = float(ssim_map[valid].mean()) if valid.any() else 0.0
                if name == "Center":
                    lpips_map = _resize_map(spatial_lpips(lpips_fn, pred, ref, device), valid.shape)
                    errors["lpips"] = float(lpips_map[valid].mean()) if valid.any() else 0.0
                else:
                    errors["lpips"] = float("nan")
                scored[name] = errors
            if any(name in scored for name in REGIONS):
                rows.append(scored)
    return scenes, missing, rows, skipped_gt


def _dense_center_mean(rows):
    values = [
        row["Center"]["dense_ssim"]
        for row in rows
        if "Center" in row and "dense_ssim" in row["Center"] and np.isfinite(row["Center"]["dense_ssim"])
    ]
    if not values:
        return None
    return float(np.mean(values))


def _mixed_ssim(rows):
    """(Left HM-SSIM + Center window SSIM + Right HM-SSIM) / 3."""
    center = _dense_center_mean(rows)
    if center is None:
        return None
    parts = [center]
    for name in ("Left", "Right"):
        values = [
            row[name]["ssim"]
            for row in rows
            if name in row and np.isfinite(row[name]["ssim"])
        ]
        if not values:
            return None
        parts.append(float(np.mean(values)))
    return float(np.mean(parts))


def _write_summary(handle, rows, indent=""):
    dense_center = _dense_center_mean(rows)
    for name in REGIONS:
        subset = [row for row in rows if name in row]
        handle.write(f"{indent}{name:8s}: n={len(subset)}, {_mean_line(subset, (name,))}\n")
        if name == "Center" and dense_center is not None:
            handle.write(f"{indent}Center window SSIM: {dense_center:.4f}\n")
    handle.write(f"{indent}mean L+R: n={len(rows)}, {_mean_line(rows, ('Left', 'Right'))}\n")
    handle.write(f"{indent}mean L+R+Center: n={len(rows)}, {_mean_line(rows, REGIONS)}\n")
    mixed = _mixed_ssim(rows)
    if mixed is not None:
        handle.write(
            f"{indent}mean L+R+Center mixed SSIM (L/R sparse, Center window): {mixed:.4f}\n"
        )


def _write_scenes(handle, rows):
    for scene in sorted({row["scene"] for row in rows}):
        scene_rows = [row for row in rows if row["scene"] == scene]
        handle.write(f"\nScene {scene}:\n")
        _write_summary(handle, scene_rows, indent="  ")


def write_report(path, title, meta, rows, missing, skipped_gt, sections=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write(f"=== {title} ===\n")
        handle.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for line in meta:
            handle.write(line.rstrip() + "\n")
        handle.write("Cameras used at render time are predicted, not dataset calibration.\n")
        handle.write("GT is sparseMultiplaneImages3, file {frame}_5_multiplane_wide.\n")
        handle.write("MAE and RMSE are on the 0-255 scale. PSNR uses [0, 1] images.\n")
        handle.write("Left and right use the GT mask. Center LPIPS uses the same mask.\n")
        handle.write("Left and right SSIM is sparse luminance SSIM. Center SSIM is an 11x11 window.\n")
        handle.write("Histogram matching replaces every region's SSIM with per-pixel HM-SSIM, including center.\n")
        handle.write("The mixed SSIM line keeps the center window SSIM and is not the HM mean.\n")
        handle.write("Lyft combined numbers pool every frame. They are not the mean of the two subset means.\n")
        handle.write(f"Missing GT frames: {skipped_gt}. Val scenes without a render dir: {len(missing)}.\n")
        handle.write("\nSummary\n")
        _write_summary(handle, rows, indent="  ")
        if sections:
            for label, section_rows in sections:
                handle.write(f"\n--- {label} ---\n")
                _write_summary(handle, section_rows, indent="  ")
        handle.write("\nPer-scene\n")
        if sections:
            for label, section_rows in sections:
                handle.write(f"\n[{label}]\n")
                _write_scenes(handle, section_rows)
        else:
            _write_scenes(handle, rows)
    return path


def _score_one(name, mode, image_dir, device, lpips_fn, histogram_match, render_root=None, gt_root=None, val_list=None):
    preset = PRESETS[name]
    root = render_root or preset[mode]
    gt = gt_root or preset["gt_root"]
    listed = val_list or preset["val_list"]
    if not os.path.isdir(root):
        raise SystemExit(f"render root not found: {root}")
    scenes, missing, rows, skipped = score_tree(root, gt, listed, image_dir, device, lpips_fn, histogram_match)
    for row in rows:
        row["scene"] = f"{name}/{row['scene']}"
    return {
        "name": name,
        "root": root,
        "gt": gt,
        "listed": listed,
        "scenes": scenes,
        "missing": missing,
        "rows": rows,
        "skipped": skipped,
    }


def main():
    args = parse_args()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = load_lpips(device)
    started = time.time()
    names = ("lyft1920", "lyft1224") if args.dataset == "lyft" else (args.dataset,)
    scored = [
        _score_one(
            name, args.mode, args.image_dir, device, lpips_fn, args.histogram_match,
            args.render_root if len(names) == 1 else None,
            args.gt_root if len(names) == 1 else None,
            args.val_list if len(names) == 1 else None,
        )
        for name in names
    ]
    rows = [row for item in scored for row in item["rows"]]
    if not rows:
        raise SystemExit("no paired frames")
    tag = "HM_" if args.histogram_match else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dataset == "lyft":
        out_dir = PRESETS["lyft1920"][args.mode]
        out_name = f"multiplane_{tag}metrics_lyft_combined_{timestamp}.txt"
        title = "DGGT Lyft combined multiplane metrics"
        sections = [(item["name"], item["rows"]) for item in scored]
    else:
        out_dir = scored[0]["root"]
        out_name = f"multiplane_{tag}metrics_{timestamp}.txt"
        title = f"DGGT {args.dataset} multiplane metrics"
        sections = None
    out_path = os.path.join(out_dir, out_name)
    meta = [f"Image dir: {args.image_dir}", f"Elapsed: {time.time() - started:.1f}s"]
    for item in scored:
        meta.append(
            f"{item['name']}: {item['root']}  gt {item['gt']}  "
            f"frames {len(item['rows'])} / listed scenes {len(item['scenes'])}"
        )
    write_report(
        out_path,
        title,
        meta,
        rows,
        [scene for item in scored for scene in item["missing"]],
        sum(item["skipped"] for item in scored),
        sections,
    )
    print(f"Wrote {out_path}")
    print(f"  mean L+R+Center: {_mean_line(rows, REGIONS)}")


if __name__ == "__main__":
    main()
