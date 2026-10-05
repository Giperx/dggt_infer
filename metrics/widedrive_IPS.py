"""IPS for WideDrive wide renders.

IPS is the mean horizontal gradient of a masked Gaussian low-pass field,
reported in 0-255 units on the left and right seam bands. Masked scores use
only the render mask. No GT is required.
"""
import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from widedrive_common import (
    choose_image_dir,
    find_render_mask,
    list_render_frames,
    load_render_mask,
    load_scene_ids,
)

SEAM_VERTICAL_RATIO = 0.5
BAND_RATIO = 0.05
SIGMA = 5.0
REGIONS = ("L", "R")


def parse_args():
    parser = argparse.ArgumentParser(description="WideDrive IPS")
    parser.add_argument(
        "--render-root",
        default="outputs_widedrive/widedrive_multiframes_inference",
    )
    parser.add_argument(
        "--val-list",
        default="data/datasets/WideDrive_processed/WideDriveVal/val.txt",
    )
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--seam-vertical-ratio", type=float, default=SEAM_VERTICAL_RATIO)
    parser.add_argument("--band-ratio", type=float, default=BAND_RATIO)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    return parser.parse_args()


def low_frequency(image_uint8, mask, sigma):
    image = image_uint8.astype(np.float64) / 255.0
    if mask is None:
        weight = np.ones(image.shape[:2] + (1,), dtype=np.float64)
    else:
        weight = mask.astype(np.float64)[..., None]
    blurred = gaussian_filter(image * weight, sigma=[sigma, sigma, 0], mode="nearest")
    blurred_weight = gaussian_filter(weight, sigma=[sigma, sigma, 0], mode="nearest")
    low = np.zeros_like(image)
    safe = blurred_weight[:, :, 0] > 1e-5
    for channel in range(3):
        low[:, :, channel][safe] = blurred[:, :, channel][safe] / blurred_weight[:, :, 0][safe]
    return low


def compute_ips(image_uint8, mask, vertical_ratio, band_ratio, sigma):
    height, width = image_uint8.shape[:2]
    low = low_frequency(image_uint8, mask, sigma)
    v_half = int(height * vertical_ratio / 2)
    v_center = height // 2
    v0 = max(0, v_center - v_half)
    v1 = min(height, v_center + v_half)
    band_half = max(1, int(width * band_ratio))
    centers = {"L": width // 3, "R": 2 * width // 3}
    results = {}
    for name, center in centers.items():
        x0 = max(0, center - band_half)
        x1 = min(width, center + band_half)
        band = low[v0:v1, x0:x1]
        grad = np.abs(band[:, 1:] - band[:, :-1]).mean(axis=2)
        unmasked = float(grad.mean() * 255.0) if grad.size else 0.0
        masked = None
        if mask is not None:
            band_mask = mask[v0:v1, x0:x1]
            pair = band_mask[:, 1:] & band_mask[:, :-1]
            if pair.any():
                masked = float(grad[pair].mean() * 255.0)
        results[name] = {"unmasked": unmasked, "masked": masked}
    return results


def main():
    args = parse_args()
    scenes, missing = load_scene_ids(args.render_root, args.val_list)
    if not scenes:
        raise SystemExit(f"no rendered scenes under {args.render_root}")
    totals = {name: {"unmasked": [], "masked": []} for name in REGIONS}
    per_scene = {}
    bad_mask = 0
    suffixes = set()
    started = time.time()
    jobs = []
    for scene in scenes:
        scene_dir = os.path.join(args.render_root, scene)
        image_dir = choose_image_dir(scene_dir, args.image_dir)
        if image_dir is None:
            continue
        for frame_id, suffix, path in list_render_frames(scene_dir, image_dir):
            jobs.append((scene, frame_id, suffix, path, scene_dir))
    for scene, frame_id, suffix, path, scene_dir in tqdm(jobs, desc="WideDrive IPS"):
        image = np.asarray(Image.open(path).convert("RGB"))
        mask_path = find_render_mask(scene_dir, frame_id, suffix)
        mask, _reason = load_render_mask(mask_path, image.shape[:2])
        if mask is None:
            bad_mask += 1
        scores = compute_ips(image, mask, args.seam_vertical_ratio, args.band_ratio, args.sigma)
        scene_scores = per_scene.setdefault(scene, {name: {"unmasked": [], "masked": []} for name in REGIONS})
        for name in REGIONS:
            scene_scores[name]["unmasked"].append(scores[name]["unmasked"])
            totals[name]["unmasked"].append(scores[name]["unmasked"])
            if scores[name]["masked"] is not None:
                scene_scores[name]["masked"].append(scores[name]["masked"])
                totals[name]["masked"].append(scores[name]["masked"])
        suffixes.add(suffix or "(none)")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.render_root, f"widedrive_IPS_{timestamp}.txt")
    with open(out_path, "w") as handle:
        handle.write("=== WideDrive IPS ===\n")
        handle.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"Render root: {args.render_root}\n")
        handle.write(
            f"Vertical ratio: {args.seam_vertical_ratio}, band ratio: {args.band_ratio}, sigma: {args.sigma}\n"
        )
        handle.write(f"Suffixes: {', '.join(sorted(suffixes)) or 'none'}\n")
        handle.write(f"Scenes: {len(per_scene)} / {len(scenes)}; missing render dirs: {len(missing)}\n")
        handle.write(f"Frames with no usable render mask: {bad_mask}\n")
        handle.write(f"Elapsed: {time.time() - started:.1f}s\n\n")
        handle.write("Summary\n")
        _write_ips(handle, totals)
        handle.write("\nPer-scene\n")
        for scene in sorted(per_scene):
            handle.write(f"\nScene {scene}:\n")
            _write_ips(handle, per_scene[scene], indent="  ")
    print(f"Results saved to {out_path}")


def _write_ips(handle, bucket, indent=""):
    for name in list(REGIONS) + ["MeanLR"]:
        if name == "MeanLR":
            unmasked = [
                0.5 * (left + right)
                for left, right in zip(bucket["L"]["unmasked"], bucket["R"]["unmasked"])
            ]
            masked_pairs = list(zip(bucket["L"]["masked"], bucket["R"]["masked"]))
            masked = [0.5 * (left + right) for left, right in masked_pairs]
        else:
            unmasked = bucket[name]["unmasked"]
            masked = bucket[name]["masked"]
        unmasked_text = f"{np.mean(unmasked):.4f}" if unmasked else "n/a"
        masked_text = f"{np.mean(masked):.4f}" if masked else "n/a"
        handle.write(
            f"{indent}{name:8s}: unmasked={unmasked_text} (n={len(unmasked)}), "
            f"masked={masked_text} (n={len(masked)})\n"
        )


if __name__ == "__main__":
    main()
