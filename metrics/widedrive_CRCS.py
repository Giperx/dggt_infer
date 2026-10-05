"""CRCS for WideDrive wide renders.

CRCS is the mean absolute horizontal color step, in 0-255 units, inside the
left seam, right seam, and full image. Masked scores use only the render mask.
No GT is required, and only frames that were actually rendered are scored.
"""
import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from tqdm import tqdm

from widedrive_common import (
    choose_image_dir,
    find_render_mask,
    list_render_frames,
    load_render_mask,
    load_scene_ids,
)

SEAM_RATIO = 0.10
SEAM_VERTICAL_RATIO = 0.5
REGIONS = ("L", "R", "Overall")


def parse_args():
    parser = argparse.ArgumentParser(description="WideDrive CRCS")
    parser.add_argument(
        "--render-root",
        default="outputs_widedrive/widedrive_multiframes_inference",
    )
    parser.add_argument(
        "--val-list",
        default="data/datasets/WideDrive_processed/WideDriveVal/val.txt",
    )
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--seam-ratio", type=float, default=SEAM_RATIO)
    parser.add_argument("--seam-vertical-ratio", type=float, default=SEAM_VERTICAL_RATIO)
    return parser.parse_args()


def compute_crcs(image_uint8, mask, seam_ratio, vertical_ratio):
    height, width = image_uint8.shape[:2]
    diff = np.abs(image_uint8[:, 1:].astype(np.int16) - image_uint8[:, :-1].astype(np.int16)).mean(axis=2)
    v_half = int(height * vertical_ratio / 2)
    v_center = height // 2
    v0 = max(0, v_center - v_half)
    v1 = min(height, v_center + v_half)
    seam_half = int(width * seam_ratio)
    left_center = width // 3
    right_center = 2 * width // 3
    spans = {
        "L": (max(0, left_center - seam_half), min(width - 1, left_center + seam_half)),
        "R": (max(0, right_center - seam_half), min(width - 1, right_center + seam_half)),
        "Overall": (0, width - 1),
    }
    results = {}
    for name, (x0, x1) in spans.items():
        if name == "Overall":
            values = diff[:, x0:x1]
            pair = None if mask is None else (mask[:, x0:x1] & mask[:, x0 + 1:x1 + 1])
        else:
            values = diff[v0:v1, x0:x1]
            pair = None if mask is None else (mask[v0:v1, x0:x1] & mask[v0:v1, x0 + 1:x1 + 1])
        unmasked = float(values.mean()) if values.size else 0.0
        masked = float(values[pair].mean()) if pair is not None and pair.any() else None
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
    for scene, frame_id, suffix, path, scene_dir in tqdm(jobs, desc="WideDrive CRCS"):
        image = np.asarray(Image.open(path).convert("RGB"))
        mask_path = find_render_mask(scene_dir, frame_id, suffix)
        mask, _reason = load_render_mask(mask_path, image.shape[:2])
        if mask is None:
            bad_mask += 1
        scores = compute_crcs(image, mask, args.seam_ratio, args.seam_vertical_ratio)
        scene_scores = per_scene.setdefault(scene, {name: {"unmasked": [], "masked": []} for name in REGIONS})
        for name in REGIONS:
            scene_scores[name]["unmasked"].append(scores[name]["unmasked"])
            totals[name]["unmasked"].append(scores[name]["unmasked"])
            if scores[name]["masked"] is not None:
                scene_scores[name]["masked"].append(scores[name]["masked"])
                totals[name]["masked"].append(scores[name]["masked"])
        suffixes.add(suffix or "(none)")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.render_root, f"widedrive_CRCS_{timestamp}.txt")
    with open(out_path, "w") as handle:
        handle.write("=== WideDrive CRCS ===\n")
        handle.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"Render root: {args.render_root}\n")
        handle.write(f"Seam ratio: {args.seam_ratio}, vertical ratio: {args.seam_vertical_ratio}\n")
        handle.write(f"Suffixes: {', '.join(sorted(suffixes)) or 'none'}\n")
        handle.write(f"Scenes: {len(per_scene)} / {len(scenes)}; missing render dirs: {len(missing)}\n")
        handle.write(f"Frames with no usable render mask: {bad_mask}\n")
        handle.write(f"Elapsed: {time.time() - started:.1f}s\n\n")
        handle.write("Summary\n")
        _write_crcs(handle, totals)
        handle.write("\nPer-scene\n")
        for scene in sorted(per_scene):
            handle.write(f"\nScene {scene}:\n")
            _write_crcs(handle, per_scene[scene], indent="  ")
    print(f"Results saved to {out_path}")


def _write_crcs(handle, bucket, indent=""):
    for name in REGIONS:
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
