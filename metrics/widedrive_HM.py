"""Histogram-matched dense metrics for WideDrive wide renders.

Same regions and mask/unmask split as widedrive_metrics.py. Each scored
region is histogram-matched to GT before PSNR, MAE, RMSE, SSIM, and LPIPS.
"""
import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from tqdm import tqdm

from widedrive_common import iter_eval_frames, load_rgb, load_render_mask, load_scene_ids
from widedrive_photometric import (
    add_frame,
    empty_buckets,
    evaluate_regions,
    load_lpips,
    write_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="WideDrive histogram-matched wide metrics")
    parser.add_argument(
        "--render-root",
        default="outputs_widedrive/widedrive_multiframes_inference",
    )
    parser.add_argument("--gt-root", default="data/datasets/WideDrive_processed/sparseWideFOVImages3_1554x294")
    parser.add_argument(
        "--val-list",
        default="data/datasets/WideDrive_processed/WideDriveVal/val.txt",
    )
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--gt-cam", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    scenes, missing = load_scene_ids(args.render_root, args.val_list)
    if not scenes:
        raise SystemExit(f"no rendered scenes under {args.render_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = load_lpips(device)
    global_buckets = empty_buckets()
    scene_buckets = {}
    skipped = {"bad_mask": 0, "missing_gt": 0}
    suffixes = set()
    image_dirs = set()
    frames = [
        item for item in iter_eval_frames(
            args.render_root, args.gt_root, scenes, args.image_dir, args.gt_cam
        )
        if item["status"] != "no_render_dir"
    ]
    started = time.time()
    for item in tqdm(frames, desc="WideDrive HM"):
        if item["status"] == "missing_gt":
            skipped["missing_gt"] += 1
            continue
        if item["status"] != "ok":
            continue
        rendered = load_rgb(item["render_path"])
        gt = load_rgb(item["gt_path"], (rendered.shape[1], rendered.shape[0]))
        mask, _reason = load_render_mask(item["mask_path"], rendered.shape[:2])
        if mask is None:
            skipped["bad_mask"] += 1
        frame_metrics = evaluate_regions(rendered, gt, mask, device, lpips_fn, histogram_match=True)
        scene_buckets.setdefault(item["scene"], empty_buckets())
        add_frame(scene_buckets[item["scene"]], frame_metrics)
        add_frame(global_buckets, frame_metrics)
        suffixes.add(item["suffix"] or "(none)")
        image_dirs.add(item["image_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.render_root, f"widedrive_HM_{timestamp}.txt")
    write_report(
        out_path,
        "WideDrive histogram-matched wide metrics",
        [
            f"Render root: {args.render_root}",
            f"GT root: {args.gt_root}",
            f"Val list: {args.val_list}",
            f"GT camera: {args.gt_cam}",
            f"Image dirs: {', '.join(sorted(image_dirs)) or 'none'}",
            f"Filename suffixes: {', '.join(sorted(suffixes)) or 'none'}",
            f"Scenes scored: {len(scene_buckets)} / listed {len(scenes)}; missing render dirs: {len(missing)}",
            "Histogram matching is fit independently on each region and mask variant.",
            f"Elapsed: {time.time() - started:.1f}s",
        ],
        scene_buckets,
        global_buckets,
        skipped,
    )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
