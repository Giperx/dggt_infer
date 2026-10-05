"""Score wide renders with CBSR and PD. No ground truth is required.

CBSR is the cross-band seam ratio (lower is better). PD is panel detail
(higher means more local contrast). The ranking number is the mean over
frames. Median and P90 are written too, but they are not the ranking numbers.

``metrics/widedrive_CRCS.py`` and ``metrics/widedrive_IPS.py`` are kept for
comparison and are not called by ``metrics/run_widedrive.sh``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

from consistency import score_image
from widedrive_common import choose_image_dir, list_render_frames, load_scene_ids

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


NAMES = ("cbsr", "pd")


def parse_args():
    parser = argparse.ArgumentParser(description="Wide-view CBSR and PD. No ground truth.")
    parser.add_argument(
        "--render-root",
        default="outputs_widedrive/widedrive_multiframes_inference",
    )
    parser.add_argument(
        "--val-list",
        default="data/datasets/WideDrive_processed/WideDriveVal/val.txt",
    )
    parser.add_argument(
        "--image-dir",
        action="append",
        default=None,
        help="rgb, before_rgb, or before_affine_rgb. Repeat to score each separately. "
        "Default picks the first directory that exists, per scene.",
    )
    return parser.parse_args()


def _stats(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def _bundle(rows):
    if not rows:
        return None
    cbsr_stats = _stats([row["cbsr"] for row in rows])
    pd_stats = _stats([row["pd"] for row in rows])
    if cbsr_stats is None or pd_stats is None:
        return None
    return {"cbsr": cbsr_stats, "pd": pd_stats}


def _write_stats(handle, stats, indent=""):
    if stats is None:
        handle.write(f"{indent}no frames\n")
        return
    handle.write(
        f"{indent}n={stats['cbsr']['n']}  "
        f"CBSR={stats['cbsr']['mean']:.4f} (median {stats['cbsr']['median']:.4f}, "
        f"p90 {stats['cbsr']['p90']:.4f})  "
        f"PD={stats['pd']['mean']:.4f} (median {stats['pd']['median']:.4f}, "
        f"p90 {stats['pd']['p90']:.4f})\n"
    )


def _score_path(item):
    scene, frame_id, path = item
    rgb = np.asarray(Image.open(path).convert("RGB"))
    row = score_image(rgb)
    row["scene"] = scene
    row["frame"] = frame_id
    return row


def evaluate_image_dir(render_root, val_list, image_dir=None, progress=None, workers=1):
    """Score one render tree.

    ``image_dir`` None selects the first existing render folder per scene.
    An explicit directory is not replaced when that scene lacks it.
    """
    if not os.path.isdir(render_root):
        raise FileNotFoundError(f"render root not found: {render_root}")
    listed = val_list if val_list else None
    scenes, missing = load_scene_ids(render_root, listed)
    jobs = []
    used_dirs = set()
    for scene in scenes:
        scene_dir = os.path.join(render_root, scene)
        if image_dir is None:
            selected = choose_image_dir(scene_dir, None)
        elif os.path.isdir(os.path.join(scene_dir, image_dir)):
            selected = image_dir
        else:
            selected = None
        if selected is None:
            continue
        frames = list_render_frames(scene_dir, selected)
        if not frames:
            continue
        used_dirs.add(selected)
        for frame_id, _suffix, path in frames:
            jobs.append((scene, frame_id, path))
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            iterator = pool.map(_score_path, jobs, chunksize=16)
            if progress:
                iterator = tqdm(iterator, total=len(jobs), desc=progress, unit="frame")
            rows = list(iterator)
    else:
        rows = []
        iterator = jobs
        if progress:
            iterator = tqdm(jobs, desc=progress, unit="frame")
        for scene, frame_id, path in iterator:
            rows.append(_score_path((scene, frame_id, path)))
    by_scene = {}
    for row in rows:
        by_scene.setdefault(row["scene"], []).append(row)
    label = image_dir if image_dir else (",".join(sorted(used_dirs)) or "none")
    return {
        "scenes": scenes,
        "missing": missing,
        "rows": rows,
        "by_scene": by_scene,
        "used_dirs": sorted(used_dirs),
        "image_dir_label": label,
    }


def _write_section(handle, result):
    _write_stats(handle, _bundle(result["rows"]))
    handle.write("Rank by the means. CBSR is lower-better. PD is higher-more-detail.\n")
    handle.write("A lower CBSR with a much lower PD is contrast collapse, not a better seam.\n")
    if not result["by_scene"]:
        return
    handle.write("\nPer-scene\n")
    for scene in sorted(result["by_scene"]):
        handle.write(f"\nScene {scene}:\n")
        _write_stats(handle, _bundle(result["by_scene"][scene]), indent="  ")


def write_report(path, title, meta_lines, sections):
    """``sections`` is a list of ``(heading, result)`` from ``evaluate_image_dir``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write(f"=== {title} ===\n")
        handle.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for line in meta_lines:
            handle.write(line.rstrip() + "\n")
        handle.write("No ground truth is used.\n")
        handle.write("CBSR = mean seam column score / (median ordinary-column score + 0.001).\n")
        handle.write(
            "A column score is the absolute cross-band median of a 40px log-luminance step, "
            "times sign agreement.\n"
        )
        handle.write("Seams are width/3 and 2*width/3. Ordinary columns are every 12px, excluding 80px around each seam.\n")
        handle.write("PD is the median interior horizontal gradient after a sigma-1 luminance blur.\n")
        handle.write("CRCS and IPS are not part of this report.\n")
        for heading, result in sections:
            handle.write("\n" + "=" * 80 + f"\n{heading}\n" + "=" * 80 + "\n")
            dirs = ", ".join(result["used_dirs"]) or "none"
            handle.write(f"Image dirs read: {dirs}\n")
            handle.write(
                f"Scenes scored: {len(result['by_scene'])} / listed {len(result['scenes'])}; "
                f"missing render dirs: {len(result['missing'])}\n"
            )
            _write_section(handle, result)
            if result["missing"]:
                handle.write("\nScenes in the val list with no render directory:\n")
                for scene in result["missing"]:
                    handle.write(f"  {scene}\n")
    return path


def run_measurement(groups, image_dirs, out_path, title, meta_lines, workers=1):
    """Score one or more render trees and write one report.

    ``groups`` is a list of ``(label, render_root, val_list)``.
    ``image_dirs`` is a list of directory names, or ``[None]`` to auto-select.
    A missing ``before_rgb`` does not fail the run when another directory has frames.
    """
    started = time.time()
    sections = []
    combined = {image_dir: [] for image_dir in image_dirs}
    found_root = False
    labels_with_rows = set()
    for label, render_root, val_list in groups:
        if not os.path.isdir(render_root):
            print(f"render root not found, skip: {render_root}", flush=True)
            continue
        found_root = True
        for image_dir in image_dirs:
            progress = f"{label} {image_dir or 'auto'}"
            result = evaluate_image_dir(
                render_root, val_list, image_dir, progress=progress, workers=workers
            )
            sections.append((f"{label} / {result['image_dir_label']}", result))
            if result["rows"]:
                labels_with_rows.add(label)
            for row in result["rows"]:
                tagged = dict(row)
                tagged["scene"] = f"{label}/{row['scene']}"
                combined[image_dir].append(tagged)
    if not found_root:
        raise SystemExit("no render root found")
    if len(labels_with_rows) > 1:
        for image_dir, rows in combined.items():
            by_scene = {}
            for row in rows:
                by_scene.setdefault(row["scene"], []).append(row)
            label = image_dir or "auto"
            sections.append((
                f"combined / {label}",
                {
                    "scenes": sorted({row["scene"] for row in rows}),
                    "missing": [],
                    "rows": rows,
                    "by_scene": by_scene,
                    "used_dirs": [label] if rows else [],
                    "image_dir_label": label,
                },
            ))
    if not any(result["rows"] for _heading, result in sections):
        raise SystemExit(f"no rendered frames for {out_path}")
    meta = list(meta_lines)
    meta.append(f"Elapsed: {time.time() - started:.1f}s")
    write_report(out_path, title, meta, sections)
    print(f"Wrote {out_path}", flush=True)
    for heading, result in sections:
        stats = _bundle(result["rows"])
        if stats is None:
            print(f"  {heading}: no frames", flush=True)
            continue
        print(
            f"  {heading}: n={stats['cbsr']['n']}  "
            f"CBSR={stats['cbsr']['mean']:.4f}  PD={stats['pd']['mean']:.4f}",
            flush=True,
        )
    return out_path


def main():
    # CBSR and PD scoring is paused. Uncomment the block below to resume.
    print("CBSR and PD scoring is paused.", flush=True)
    return
    # args = parse_args()
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # out_path = os.path.join(args.render_root, f"consistency_{timestamp}.txt")
    # image_dirs = args.image_dir if args.image_dir else [None]
    # run_measurement(
    #     [("WideDrive", args.render_root, args.val_list)],
    #     image_dirs,
    #     out_path,
    #     "DGGT WideDrive CBSR and PD",
    #     [
    #         f"Render root: {args.render_root}",
    #         f"Val list: {args.val_list}",
    #     ],
    # )


if __name__ == "__main__":
    main()
