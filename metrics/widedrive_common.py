"""Pair WideDrive wide renders with full-pixel GT.

Render names vary (`023_wide.jpg`, `002_fixedfov_wide.jpg`, `023.png`).
GT is `{frame}_2.jpg` under each scene's `images/` directory and is dense.
A render mask is usable only when it has the same height and width as the RGB.
"""
import os
import re
from collections import Counter

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
RENDER_IMAGE_DIRS = ("before_affine_rgb", "before_rgb", "rgb")
GT_CAM_ID = 2


def load_scene_ids(render_root, val_list=None):
    """Scenes that actually have rendered images. Optional val list filters them."""
    if not os.path.isdir(render_root):
        raise FileNotFoundError(f"render root does not exist: {render_root}")
    rendered = []
    for name in sorted(os.listdir(render_root)):
        scene_dir = os.path.join(render_root, name)
        if not os.path.isdir(scene_dir):
            continue
        if any(os.path.isdir(os.path.join(scene_dir, sub)) for sub in RENDER_IMAGE_DIRS):
            rendered.append(name)
    if not val_list:
        return rendered, []
    if not os.path.exists(val_list):
        raise FileNotFoundError(f"val list does not exist: {val_list}")
    with open(val_list, "r") as handle:
        wanted = [line.strip() for line in handle if line.strip()]
    present = set(rendered)
    return [scene for scene in wanted if scene in present], [scene for scene in wanted if scene not in present]


def _split_stem(stem):
    match = re.match(r"^(\d+)(.*)$", stem)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _suffix_rank(suffix):
    if suffix in ("_wide", "wide"):
        return (0, len(suffix), suffix)
    if "wide" in suffix:
        return (1, len(suffix), suffix)
    return (2, len(suffix), suffix)


def choose_image_dir(scene_dir, requested=None):
    if requested:
        folder = os.path.join(scene_dir, requested)
        return requested if os.path.isdir(folder) else None
    for name in RENDER_IMAGE_DIRS:
        if os.path.isdir(os.path.join(scene_dir, name)):
            return name
    return None


def list_render_frames(scene_dir, image_dir):
    """One file per frame id. Mixed suffixes keep the most common suffix."""
    folder = os.path.join(scene_dir, image_dir)
    grouped = {}
    for name in os.listdir(folder):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        parsed = _split_stem(stem)
        if parsed is None:
            continue
        frame_id, suffix = parsed
        grouped.setdefault(frame_id, []).append((suffix, os.path.join(folder, name)))
    if not grouped:
        return []
    counts = Counter(suffix for items in grouped.values() for suffix, _ in items)
    best_count = max(counts.values())
    suffix = sorted((item for item, count in counts.items() if count == best_count), key=_suffix_rank)[0]
    frames = []
    for frame_id, items in grouped.items():
        matches = [path for item_suffix, path in items if item_suffix == suffix]
        if matches:
            frames.append((frame_id, suffix, matches[0]))
    return sorted(frames, key=lambda item: int(item[0]))


def _id_stems(frame_id):
    target = int(frame_id)
    stems = []
    for width in (len(str(frame_id)), 3, 4):
        stems.append(f"{target:0{width}d}")
    stems.append(str(target))
    seen = []
    for stem in stems:
        if stem not in seen:
            seen.append(stem)
    return seen


def _find_named(folder, stems):
    if not os.path.isdir(folder):
        return None
    for stem in stems:
        for ext in IMAGE_EXTS:
            path = os.path.join(folder, stem + ext)
            if os.path.exists(path):
                return path
    return None


def find_gt_image(gt_scene_dir, frame_id, gt_cam=GT_CAM_ID):
    """Camera-2 GT from the val tree or from sparseWideFOVImages3_{W}x{H}."""
    ids = _id_stems(frame_id)
    found = _find_named(
        os.path.join(gt_scene_dir, "images"),
        [f"{stem}_{gt_cam}" for stem in ids],
    )
    if found:
        return found
    images_dir = os.path.join(gt_scene_dir, "images")
    target = int(frame_id)
    if os.path.isdir(images_dir):
        for name in os.listdir(images_dir):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            parsed = _split_stem(stem)
            if parsed is None:
                continue
            number, rest = parsed
            if int(number) == target and rest in (f"_{gt_cam}", str(gt_cam)):
                return os.path.join(images_dir, name)
    return _find_named(
        os.path.join(gt_scene_dir, "rgb"),
        [f"{stem}_{gt_cam}_sparse_wide" for stem in ids],
    )


def find_render_mask(scene_dir, frame_id, suffix):
    mask_dir = os.path.join(scene_dir, "mask")
    if not os.path.isdir(mask_dir):
        return None
    stems = [f"{frame_id}{suffix}"]
    if suffix:
        stems.append(frame_id)
    for stem in stems:
        for ext in IMAGE_EXTS:
            path = os.path.join(mask_dir, stem + ext)
            if os.path.exists(path):
                return path
    target = int(frame_id)
    for name in os.listdir(mask_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        parsed = _split_stem(stem)
        if parsed is not None and int(parsed[0]) == target:
            return os.path.join(mask_dir, name)
    return None


def load_rgb(path, size_wh=None):
    image = Image.open(path).convert("RGB")
    if size_wh is not None and image.size != size_wh:
        image = image.resize(size_wh, Image.BICUBIC)
    return np.asarray(image).astype(np.float64) / 255.0


def load_render_mask(path, shape_hw):
    """Return a full-image bool mask, or None when the file is missing or degenerate."""
    if path is None or not os.path.exists(path):
        return None, "missing"
    mask = np.asarray(Image.open(path).convert("L"))
    expected = tuple(shape_hw)
    if mask.ndim != 2 or mask.shape[0] < 2 or mask.shape != expected:
        return None, f"invalid shape {mask.shape}, expected {expected}"
    return mask > 127, None


def region_bounds(width):
    third = width // 3
    return {
        "Left": (0, third),
        "Center": (third, third * 2),
        "Right": (third * 2, width),
    }


def iter_eval_frames(render_root, gt_root, scenes, image_dir=None, gt_cam=GT_CAM_ID):
    """Yield frames whose render exists. Missing GT is reported and skipped."""
    for scene in scenes:
        scene_dir = os.path.join(render_root, scene)
        selected_dir = choose_image_dir(scene_dir, image_dir)
        if selected_dir is None:
            yield {"scene": scene, "status": "no_render_dir"}
            continue
        frames = list_render_frames(scene_dir, selected_dir)
        if not frames:
            yield {"scene": scene, "status": "no_render_frames", "image_dir": selected_dir}
            continue
        for frame_id, suffix, render_path in frames:
            gt_path = find_gt_image(os.path.join(gt_root, scene), frame_id, gt_cam)
            mask_path = find_render_mask(scene_dir, frame_id, suffix)
            yield {
                "scene": scene,
                "frame": frame_id,
                "suffix": suffix,
                "image_dir": selected_dir,
                "render_path": render_path,
                "gt_path": gt_path,
                "mask_path": mask_path,
                "status": "ok" if gt_path else "missing_gt",
            }
