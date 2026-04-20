import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as TF


CAMERA_GROUPS = {
    "front": [0, 1, 2],
    "back": [5, 4, 3],
}

# Camera mapping based on file naming convention {timestep}_{cam_id}.jpg
# 0: CAM_FRONT
# 1: CAM_FRONT_LEFT
# 2: CAM_FRONT_RIGHT
# 3: CAM_BACK_LEFT
# 4: CAM_BACK_RIGHT
# 5: CAM_BACK


def _load_scene_list(scene_list_file: str) -> List[str]:
    with open(scene_list_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _parse_frame_file_name(file_name: str):
    match = re.match(r"^(\d+)_([0-5])\.(jpg|jpeg|png)$", file_name, flags=re.IGNORECASE)
    if match is None:
        return None
    timestep = int(match.group(1))
    cam_id = int(match.group(2))
    return timestep, cam_id


def _build_frame_map(folder: str) -> Dict[int, Dict[int, str]]:
    frame_map: Dict[int, Dict[int, str]] = defaultdict(dict)
    if not os.path.isdir(folder):
        return frame_map

    for file_name in os.listdir(folder):
        parsed = _parse_frame_file_name(file_name)
        if parsed is None:
            continue
        timestep, cam_id = parsed
        frame_map[timestep][cam_id] = os.path.join(folder, file_name)
    return frame_map


def _build_pose_map(folder: str) -> Dict[int, str]:
    pose_map: Dict[int, str] = {}
    if not os.path.isdir(folder):
        return pose_map

    for file_name in os.listdir(folder):
        match = re.match(r"^(\d+)\.txt$", file_name, flags=re.IGNORECASE)
        if match is None:
            continue
        timestep = int(match.group(1))
        pose_map[timestep] = os.path.join(folder, file_name)
    return pose_map


def _load_pose_matrix(path: str) -> np.ndarray:
    pose = np.loadtxt(path).reshape(4, 4)
    return pose


def _resize_and_to_tensor(image: Image.Image, target_size: int, is_mask: bool = False) -> torch.Tensor:
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)

    if is_mask:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    width, height = image.size
    new_width = target_size
    new_height = round(height * (new_width / width) / 14) * 14
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    image = image.resize((new_width, new_height), resample)

    to_tensor = TF.ToTensor()
    tensor = to_tensor(image)

    if new_height > target_size:
        start_y = (new_height - target_size) // 2
        tensor = tensor[:, start_y : start_y + target_size, :]

    return tensor


def _load_images(paths: Sequence[str], target_size: int, is_mask: bool = False) -> torch.Tensor:
    tensors = []
    shapes = set()

    for path in paths:
        image = Image.open(path)
        tensor = _resize_and_to_tensor(image, target_size=target_size, is_mask=is_mask)
        shapes.add((tensor.shape[1], tensor.shape[2]))
        tensors.append(tensor)

    if len(shapes) > 1:
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded_tensors = []
        for tensor in tensors:
            h_padding = max_height - tensor.shape[1]
            w_padding = max_width - tensor.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                tensor = torch.nn.functional.pad(
                    tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0 if not is_mask else 0.0,
                )
            padded_tensors.append(tensor)
        tensors = padded_tensors

    return torch.stack(tensors)


class NuScenesTemporalMultiCamDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        scene_list_file: str,
        sequence_length: int = 3,
        num_cameras_per_frame: int = 3,
        target_size: int = 518,
        max_interval: int = 5,
        is_val: bool = False,
    ):
        if sequence_length != 3:
            raise ValueError("This dataset is designed for exactly 3 frames per sample.")
        if num_cameras_per_frame != 3:
            raise ValueError("This dataset is designed for exactly 3 cameras per frame.")

        self.root_dir = root_dir
        self.scene_names = _load_scene_list(scene_list_file)
        self.target_size = target_size
        self.max_interval = max_interval
        self.sequence_length = sequence_length
        self.num_cameras_per_frame = num_cameras_per_frame
        self.is_val = is_val

        self.scene_entries = []
        self.train_sequence_index = []
        self.train_sequence_count_by_scene = {}
        for scene_name in self.scene_names:
            scene_dir = os.path.join(self.root_dir, scene_name)
            image_dir = os.path.join(scene_dir, "images")
            mask_dir = os.path.join(scene_dir, "fine_dynamic_masks", "all")
            ego_pose_dir = os.path.join(scene_dir, "ego_pose")

            image_map = _build_frame_map(image_dir)
            mask_map = _build_frame_map(mask_dir)
            ego_pose_map = _build_pose_map(ego_pose_dir)

            if len(image_map) == 0:
                continue

            valid_timesteps = {}
            for group_name, cam_ids in CAMERA_GROUPS.items():
                valid = []
                for timestep in sorted(set(image_map.keys()) & set(mask_map.keys())):
                    image_cams = image_map[timestep]
                    mask_cams = mask_map[timestep]
                    has_pose = timestep in ego_pose_map
                    if has_pose and all(cam_id in image_cams for cam_id in cam_ids) and all(cam_id in mask_cams for cam_id in cam_ids):
                        valid.append(timestep)
                valid_timesteps[group_name] = valid

            if len(valid_timesteps["front"]) == 0 and len(valid_timesteps["back"]) == 0:
                continue

            self.scene_entries.append(
                {
                    "scene_name": scene_name,
                    "image_map": image_map,
                    "mask_map": mask_map,
                    "ego_pose_map": ego_pose_map,
                    "valid_timesteps": valid_timesteps,
                }
            )

            # For training, each valid newest timestep is treated as one sequence anchor.
            if not self.is_val:
                anchor_candidates = sorted(set(valid_timesteps["front"]) | set(valid_timesteps["back"]))
                self.train_sequence_count_by_scene[scene_name] = len(anchor_candidates)
                scene_idx = len(self.scene_entries) - 1
                for anchor_timestep in anchor_candidates:
                    self.train_sequence_index.append((scene_idx, anchor_timestep))

        if len(self.scene_entries) == 0:
            raise RuntimeError(f"No valid nuScenes scenes found in {root_dir} using {scene_list_file}")

    def __len__(self) -> int:
        if self.is_val:
            return len(self.scene_entries)
        return len(self.train_sequence_index)

    def get_dataset_stats(self):
        if self.is_val:
            return {
                "num_scenes": len(self.scene_entries),
                "num_sequences": len(self.scene_entries),
                "sequences_per_scene": None,
            }
        return {
            "num_scenes": len(self.scene_entries),
            "num_sequences": len(self.train_sequence_index),
            "sequences_per_scene": self.train_sequence_count_by_scene,
        }

    def _sample_temporal_window(self, valid_timesteps: List[int]) -> Tuple[int, int, int, int]:
        if len(valid_timesteps) < 3:
            raise RuntimeError("Not enough valid timesteps to sample a 3-frame window.")

        # Validation uses deterministic interval=3.
        if self.is_val:
            interval = 3
            valid_set = set(valid_timesteps)
            candidates = [t for t in sorted(valid_timesteps) if (t + interval) in valid_set and (t + 2 * interval) in valid_set]
            if not candidates:
                raise RuntimeError("Validation requires fixed interval=3, but no valid (t, t+3, t+6) triplet was found.")
            t0 = candidates[0]
            return t0, t0 + interval, t0 + 2 * interval, interval

        valid_set = set(valid_timesteps)
        for _ in range(100):
            interval = random.randint(1, self.max_interval)
            candidates = [t for t in valid_timesteps if (t + interval) in valid_set and (t + 2 * interval) in valid_set]
            if candidates:
                start_timestep = random.choice(candidates)
                return start_timestep, start_timestep + interval, start_timestep + 2 * interval, interval

        raise RuntimeError("Failed to sample a valid 3-frame temporal window.")

    def _sample_temporal_window_train(self, valid_timesteps: List[int], anchor_timestep: int) -> Tuple[int, int, int, int]:
        if len(valid_timesteps) < 3:
            raise RuntimeError("Not enough valid timesteps to sample a 3-frame window.")

        valid_set = set(valid_timesteps)

        # Prefer using the indexed anchor as the newest frame.
        for _ in range(100):
            interval = random.randint(1, self.max_interval)
            t0 = anchor_timestep - 2 * interval
            t1 = anchor_timestep - interval
            t2 = anchor_timestep
            if t0 in valid_set and t1 in valid_set and t2 in valid_set:
                return t0, t1, t2, interval

        # Fallback to any valid triplet in this scene/group.
        for _ in range(100):
            interval = random.randint(1, self.max_interval)
            candidates = [
                t for t in valid_timesteps
                if (t - interval) in valid_set and (t - 2 * interval) in valid_set
            ]
            if candidates:
                t2 = random.choice(candidates)
                return t2 - 2 * interval, t2 - interval, t2, interval

        raise RuntimeError("Failed to sample a valid 3-frame temporal window for training.")

    def __getitem__(self, idx: int):
        if self.is_val:
            scene_entry = self.scene_entries[idx]
            anchor_timestep = None
        else:
            scene_idx, anchor_timestep = self.train_sequence_index[idx]
            scene_entry = self.scene_entries[scene_idx]

        scene_name = scene_entry["scene_name"]
        image_map = scene_entry["image_map"]
        mask_map = scene_entry["mask_map"]
        ego_pose_map = scene_entry["ego_pose_map"]

        if self.is_val:
            group_name = "front" if len(scene_entry["valid_timesteps"]["front"]) > 0 else "back"
        else:
            group_name = "front" if random.random() < 0.5 else "back"
        valid_timesteps = scene_entry["valid_timesteps"][group_name]

        if len(valid_timesteps) == 0:
            group_name = "back" if group_name == "front" else "front"
            valid_timesteps = scene_entry["valid_timesteps"][group_name]

        if self.is_val:
            start_timestep, middle_timestep, end_timestep, interval = self._sample_temporal_window(valid_timesteps)
        else:
            start_timestep, middle_timestep, end_timestep, interval = self._sample_temporal_window_train(valid_timesteps, anchor_timestep)
        # Model input order is newest -> oldest.
        frame_timesteps = [end_timestep, middle_timestep, start_timestep]
        cam_ids = CAMERA_GROUPS[group_name]

        image_paths = []
        mask_paths = []
        view_frame_ids = []
        view_camera_ids = []
        timestamps = []
        frame_ego_to_world = []

        max_time = 2 * interval
        for frame_offset, timestep in enumerate(frame_timesteps):
            frame_ego_to_world.append(_load_pose_matrix(ego_pose_map[timestep]))
            for cam_id in cam_ids:
                image_paths.append(image_map[timestep][cam_id])
                mask_paths.append(mask_map[timestep][cam_id])
                view_frame_ids.append(timestep)
                view_camera_ids.append(cam_id)
            timestamps.append(max_time - frame_offset * interval)

        # Normalize using the newest frame as Ego0, and output EgoN->Ego0 transforms.
        ego0_world_to_ego = np.linalg.inv(frame_ego_to_world[0])
        frame_ego_to_ego0 = [ego0_world_to_ego @ ego_to_world for ego_to_world in frame_ego_to_world]
        view_ego_to_ego0 = []
        for pose in frame_ego_to_ego0:
            for _ in cam_ids:
                view_ego_to_ego0.append(pose)

        images = _load_images(image_paths, target_size=self.target_size, is_mask=False)
        dynamic_mask = _load_images(mask_paths, target_size=self.target_size, is_mask=True)

        return {
            "images": images,
            "dynamic_mask": dynamic_mask,
            "scene_name": scene_name,
            "camera_group": group_name,
            "frame_ids": torch.tensor(frame_timesteps, dtype=torch.long),
            "camera_ids": torch.tensor(cam_ids, dtype=torch.long),
            "view_frame_ids": torch.tensor(view_frame_ids, dtype=torch.long),
            "view_camera_ids": torch.tensor(view_camera_ids, dtype=torch.long),
            "timestamps": torch.tensor(timestamps, dtype=torch.float32),
            # "frame_ego_to_ego0": torch.tensor(np.stack(frame_ego_to_ego0), dtype=torch.float32),
            # "ego_to_ego0": torch.tensor(np.stack(view_ego_to_ego0), dtype=torch.float32),
            "interval": interval,
            # "image_paths": image_paths,
            # "dynamic_mask_paths": mask_paths,
        }