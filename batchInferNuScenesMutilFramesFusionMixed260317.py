import os
import sys
import torch
import re
import math
from pathlib import Path
from torchvision.utils import save_image
from tqdm import tqdm

# ================= 项目路径设置 =================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image_nuscenes
# CUDA_VISIBLE_DEVICES=1 python batchInferNuScenesMutilFramesFusionMixed260317.py
# ================= 配置区域 =================
BATCH_SIZE = 4
RENDER_H = 252
TARGET_W = 448
WIDE_W = 896
NUMBER_FRAMES = 3
FRAME_INTERVAL = 5

# [推理精度设置]
USE_BF16 = False  # 全局配置：是否尝试使用 BF16

ENABLE_FUSION = False
BLEND_EDGE_WIDTH = 100
FUSION_METHOD = 'two_band'

# PRETRAINED_PATH = "AnySplat_1218infer/finetune_weights/260129_singleFramesReTrainGSHeadEpoch2Iter25000/weights"
PRETRAINED_PATH = "outputs/exp_nuScenes_reTrain_mutilFrames/2026-03-15_00-30-30_onlyTrainDynHead10w/checkpoints/epoch_6-step_100000"
DATASET_ROOT = "AnySplat_1218infer/datasets/nuscenes/processed_10Hz/trainval2"
VAL_LIST_PATH = "nuScenes_Val.txt"

SAVE_ROOT = f"./renders_val_MutilFrames_Intreval{FRAME_INTERVAL}/xxxx"

if ENABLE_FUSION:
    folder_suffix = f"fusion_{FUSION_METHOD}_{BLEND_EDGE_WIDTH}px"
else:
    folder_suffix = "render_only"

SAVE_ROOT = SAVE_ROOT.replace('xxxx', PRETRAINED_PATH.split('/')[-1]).replace('fusion', folder_suffix)

# ================= 工具函数 =================

def load_scene_list(txt_path):
    if not os.path.exists(txt_path):
        print(f"Error: Val list not found at {txt_path}")
        return []
    with open(txt_path, 'r') as f:
        scenes = [line.strip() for line in f.readlines() if line.strip()]
    return scenes

def get_scene_frames(scene_path):
    img_dir = os.path.join(scene_path, "images")
    if not os.path.exists(img_dir):
        return []
    frame_ids = []
    pattern = re.compile(r"^(.*)_0\.(jpg|png|jpeg)$", re.IGNORECASE)
    for f in sorted(os.listdir(img_dir)):
        match = pattern.match(f)
        if match:
            frame_ids.append(match.group(1))
    return sorted(frame_ids)

def get_image_paths_for_frame(scene_path, frame_ids, group_type):
    img_dir = os.path.join(scene_path, "images")
    if group_type == 'front':
        indices = [0, 1, 2]
    else: 
        indices = [5, 4, 3]

    paths = []
    for frame_id in frame_ids:
        for idx in indices:
            found = False
            for ext in ['.jpg', '.png', '.jpeg']:
                p = os.path.join(img_dir, f"{frame_id}_{idx}{ext}")
                if os.path.exists(p):
                    paths.append(p)
                    found = True
                    break
            if not found:
                return None, None
    return paths, indices

def create_edge_blend_mask(h, w, edge_width, device, strategy='smoothstep'):
    mask = torch.ones((1, h, w), device=device)
    if edge_width <= 0: return mask
    
    t = torch.linspace(0, 1, edge_width, device=device)
    if strategy == 'smoothstep':
        weight = 3 * t**2 - 2 * t**3
    else:
        weight = t
    
    mask[:, :, :edge_width] = weight.view(1, 1, -1)
    mask[:, :, -edge_width:] = torch.flip(weight, dims=[0]).view(1, 1, -1)
    return mask

def two_band_blending(render_img, gt_img, mask, blur_sigma=5):
    import torchvision
    blurrer = torchvision.transforms.GaussianBlur(kernel_size=21, sigma=blur_sigma)
    render_low = blurrer(render_img)
    gt_low = blurrer(gt_img)
    render_high = render_img - render_low
    gt_high = gt_img - gt_low
    
    result_low = gt_low * mask + render_low * (1 - mask)
    result_high = gt_high * mask + render_high * (1 - mask)
    return result_low + result_high

def blend_images(render_img, gt_img, mask):
    if render_img.shape != gt_img.shape:
        gt_resized = torch.nn.functional.interpolate(
            gt_img.unsqueeze(0), 
            size=render_img.shape[-2:], 
            mode='bilinear'
        ).squeeze(0)
    else:
        gt_resized = gt_img
    return gt_resized * mask + render_img * (1 - mask)

# ================= 主流程 =================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {PRETRAINED_PATH}...")
    model = AnySplat.from_pretrained(PRETRAINED_PATH).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # [修复] 使用局部变量，从全局配置读取
    use_bf16 = USE_BF16
    
    # 检查是否支持 bfloat16
    if use_bf16:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            print("⚠ Warning: BF16 not supported on this device, falling back to FP32")
            use_bf16 = False
        else:
            print("✓ Using BF16 mixed precision for inference")
    else:
        print("Using FP32 precision")
    
    # 创建融合 mask
    fusion_mask = None
    if ENABLE_FUSION:
        fusion_mask = create_edge_blend_mask(RENDER_H, TARGET_W, BLEND_EDGE_WIDTH, device)

    # 1. 收集所有任务
    scenes = load_scene_list(VAL_LIST_PATH)
    all_tasks = []
    
    print("Scanning scenes to collect tasks...")
    for scene_id in tqdm(scenes, desc="Scanning"):
        scene_dir = os.path.join(DATASET_ROOT, scene_id)
        frames = get_scene_frames(scene_dir)
        required = NUMBER_FRAMES
        max_start_idx = len(frames) - ((required - 1) * FRAME_INTERVAL + 1)
        if max_start_idx < 0:
            continue
        
        scene_save_dir = os.path.join(SAVE_ROOT, scene_id)
        os.makedirs(scene_save_dir, exist_ok=True)

        for start_idx in range(0, max_start_idx + 1):
            sampled_frames = frames[start_idx : start_idx + required * FRAME_INTERVAL : FRAME_INTERVAL]
            if len(sampled_frames) < required:
                continue
            # 与 dataset_nuscenes 保持一致：最新帧放在最前
            sampled_frames = sampled_frames[::-1]
            current_frame_id = sampled_frames[0]
            for group_type in ['front', 'back']:
                all_tasks.append({
                    'scene_id': scene_id,
                    'frame_id': current_frame_id,
                    'frame_ids': sampled_frames,
                    'group_type': group_type,
                    'scene_dir': scene_dir,
                    'save_dir': scene_save_dir
                })

    total_tasks = len(all_tasks)
    num_batches = math.ceil(total_tasks / BATCH_SIZE)
    print(f"Total tasks: {total_tasks}. Batch size: {BATCH_SIZE}. Total batches: {num_batches}")
    print(f"Fusion Enabled: {ENABLE_FUSION}")
    print(f"BF16 Precision: {use_bf16}")
    print(f"Saving to {SAVE_ROOT}")

    # 2. 按 Batch 处理
    for i in tqdm(range(num_batches), desc="Processing Batches"):
        batch_tasks_meta = all_tasks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        
        batch_images = []
        valid_tasks = []
        
        # 加载数据
        for task in batch_tasks_meta:
            paths, indices = get_image_paths_for_frame(task['scene_dir'], task['frame_ids'], task['group_type'])
            if paths:
                # print(f"Loaded {len(paths)} images for task {task['scene_id']} frame {task['frame_id']} group {task['group_type']}")
                # print(f"Paths: {paths}")
                imgs = [process_image_nuscenes(p) for p in paths]
                batch_images.append(torch.stack(imgs))
                
                task['cam_indices'] = indices
                
                if ENABLE_FUSION:
                    task['input_images_cpu'] = torch.stack(imgs)
                
                valid_tasks.append(task)
        
        if not valid_tasks:
            continue
        
        # 堆叠 Batch
        input_tensor = torch.stack(batch_images).to(device)
        current_bs = input_tensor.shape[0]
        views_per_group = 3
        
        # [修复] 使用局部变量 use_bf16
        with torch.no_grad():
            # 创建 autocast 上下文
            # autocast_context = torch.autocast(
            #     device_type='cuda', 
            #     dtype=torch.bfloat16, 
            #     enabled=use_bf16
            # )
            
            # with autocast_context:
            # Batch Inference
            gaussians, _, pred_context_pose = model.inference((input_tensor + 1) * 0.5)
            
            # Modify Intrinsics
            new_intrinsics = pred_context_pose['intrinsic'][:, :views_per_group].clone()
            width_scale = TARGET_W / WIDE_W
            new_intrinsics[..., 0, 0] *= width_scale
            
            # Prepare depth range
            t_near = torch.ones(current_bs, views_per_group, device=device) * 0.01
            t_far = torch.ones(current_bs, views_per_group, device=device) * 100.0
            
            # Batch Render
            outputs = model.decoder.forward(
                gaussians,
                pred_context_pose['extrinsic'][:, :views_per_group],
                new_intrinsics.float(),
                t_near,
                t_far,
                (RENDER_H, WIDE_W),
                "depth"
            )
            
            # 转回 FP32
            # rendered_batch = outputs.color.float()
            rendered_batch = outputs.color
            
            # 3. 保存与融合
            for b in range(current_bs):
                task = valid_tasks[b]
                group_render = rendered_batch[b]
                
                if ENABLE_FUSION:
                    group_input_gt = task['input_images_cpu'].to(device)
                
                for v in range(views_per_group):
                    cam_idx = task['cam_indices'][v]
                    pred_img = group_render[v]
                    
                    # 保存原始宽图
                    wide_name = f"{task['frame_id']}_{cam_idx}_wide.jpg"
                    save_image(pred_img, os.path.join(task['save_dir'], wide_name))
                    
                    # 处理融合
                    if ENABLE_FUSION and v == 0:
                        gt_img = (group_input_gt[v] + 1) * 0.5
                        
                        start_x = (WIDE_W - TARGET_W) // 2
                        pred_crop = pred_img[:, :, start_x : start_x + TARGET_W]
                        
                        if FUSION_METHOD == 'simple':
                            fusion_crop = blend_images(pred_crop, gt_img, fusion_mask)
                        elif FUSION_METHOD == 'two_band':
                            fusion_crop = two_band_blending(pred_crop, gt_img, fusion_mask)
                        else:
                            fusion_crop = pred_crop
                        
                        pred_img_fusion = pred_img.clone()
                        pred_img_fusion[:, :, start_x : start_x + TARGET_W] = fusion_crop
                        
                        wide_fusion_name = f"{task['frame_id']}_{cam_idx}_wide_fusion.jpg"
                        save_image(pred_img_fusion, os.path.join(task['save_dir'], wide_fusion_name))
        if i % 10 == 0 and i > 0:    
            torch.cuda.empty_cache()
    print(f"✓ Rendering Finished. Saved to {SAVE_ROOT}")

if __name__ == "__main__":
    main()