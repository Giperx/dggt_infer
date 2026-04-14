import os
import sys
import torch
import re
import math
from tqdm import tqdm
# CUDA_VISIBLE_DEVICES=1 python tmpRun_nuscenes.py
# ================= 项目路径设置 =================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dggt.models.aggregator import Aggregator
from dggt.heads.dpt_head import DPTHead
from dggt.models.vggt import VGGT
from dggt.utils.load_fn import load_and_preprocess_images
from PIL import Image
import torchvision.transforms as T
# ================= 配置区域 =================
# 批处理大小 (根据显存大小调整，建议 2, 4, 8)
BATCH_SIZE = 4
NUMBER_FRAMES = 3
FRAME_INTERVAL = 3

RENDER_H = 252
TARGET_W = 448     # 原始/裁切目标宽度 (GT宽度)
WIDE_W = 896       # 渲染宽度

# --- 融合功能开关 ---
ENABLE_FUSION = False  # <--- [核心修改] True: 开启融合并保存fusion图; False: 仅保存原始render图

# 融合设置
BLEND_EDGE_WIDTH = 100 # 边缘渐变宽度 (像素)
FUSION_METHOD = 'two_band' # 'simple' or 'two_band'

PRETRAINED_PATH='./model_latest_waymo.pt'
DATASET_ROOT = "AnySplat_1218infer/datasets/nuscenes/processed_10Hz/trainval2"
# VAL_LIST_PATH = "AnySplat_1218infer/nuScenes_ValTmp2.txt"
VAL_LIST_PATH = "nuScenes_Val.txt"
# 保存路径处理
SAVE_ROOT = f"./renderMask_nuscenes_model_latest_waymo_pt_{FRAME_INTERVAL}" # 最终保存路径，内部会按场景分文件夹
# [核心修改] 根据开关调整文件夹命名，方便区分
if ENABLE_FUSION:
    folder_suffix = f"fusion_{FUSION_METHOD}_{BLEND_EDGE_WIDTH}px"
else:
    folder_suffix = "render_only" # 如果不融合，标记文件夹为纯渲染

# SAVE_ROOT = SAVE_ROOT.replace('xxxx', PRETRAINED_PATH.split('/')[-2]).replace('fusion', folder_suffix)

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
    # 扫描一次文件夹
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


def build_grid_image(gt_data, mask_data, prob_data):
    mask_rgb = mask_data.unsqueeze(1).repeat(1, 3, 1, 1)
    prob_rgb = prob_data.unsqueeze(1).repeat(1, 3, 1, 1)

    row_images = []
    for row_idx in range(gt_data.shape[0]):
        row_img = torch.cat([gt_data[row_idx], mask_rgb[row_idx], prob_rgb[row_idx]], dim=2)
        row_images.append(row_img)

    return torch.cat(row_images, dim=1)

# ================= 融合策略函数 =================

def create_edge_blend_mask(h, w, edge_width, device, strategy='smoothstep'):
    mask = torch.ones((1, h, w), device=device)
    if edge_width <= 0: return mask
    
    t = torch.linspace(0, 1, edge_width, device=device)
    if strategy == 'smoothstep':
        weight = 3 * t**2 - 2 * t**3
    else:
        weight = t # linear

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
    device = torch.device("cuda")
    dtype = torch.float32
    # model = VGGT().to(device)
    # checkpoint = torch.load(PRETRAINED_PATH, map_location="cpu")
    # model.load_state_dict(checkpoint, strict=False)
    aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
    instance_head = DPTHead(dim_in= 1024, output_dim = 1 + 1, activation="linear")
    aggregator.to(device)
    instance_head.to(device)
    # aggregator.load_state_dict(aggregator_checkpoint)
    # 移除 key 中的一级前缀（例如 "aggregator." 或 "instance_head."）后再加载
    def fix_state_dict(state_dict, prefix):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix + "."):
                new_state_dict[k[len(prefix)+1:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    aggregator_checkpoint = torch.load("splitPtWeights/aggregator.pt", map_location="cpu")
    aggregator.load_state_dict(fix_state_dict(aggregator_checkpoint, "aggregator"))
    
    instance_head_checkpoint = torch.load("splitPtWeights/instance_head.pt", map_location="cpu")
    instance_head.load_state_dict(fix_state_dict(instance_head_checkpoint, "instance_head"))
    
    # model = AnySplat.from_pretrained(PRETRAINED_PATH).to(device)
    # model.eval()
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # [核心修改] 仅当开启融合时才创建 Mask
    fusion_mask = None
    if ENABLE_FUSION:
        fusion_mask = create_edge_blend_mask(RENDER_H, TARGET_W, BLEND_EDGE_WIDTH, device)

    # 1. 收集所有任务（多帧窗口）
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

        # 确保该场景的保存目录存在
        scene_save_dir = os.path.join(SAVE_ROOT, scene_id)
        os.makedirs(scene_save_dir, exist_ok=True)
        
        for start_idx in range(0, max_start_idx + 1):
            sampled_frames = frames[start_idx : start_idx + required * FRAME_INTERVAL : FRAME_INTERVAL]
            if len(sampled_frames) < required:
                continue
            # 与训练/推理数据组织保持一致：最新帧在最前
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
    print(f"Saving to {SAVE_ROOT}")

    # 2. 按 Batch 处理
    frame_buffer = {} # 用于暂存不同组的视图以拼成完整帧

    for i in tqdm(range(num_batches), desc="Processing Batches"):
        # 获取当前 Batch 的任务
        batch_tasks_meta = all_tasks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        
        batch_images = []
        valid_tasks = [] # 记录有效任务（防止某些图片缺失）
        
        # 加载数据 (IO密集)
        for task in batch_tasks_meta:
            paths, indices = get_image_paths_for_frame(task['scene_dir'], task['frame_ids'], task['group_type'])
            if paths:
                # 加载 3*NUMBER_FRAMES 张图 -> [3*NUMBER_FRAMES, C, H, W]
                # imgs = [load_and_preprocess_images(p) for p in paths]
                imgs = load_and_preprocess_images(paths, target_size=TARGET_W)
                # imgs 已经是一个 Tensor [3*NUMBER_FRAMES, C, H, W]（因为一次性传了路径列表）
                # 直接加入列表即可
                batch_images.append(imgs)
                
                # 记录元数据供后续保存使用
                task['cam_indices'] = indices
                
                # [核心修改] 仅当开启融合时，才在 CPU 缓存原始图片用于融合计算
                # 这样如果只跑 render，可以节省大量内存
                if ENABLE_FUSION:
                    task['input_images_cpu'] = imgs
                
                valid_tasks.append(task)
        
        if not valid_tasks:
            continue
            
        # [显存管理] 定期清理缓存，防止由于 batch 累积导致的 OOM
        if i % 10 == 0 and i > 0:
            torch.cuda.empty_cache()
            
        # 堆叠 Batch -> [B, 3*NUMBER_FRAMES, 3, H, W]
        input_tensor = torch.stack(batch_images).to(device)
        current_bs = input_tensor.shape[0]
        views_per_group = 3 # 每帧固定3张图
        
        with torch.no_grad():
            # predictions = model(input_tensor)
            _, _, dino_token_list, _, patch_start_idx = aggregator(input_tensor)
            dynamic_conf, _ = instance_head(dino_token_list, input_tensor, patch_start_idx)
            
            # predictions["dynamic_conf"] 形状通常为 [B, 3*NUMBER_FRAMES, H, W, 1]
            # 将其提取出来并转为 mask
            dy_map = dynamic_conf.squeeze(-1) # [B, 3*NUMBER_FRAMES, H, W]
            dy_mask = (dy_map > 0.5).type(torch.uint8) * 255 # [B, 3*NUMBER_FRAMES, H, W]
            
            # 3. 处理 Batch 中的每个任务，按帧存放
            for b_idx in range(current_bs):
                task = valid_tasks[b_idx]
                key = (task['scene_id'], tuple(task['frame_ids']))
                if key not in frame_buffer:
                    frame_buffer[key] = {}
                
                # 存放该组的 3*NUMBER_FRAMES 个 GT/Mask/Prob
                # GT: [3*NUMBER_FRAMES, 3, H, W] -> 转为 [0, 255] uint8 CPU 数据
                gt_data = ((batch_images[b_idx]) * 225).clamp(0, 255).type(torch.uint8).cpu()
                mask_data = dy_mask[b_idx].cpu() # [3*NUMBER_FRAMES, H, W] 二值化掩码
                # 动态程度图 (sigmoid 概率图)
                prob_data = (dy_map[b_idx].sigmoid() * 255).type(torch.uint8).cpu() # [3*NUMBER_FRAMES, H, W]
                
                frame_buffer[key][task['group_type']] = {
                    'gt': gt_data,
                    'mask': mask_data,
                    'prob': prob_data,
                    'save_dir': task['save_dir']
                }

                # 如果 front 和 back 都齐了，分别输出 front/back 拼图
                if 'front' in frame_buffer[key] and 'back' in frame_buffer[key]:
                    fb = frame_buffer[key]
                    
                    # 1. Front: 3*NUMBER_FRAMES 行 x 3 列 (GT|Mask|Prob)
                    front_gt = fb['front']['gt']
                    front_mask = fb['front']['mask']
                    front_prob = fb['front']['prob']
                    front_combined = build_grid_image(front_gt, front_mask, front_prob)
                    
                    front_save_path = os.path.join(fb['front']['save_dir'], f"{task['frame_id']}_front.jpg")
                    T.ToPILImage()(front_combined).save(front_save_path)
                    
                    # 2. Back: 3*NUMBER_FRAMES 行 x 3 列 (GT|Mask|Prob)
                    back_gt = fb['back']['gt']
                    back_mask = fb['back']['mask']
                    back_prob = fb['back']['prob']
                    back_combined = build_grid_image(back_gt, back_mask, back_prob)
                    
                    back_save_path = os.path.join(fb['back']['save_dir'], f"{task['frame_id']}_back.jpg")
                    T.ToPILImage()(back_combined).save(back_save_path)
                    
                    # 清理缓存
                    del frame_buffer[key]

    print(f"Rendering Finished. Saved to {SAVE_ROOT}")

if __name__ == "__main__":
    main()