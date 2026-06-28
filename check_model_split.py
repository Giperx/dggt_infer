import torch
from safetensors.torch import load_file
import os
import argparse
from collections import OrderedDict


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in ("true", "1", "yes", "y", "on"):
        return True
    if val in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("布尔值应为 true/false")


def load_state_dict_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".safetensors":
        return load_file(file_path)

    # 兼容 Lightning 的 .ckpt 以及普通 .pt/.pth 文件
    checkpoint = torch.load(file_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]

        # 某些权重文件直接就是 state_dict
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint

    raise ValueError(
        "无法从该文件中解析 state_dict。"
        "若是 Lightning ckpt，请确认其中包含 'state_dict' 字段。"
    )


def get_module_root_for_aggregator(keys):
    """自动推断 aggregator 所在的父级前缀，例如 model。"""
    candidates = []
    found_top_level = False
    for key in keys:
        parts = key.split(".")
        for index, part in enumerate(parts):
            if part == "aggregator":
                parent = ".".join(parts[:index])
                if parent:
                    candidates.append(parent)
                else:
                    found_top_level = True
                break

    if not candidates and not found_top_level:
        return None

    if not candidates and found_top_level:
        # aggregator 位于顶层，无父级前缀。
        return ""

    # 优先选择最短的父级，一般就是 model。
    return sorted(set(candidates), key=len)[0]


def collect_sibling_module_prefixes(state_dict, module_root):
    """收集 module_root 同级下的所有一级子模块前缀。"""
    module_prefixes = OrderedDict()
    is_top_level = module_root == ""
    if is_top_level:
        root_prefix = ""
        root_depth = 0
    else:
        root_prefix = module_root + "."
        root_depth = len(module_root.split("."))

    for key in state_dict.keys():
        if not is_top_level and not key.startswith(root_prefix):
            continue

        parts = key.split(".")
        if len(parts) <= root_depth:
            continue

        module_prefix = ".".join(parts[: root_depth + 1])
        module_prefixes.setdefault(module_prefix, [])
        module_prefixes[module_prefix].append(key)

    return module_prefixes


def extract_module_state(state_dict, module_prefix):
    module = OrderedDict(
        (k[len(module_prefix):], v)
        for k, v in state_dict.items()
        if k.startswith(module_prefix)
    )
    if not module:
        raise KeyError(f"未找到模块参数，prefix={module_prefix}")
    return module


def merge_lora_module_state(state_dict, module_prefix, lora_alpha=32.0):
    """合并指定模块前缀下的 LoRA 线性层，并返回模块内 state_dict。"""
    module_state = extract_module_state(state_dict, module_prefix)
    merged = OrderedDict()

    lora_bases = [
        k[: -len(".lora_down.weight")]
        for k in module_state.keys()
        if k.endswith(".lora_down.weight")
    ]
    lora_base_set = set(lora_bases)

    for base in sorted(lora_base_set):
        linear_w_key = f"{base}.linear.weight"
        linear_b_key = f"{base}.linear.bias"
        down_key = f"{base}.lora_down.weight"
        up_key = f"{base}.lora_up.weight"

        missing = [k for k in (linear_w_key, down_key, up_key) if k not in module_state]
        if missing:
            raise KeyError(f"LoRA 合并缺少键，base={base}, missing={missing}")

        linear_w = module_state[linear_w_key].float()
        down_w = module_state[down_key].float()
        up_w = module_state[up_key].float()

        rank = down_w.shape[0]
        if rank <= 0:
            raise ValueError(f"无效 LoRA rank，base={base}, rank={rank}")

        scaling = float(lora_alpha) / float(rank)
        merged_w = (linear_w + torch.matmul(up_w, down_w) * scaling).to(
            dtype=module_state[linear_w_key].dtype
        )
        merged[f"{base}.weight"] = merged_w
        if linear_b_key in module_state:
            merged[f"{base}.bias"] = module_state[linear_b_key]

    for key, value in module_state.items():
        if key.endswith(".lora_down.weight") or key.endswith(".lora_up.weight"):
            continue

        copied_key = key
        if key.endswith(".linear.weight"):
            base = key[: -len(".linear.weight")]
            if base in lora_base_set:
                continue
            copied_key = f"{base}.weight"
        elif key.endswith(".linear.bias"):
            base = key[: -len(".linear.bias")]
            if base in lora_base_set:
                continue
            copied_key = f"{base}.bias"

        if copied_key not in merged:
            merged[copied_key] = value

    return merged, len(lora_base_set)


def save_ckpt(path, module_state_dict, meta=None):
    payload = {
        "state_dict": module_state_dict,
        "meta": meta or {},
    }
    torch.save(payload, path)


def human_size(num_bytes):
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def export_sibling_modules(file_path, output_dir=None, lora_alpha=32.0, state_dict=None):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    if output_dir is None:
        file_dir = os.path.dirname(file_path)
        file_stem = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(file_dir, f"{file_stem}_merged_ckpts")

    os.makedirs(output_dir, exist_ok=True)

    if state_dict is None:
        print(f"正在加载: {file_path}")
        state_dict = load_state_dict_from_file(file_path)

    module_root = get_module_root_for_aggregator(state_dict.keys())
    if module_root is None:
        raise KeyError("未找到 aggregator，无法自动推断导出层级")

    sibling_modules = collect_sibling_module_prefixes(state_dict, module_root)
    if not sibling_modules:
        raise KeyError(f"在 {module_root} 下未找到可导出的同级模块")

    exported = []
    for module_prefix in sibling_modules.keys():
        leaf_name = module_prefix.split(".")[-1]
        save_path = os.path.join(output_dir, f"{leaf_name}.pt")

        if leaf_name == "aggregator":
            module_state, merged_count = merge_lora_module_state(
                state_dict,
                module_prefix,
                lora_alpha=lora_alpha,
            )
            meta = {
                "source_ckpt": os.path.abspath(file_path),
                "module": module_prefix,
                "lora_merged": True,
                "lora_alpha": float(lora_alpha),
                "merged_lora_layers": int(merged_count),
            }
        else:
            module_state = extract_module_state(state_dict, module_prefix)
            meta = {
                "source_ckpt": os.path.abspath(file_path),
                "module": module_prefix,
                "lora_merged": False,
            }

        save_ckpt(save_path, module_state, meta=meta)
        exported.append((module_prefix, save_path, len(module_state)))

    print(f"[OK] source: {os.path.abspath(file_path)}")
    detected_root = module_root if module_root else "<top-level>"
    print(f"[OK] detected module root: {detected_root}")
    print(f"[OK] exported modules: {len(exported)}")
    for module_prefix, save_path, key_count in exported:
        print(f"  - {module_prefix} -> {save_path}  ({key_count} keys)")


def inspect_weights(file_path, max_modules=12, max_keys_per_module=3, state_dict=None):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    if state_dict is None:
        print(f"正在加载: {file_path}")
        state_dict = load_state_dict_from_file(file_path)
    keys = list(state_dict.keys())
    keys.sort()
    total_tensors = len(keys)
    total_params = 0
    total_bytes = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total_params += value.numel()
            total_bytes += value.numel() * value.element_size()

    module_root = get_module_root_for_aggregator(keys)
    if module_root is not None:
        module_prefixes = collect_sibling_module_prefixes(state_dict, module_root)
        module_items = list(module_prefixes.items())
        detected_root = module_root if module_root else "<top-level>"
        scope_label = f"aggregator 同级模块 (parent={detected_root})"
    else:
        module_prefixes = OrderedDict()
        for key in keys:
            prefix = key.split('.')[0]
            module_prefixes.setdefault(prefix, []).append(key)
        module_items = list(module_prefixes.items())
        scope_label = "顶层模块"

    print("\n" + "=" * 60)
    print(f"总计权重键数: {total_tensors}")
    print(f"总参数量: {total_params:,}")
    print(f"估计权重大小: {human_size(total_bytes)}")
    print("=" * 60)

    print(f"\n模块概览: {scope_label}")
    shown = module_items[:max_modules]
    for mod_name, mod_keys in shown:
        sample = ", ".join(mod_keys[:max_keys_per_module])
        suffix = " ..." if len(mod_keys) > max_keys_per_module else ""
        print(f"- {mod_name:35} ({len(mod_keys):4} 个键)  示例: {sample}{suffix}")

    if len(module_items) > max_modules:
        print(f"... 其余 {len(module_items) - max_modules} 个模块已省略")

    print("\n提示: 如需导出同级模块 pt，请加 --export-pt true")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查模型，或导出 aggregator 同级模块 pt（aggregator 自动合并 LoRA）")
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="权重文件路径，支持 .safetensors / .ckpt / .pt / .pth",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="导出 pt 的输出目录（仅在 --export-pt true 时生效）",
    )
    parser.add_argument(
        "--export-pt",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="是否导出 aggregator 同级所有模块的 pt；其中 aggregator 会自动合并 LoRA",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=32.0,
        help="LoRA alpha（默认 32）",
    )
    args = parser.parse_args()
    print(f"正在加载: {args.file}")
    state_dict = load_state_dict_from_file(args.file)

    inspect_weights(args.file, state_dict=state_dict)

    if args.export_pt:
        export_sibling_modules(
            args.file,
            output_dir=args.output_dir,
            lora_alpha=args.lora_alpha,
            state_dict=state_dict,
        )

# 仅检查 ckpt/safetensors，不导出
# python check_model_split.py --file /path/to/epoch_xx.ckpt

# 检查并导出 aggregator 同级所有模块 pt；其中 aggregator 会自动合并 LoRA
# python check_model_split.py --file /path/to/epoch_xx.ckpt --export-pt true --output-dir /path/to/exported_modules --lora-alpha 32