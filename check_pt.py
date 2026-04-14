import torch
import os

def inspect_and_split_checkpoint(checkpoint_path, save_dir='splitPtWeights'):
    print(f"Inspecting checkpoint: {checkpoint_path}")
    os.makedirs(save_dir, exist_ok=True)
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Determine if we're dealing with a state_dict or a structured checkpoint
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                print("Found 'state_dict' key, using it for splitting.")
                state_dict = checkpoint['state_dict']
            elif any('.' in k for k in checkpoint.keys()):
                print("Keys contain '.', treating the main object as state_dict.")
                state_dict = checkpoint
            else:
                print("No clear state_dict structure found in main keys. Inspecting top-level keys as potential categories.")
        
        if state_dict:
            # Group keys by their top-level name (e.g., 'aggregator', 'encoder')
            groups = {}
            for key in state_dict.keys():
                top_level_name = key.split('.')[0]
                if top_level_name not in groups:
                    groups[top_level_name] = {}
                groups[top_level_name][key] = state_dict[key]
            
            print(f"\n=== Splitting by top-level parameter names ({len(groups)} groups) ===")
            for name, weights in groups.items():
                save_path = os.path.join(save_dir, f"{name}.pt")
                torch.save(weights, save_path)
                print(f"- {name}: contains {len(weights)} parameters -> saved to {save_path}")
        else:
            # If it's just a simple dictionary without dots, save each top key
            print("\n=== No nested parameter dots found, saving top-level keys directly ===")
            for key, val in checkpoint.items():
                save_path = os.path.join(save_dir, f"{key}.pt")
                torch.save(val, save_path)
                print(f"- {key}: saved to {save_path}")
                        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    inspect_and_split_checkpoint('./model_latest_waymo.pt')
