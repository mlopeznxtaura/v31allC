import torch
from pathlib import Path

def inspect_file(filepath):
    # Check both Windows and WSL paths
    path = Path(filepath)
    if not path.exists():
        # Try WSL path
        wsl_path = Path(str(filepath).replace("D:", "/mnt/d"))
        if wsl_path.exists():
            path = wsl_path
        else:
            print(f"File {filepath} does not exist (tried {path} and {wsl_path}).")
            return
    try:
        data = torch.load(str(path), map_location="cpu")
        print(f"\n--- Inspecting {path.name} ---")
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for k, v in data.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: shape {v.shape}")
                else:
                    print(f"  {k}: {type(v)}")
        elif hasattr(data, "shape"):
            print(f"Shape: {data.shape}")
        else:
            print(f"Data representation: {str(data)[:200]}")
    except Exception as e:
        print(f"Error reading {path.name}: {e}")

if __name__ == "__main__":
    inspect_file("D:/NextAura/v31all_1/Pt's/layer1_combined.pt")
    inspect_file("D:/NextAura/v31all_1/Pt's/layer1_frozen.pt")
    inspect_file("D:/NextAura/v31all_1/Pt's/layer1_variance.pt")
    inspect_file("D:/NextAura/v31all_1/Pt's/l2_baby_grounded_v3.pt")
