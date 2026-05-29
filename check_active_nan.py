import torch
from pathlib import Path

path = Path("models/v31_neural_model.pt")
if not path.exists():
    print("Checkpoint file does not exist!")
else:
    checkpoint = torch.load(str(path), weights_only=True)
    model_state = checkpoint["model_state_dict"]
    has_nan = False
    for k, v in model_state.items():
        if torch.isnan(v).any():
            print(f"Weight {k} contains NaN!")
            has_nan = True
    if not has_nan:
        print("Checkpoint is clean (no NaN weights)!")
