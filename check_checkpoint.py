import sys, torch
from pathlib import Path

existing = Path("models/v31_neural_model.pt")
if not existing.exists():
    print("models/v31_neural_model.pt does not exist!")
    sys.exit(0)

checkpoint = torch.load(str(existing))
print("Keys in checkpoint:", checkpoint.keys())
print("Step count:", checkpoint.get("step", 0))

model_state = checkpoint.get("model_state_dict", {})
any_nan = False
for k, v in model_state.items():
    has_nan = torch.isnan(v).any().item()
    print(f"Parameter {k}: shape={v.shape}, contains NaN={has_nan}")
    if has_nan:
        any_nan = True

if any_nan:
    print("CRITICAL: The existing checkpoint models/v31_neural_model.pt contains NaN weights!")
else:
    print("No NaN found in checkpoint weights.")
