import sys
sys.path.insert(0, "/mnt/d/NextAura/v31all_1/v31allC")
from stage1.core.neural_l1_engine import NeuralL1State
from stage1.core.record_builder import RecordBuilder
from pathlib import Path

print("Testing NeuralL1State weight loading...")
state = NeuralL1State()
print(f"Is loaded with default checkpoint? {state.is_loaded}")
print(f"Loaded checkpoint path: {state.checkpoint_path}")

# Load specific checkpoints
stone_core_path = "/mnt/d/NextAura/v31all_1/Pt's/stone_core_english.pt"
print(f"Loading stone_core_english.pt from {stone_core_path}...")
state.load_weights(stone_core_path)
print(f"Is loaded? {state.is_loaded}")
print(f"Loaded checkpoint path: {state.checkpoint_path}")

# Build records with neural l1 enabled
builder = RecordBuilder(total_entries=5, mission="neural-test", codebase_hash="v31allC", neural_l1=state)
print("Building 3 records using Neural Layer 1...")
for i in range(3):
    rec = builder.build(
        binary_text="test binary",
        geometry_text="test geometry",
        language_text="test language context for neural l1 verification",
        stimulus=1.2
    )
    print(f"Record {i+1}:")
    print(f"  M1_raw: {rec['_meta']['M1_raw']:.6f}")
    print(f"  M2_raw: {rec['_meta']['M2_raw']:.6f}")
    print(f"  M3_raw: {rec['_meta']['M3_raw']:.6f}")
    print(f"  V: {rec['_meta']['V']:.6f}")
