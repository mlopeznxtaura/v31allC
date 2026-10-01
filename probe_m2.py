"""probe_m2.py — diagnose M2 flatline + gate threshold"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage3.inference_engine.main import V31InferenceEngine
from stage1.core.triangulation import build_mathematical_relationship

e = V31InferenceEngine()

# 1. What's driving M2 flat at 0.990?
print("=== M2 / GATE PROBE ===")
tests = [
    ("minimal", "short", "What is RL?"),
    ("standard", "Entry 1/10. SystemID=CASEBELIZE. TelemetryStable=True.", "What is reinforcement learning?"),
    ("rich", "Deep technical query. Complexity=HIGH. Domain=AI. Confidence=0.95. SystemID=CASEBELIZE. TelemetryStable=True. PhaseReady=True. Convergence=0.85.", "What is reinforcement learning and how does it relate to world models?"),
    ("max", "SystemID=CASEBELIZE. TelemetryStable=True. PhaseReady=True. Convergence=0.95. Complexity=MAXIMUM. Domain=AI.Architecture=MoE. Cycles=5. Corpus=3513. V_prev=0.70.", "Describe the full V31 inference pipeline including trimodal vectors, MoE routing, and corpus growth strategy."),
]

for label, geo, lang in tests:
    e.reset()
    r = e.infer(binary_text="probe", geometry_text=geo, language_text=lang, entry_index=1, total_entries=1)
    print(f"  [{label:<10}] M1={r['M1']:.4f} M2={r['M2']:.4f} M3={r['M3']:.4f} V={r['V']:.4f} gate={r['gate_status']}")

# 2. Multi-turn: does M2 evolve across turns?
print("\n=== MULTI-TURN V ACCUMULATION ===")
e.reset()
questions = [
    "What is reinforcement learning?",
    "How do transformers use attention mechanisms?",
    "What is a mixture of experts architecture?",
    "How does knowledge representation work in neural networks?",
    "What is the Agent Trust Protocol?",
    "How do provenance hashes secure corpus records?",
    "Describe the trimodal scalar vector system.",
    "How does the gate phase open in V31?",
    "What drives MSE loss convergence?",
    "How should inference feedback grow the corpus?",
]
for i, q in enumerate(questions, 1):
    geo = f"Entry {i}/{len(questions)}. SystemID=CASEBELIZE. TelemetryStable=True. Complexity=HIGH. PhaseReady=True. Convergence={0.5+i*0.04:.2f}."
    r = e.infer(binary_text=f"turn={i}", geometry_text=geo, language_text=q, entry_index=i, total_entries=len(questions))
    print(f"  Turn {i:02d}  V={r['V']:.4f}  M1={r['M1']:.4f}  M2={r['M2']:.4f}  M3={r['M3']:.4f}  gate={r['gate_status']:<10}  tok={r['language_token_output'][:30]}")

# 3. Show gate threshold from source
print("\n=== GATE CONFIG ===")
import inspect
src = inspect.getsource(type(e))
for line in src.splitlines():
    if any(k in line for k in ['OPEN', 'threshold', 'gate', 'convergence', 'GATE', 'phase']):
        print(" ", line.strip())
