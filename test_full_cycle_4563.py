import sys
sys.path.insert(0, "/mnt/d/NextAura/v31all_1/v31allC")
from stage3.inference_engine.main import V31InferenceEngine
import json

engine = V31InferenceEngine()
results = engine.run_corpus("/mnt/d/NextAura/v31all_1/v31allC/output/corpus_4563.jsonl")

v_vals  = [r["V"] for r in results]
actions = [r["language_token_output"] for r in results]
gates   = [r["gate_status"] for r in results]
open_ct = gates.count("OPEN")

act_count = {}
for a in actions:
    act_count[a] = act_count.get(a, 0) + 1

print(f"records : {len(results)}")
print(f"V_mean={sum(v_vals)/len(v_vals):.4f}  V_min={min(v_vals):.4f}  V_max={max(v_vals):.4f}")
print(f"gate OPEN: {open_ct}/{len(results)}")
print(f"actions  : {act_count}")
print()
print("--- last 10 turns ---")
for r in results[-10:]:
    print(f"  t={r['turn']:>3}  V={r['V']:.4f}  gate={r['gate_status']:<10}  lto={r['language_token_output']:<14}  geo={r['geometry_scalar']:.3f}  lang={r['language_scalar']:.3f}")
print()
print("--- V trajectory ---")
for i, r in enumerate(results):
    bar = "#" * int(r["V"] * 40)
    gate_marker = " <OPEN" if r["gate_status"] == "OPEN" else ""
    print(f"  [{i+1:>3}] {r['V']:.4f} {bar}{gate_marker}")