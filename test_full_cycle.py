from stage3.inference_engine.main import V31InferenceEngine
import json, sys

CORPUS = "/mnt/d/NextAura/v31all_1/v31allC/output/test_corpus_50.jsonl"

engine = V31InferenceEngine()

try:
    results = engine.run_corpus(CORPUS)
except Exception as e:
    print(f"FATAL run_corpus: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

v_vals  = [r["V"] for r in results]
actions = [r["language_token_output"] for r in results]
gates   = [r["gate_status"] for r in results]
open_ct = gates.count("OPEN")

act_count = {}
for a in actions:
    act_count[a] = act_count.get(a, 0) + 1

print(f"records processed : {len(results)}")
print(f"V_mean={sum(v_vals)/len(v_vals):.4f}  V_min={min(v_vals):.4f}  V_max={max(v_vals):.4f}")
print(f"gate OPEN         : {open_ct}/{len(results)}")
print(f"action dist       : {act_count}")
print()
print("--- first 5 turns ---")
for r in results[:5]:
    print(f"  turn={r['turn']:>3}  V={r['V']:.4f}  gate={r['gate_status']:<10}  lto={r['language_token_output']:<12}  M2={r['M2']}  M3={r['M3']}")
print()
print("--- last 5 turns ---")
for r in results[-5:]:
    print(f"  turn={r['turn']:>3}  V={r['V']:.4f}  gate={r['gate_status']:<10}  lto={r['language_token_output']:<12}  M2={r['M2']}  M3={r['M3']}")
print()
print("--- V trajectory (all 50) ---")
for i, r in enumerate(results):
    bar = "#" * int(r["V"] * 30)
    print(f"  [{i+1:>2}] {r['V']:.4f} {bar}")
