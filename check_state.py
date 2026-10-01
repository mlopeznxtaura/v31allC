import json, sys
from pathlib import Path

qa = Path("output/corpus_live_qa.jsonl")
master = Path("output/master_corpus.jsonl")

print("=== LIVE Q&A RESULTS ===")
for i, line in enumerate(qa.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip(): continue
    r = json.loads(line)
    m = r.get("_meta", {})
    si = r.get("scalar_inputs", {})
    print(f"Q{i:02d} V={m.get('V',0):.4f}  gate={m.get('gate_status','?'):<6}  M1={m.get('M1_raw',0):.3f}  M2={m.get('M2_raw',0):.3f}  M3={m.get('M3_raw',0):.3f}")
    print(f"     Q: {si.get('language','')[:75]}")
    print(f"     T: {r.get('output_tokens',{}).get('language_token_output','')[:60]}")
    print()

print("=== CORPUS STATE ===")
lines = [l for l in master.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"Master records : {len(lines)}")
print(f"Master size    : {master.stat().st_size/1024/1024:.1f} MB")

# V distribution
vs = []
for l in lines:
    try: vs.append(json.loads(l).get("_meta",{}).get("V",0))
    except: pass
if vs:
    print(f"V mean={sum(vs)/len(vs):.4f}  min={min(vs):.4f}  max={max(vs):.4f}")

# Model checkpoint
from pathlib import Path as P
ckpt = P("models/v31_neural_model.pt")
print(f"Model checkpoint: {ckpt.stat().st_size/1024/1024:.1f} MB  modified={__import__('datetime').datetime.fromtimestamp(ckpt.stat().st_mtime)}")

# State registry
sr = P("models/state_registry.json")
if sr.exists():
    states = json.loads(sr.read_text(encoding="utf-8"))
    print(f"State registry entries: {len(states)}")
    last = list(states.values())[-1] if isinstance(states, dict) else states[-1]
    print(f"Last registered: loss={last.get('metrics',{}).get('loss','?')}  cycle={last.get('metrics',{}).get('cycle','?')}")
