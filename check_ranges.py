import json
import math
from pathlib import Path

path = Path("output/master_corpus.jsonl")
vs = []
m1s = []
m2s = []
m3s = []

with open(path, encoding="utf-8") as f:
    for idx, line in enumerate(f):
        try:
            rec = json.loads(line)
            meta = rec.get("_meta", {})
            vs.append(meta.get("V", 0.5))
            m1s.append(meta.get("M1_raw", 1.0))
            m2s.append(meta.get("M2_raw", 1.0))
            m3s.append(meta.get("M3_raw", 1.0))
        except Exception:
            pass

print("V range:", min(vs), "to", max(vs))
print("M1 range:", min(m1s), "to", max(m1s))
print("M2 range:", min(m2s), "to", max(m2s))
print("M3 range:", min(m3s), "to", max(m3s))
