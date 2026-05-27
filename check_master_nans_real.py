import json
import math
from pathlib import Path

path = Path("output/master_corpus.jsonl")
nan_count = 0
total_count = 0

with open(path, encoding="utf-8") as f:
    for idx, line in enumerate(f):
        total_count += 1
        try:
            rec = json.loads(line)
            meta = rec.get("_meta", {})
            has_nan = False
            for k, v in meta.items():
                if isinstance(v, float) and math.isnan(v):
                    has_nan = True
                    print(f"Row {idx}: float field {k} is NaN!")
                elif isinstance(v, dict):
                    # Check provenance
                    pass
            
            # Check if M1_raw, M2_raw, M3_raw are NaN
            m1 = meta.get("M1_raw")
            m2 = meta.get("M2_raw")
            m3 = meta.get("M3_raw")
            v = meta.get("V")
            
            if m1 is not None and (math.isnan(m1) if isinstance(m1, float) else False):
                has_nan = True
            if m2 is not None and (math.isnan(m2) if isinstance(m2, float) else False):
                has_nan = True
            if m3 is not None and (math.isnan(m3) if isinstance(m3, float) else False):
                has_nan = True
            if v is not None and (math.isnan(v) if isinstance(v, float) else False):
                has_nan = True
                
            if has_nan:
                nan_count += 1
        except Exception as e:
            print(f"Error parsing row {idx}: {e}")

print(f"Total lines: {total_count}, lines with actual NaN values: {nan_count}")
