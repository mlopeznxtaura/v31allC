import json
from pathlib import Path

path = Path("output/master_corpus.jsonl")
nan_count = 0
total_count = 0

with open(path, encoding="utf-8") as f:
    for idx, line in enumerate(f):
        total_count += 1
        if "nan" in line.lower():
            nan_count += 1
            print(f"Row {idx} contains 'nan' in raw line!")
            try:
                rec = json.loads(line)
                print(f"  V={rec.get('_meta', {}).get('V')}, M1_raw={rec.get('_meta', {}).get('M1_raw')}")
            except Exception:
                pass

print(f"Total lines: {total_count}, lines with nan: {nan_count}")
