import sys
sys.path.insert(0, "/mnt/d/NextAura/v31all_1/v31allC")
from gui.stage1.ingest import Stage1Ingest
from stage1.core.record_builder import RecordBuilder
import json
from pathlib import Path

raw = open("/mnt/d/NextAura/L2EVAL/OpenSourceWorldModel-headless/v2L2/origin/4563-fixed.json",
           encoding="utf-8", errors="replace").read()

records = Stage1Ingest.parse_json(raw)
print(f"parsed: {len(records)} records")
print(f"keys[0]:  {list(records[0].keys())}")
print(f"keys[-1]: {list(records[-1].keys())}")

# Map 4563 fields -> RecordBuilder inputs
def extract(rec):
    lang = rec.get("description") or rec.get("summary") or rec.get("recommendation") or ""
    geo  = rec.get("analysis") or rec.get("phase") or rec.get("action") or ""
    bin_ = str(rec.get("step") or rec.get("entry_id") or "")
    return bin_[:300], geo[:400], lang[:600]

builder = RecordBuilder(total_entries=len(records), mission="4563-ingest", codebase_hash="v31allC")
out_lines = []
skipped = 0
for i, rec in enumerate(records):
    b, g, l = extract(rec)
    if not l and not g:
        skipped += 1
        continue
    built = builder.build(binary_text=b, geometry_text=g, language_text=l)
    out_lines.append(json.dumps(built))

out = Path("/mnt/d/NextAura/v31all_1/v31allC/output/corpus_4563.jsonl")
out.write_text("\n".join(out_lines), encoding="utf-8")
print(f"wrote: {len(out_lines)} records, skipped: {skipped} -> {out}")
vs = [json.loads(l)["_meta"]["V"] for l in out_lines[:5]]
print(f"first 5 V: {vs}")