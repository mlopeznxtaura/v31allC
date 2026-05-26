# -*- coding: utf-8 -*-
import json, re, sys
from pathlib import Path
sys.path.insert(0, '/mnt/d/NextAura/v31all_1/v31allC')
from stage1.core.record_builder import RecordBuilder

SRC = Path('/mnt/d/NextAura/L2EVAL/OpenSourceWorldModel-headless/v2L2/origin/4563-fixed.json')
OUT = Path('/mnt/d/NextAura/v31all_1/v31allC/output/corpus_4563.jsonl')

raw = SRC.read_text(encoding='utf-8', errors='replace')

# Repair missing commas between top-level objects
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    fixed = re.sub(r'\}\s*\n(\s*)\{', r'},\n\1{', raw)
    try:
        data = json.loads(fixed)
        print(f'Repaired: {len(data)} records')
    except json.JSONDecodeError as e:
        print(f'Still broken at {e.lineno}:{e.colno} — {e.msg}')
        sys.exit(1)
else:
    print(f'Clean parse: {len(data)} records')

# Normalize each record to binary/geometry/language
def extract(rec):
    lang = rec.get('description') or rec.get('summary') or rec.get('recommendation') or ''
    geo  = rec.get('analysis') or rec.get('phase') or rec.get('action') or ''
    bin_ = str(rec.get('step') or rec.get('entry_id') or rec.get('key_insight') or '')
    return bin_[:300], geo[:400], lang[:600]

builder = RecordBuilder(total_entries=len(data), mission='4563-corpus-ingest', codebase_hash='v31allC')
lines = []
skipped = 0
for i, rec in enumerate(data):
    b, g, l = extract(rec)
    if not l and not g:
        skipped += 1
        continue
    record = builder.build(binary_text=b, geometry_text=g, language_text=l)
    lines.append(json.dumps(record))

OUT.write_text('\n'.join(lines), encoding='utf-8')
print(f'Wrote {len(lines)} records ({skipped} skipped) -> {OUT}')
print(f'Sample V values: {[json.loads(l)[" _meta\][\V\]
