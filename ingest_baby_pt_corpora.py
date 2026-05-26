# -*- coding: utf-8 -*-
"""ingest_baby_pt_corpora.py — Robust Ingesting of Ancient Baby Model Corpora into v31 Schema

This script reads the 13 training files from D:\\NextAura\\v31all_1\\Pt's\\Corpus\\*,
translates their fields into the modern stateful 7-input -> 2-output v31 schema,
utilizing the robust JSON parser in Stage1Ingest and the stateful RecordBuilder, 
and outputs a consolidated JSONL dataset.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage1.core.record_builder import RecordBuilder
from gui.stage1.ingest import Stage1Ingest

def main():
    print("=" * 70)
    print("[START] ROBUST ANCIENT BABY MODEL CORPORA INGESTION")
    print("=" * 70)
    
    pt_corpus_dir = Path("D:/NextAura/v31all_1/Pt's/Corpus")
    if not pt_corpus_dir.exists():
        pt_corpus_dir = Path("/mnt/d/NextAura/v31all_1/Pt's/Corpus")
        
    out_dir = Path(__file__).resolve().parent / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'corpus_baby_ingested.jsonl'
    
    all_records = []
    
    for i in range(1, 14):
        file_path = pt_corpus_dir / f"{i}.jsonl"
        if not file_path.exists():
            print(f"[WARNING] File {file_path} not found, skipping.")
            continue
            
        print(f"Reading and mapping {file_path.name}...")
        
        # Read the entire file content
        content = file_path.read_text(encoding="utf-8", errors="replace")
        
        # Parse using Stage1Ingest robust parser which repairs unescaped characters and structures
        data = Stage1Ingest.parse_json(content)
        print(f"  - Parsed {len(data)} records from {file_path.name}.")
        
        # Initialize record builder with actual parsed count to evolve state
        builder = RecordBuilder(
            total_entries=max(len(data), 1),
            mission=f"baby-model-ingest-{i}",
            codebase_hash="v31allC",
            system_id=f"BABY_MODEL_{i}"
        )
        
        for turn_idx, rec_data in enumerate(data, 1):
            try:
                binary_text = rec_data.get("Binary", "")
                geometry_text = rec_data.get("Geometry", "")
                language_text = rec_data.get("LanguageContext", "")
                baby_lto = rec_data.get("LanguageTokenOutput", "")
                
                # Compute M-scalars, layer2, and synthesize frame using modern pipeline
                rec = builder.build(
                    binary_text=binary_text,
                    geometry_text=geometry_text,
                    language_text=language_text
                )
                
                # Merge baby model's precise language output target
                if baby_lto:
                    rec["output_tokens"]["language_token_output"] = baby_lto
                    
                all_records.append(rec)
            except Exception as e:
                print(f"  - Error building record {turn_idx} in {file_path.name}: {e}")
                    
    if all_records:
        out_file.write_text('\n'.join(json.dumps(r) for r in all_records), encoding='utf-8')
        print(f"[OK] Successfully wrote {len(all_records)} records -> {out_file}")
    else:
        print("[ERROR] No records were parsed/ingested.")
        
    print("=" * 70)
    print("[SUCCESS] BABY CORPORA INGESTION COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
