# -*- coding: utf-8 -*-
"""ingest_deepseek_phone_pc.py — GPU-Accelerated DeepSeek PC & iPhone Corpora Ingestion

This script parses the raw nested multi-turn JSON conversations of both the PC and iPhone deepseek-reasoner
corpora, extracts their conversational turns (User Prompt vs Assistant Think + Response), and builds 
fully populated JSONL stateful records using the GPU-accelerated RecordBuilder.
"""

import os
import sys
import json
import time
from pathlib import Path

# Insert project root into python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage1.core.record_builder import RecordBuilder

# Input/Output paths
PATH_PC = Path(r"D:\NextAura\L2EVAL\OpenSourceWorldModel-headless\v2L2\v1corpus4domaindeepseekphone&pc\conversations_pc.json")
PATH_IPHONE = Path(r"D:\NextAura\L2EVAL\OpenSourceWorldModel-headless\v2L2\v1corpus4domaindeepseekphone&pc\conversations_iphone.json")

OUT_DIR = Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PC = OUT_DIR / 'corpus_pc.jsonl'
OUT_IPHONE = OUT_DIR / 'corpus_iphone.jsonl'


def parse_conversations_to_rows(path: Path, system_id: str):
    """
    Load nested conversations and extract flat list of turn dictionaries
    ready to be processed in batches by RecordBuilder.
    """
    print(f"Loading and parsing {path.name}...")
    with open(path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
        
    print(f"Loaded {len(conversations)} raw conversations from {path.name}.")
    
    rows = []
    for c_idx, conv in enumerate(conversations):
        mapping = conv.get("mapping", {})
        title = conv.get("title", "Untitled")
        conv_id = conv.get("id", "Unknown")
        
        # Traverse parent-child conversation chain starting at root
        node_id = 'root'
        chain = []
        visited = set()
        while node_id in mapping and node_id not in visited:
            visited.add(node_id)
            node = mapping[node_id]
            msg = node.get("message")
            if msg:
                chain.append(msg)
            children = node.get("children", [])
            if not children:
                break
            # standard: follow primary child
            node_id = children[0]
            
        # Extract dialogue turns
        turns = []
        current_request = None
        for msg in chain:
            fragments = msg.get("fragments", [])
            if not fragments:
                continue
                
            req_content = ""
            think_content = ""
            resp_content = ""
            for frag in fragments:
                ftype = frag.get("type")
                fcontent = frag.get("content", "")
                if ftype == "REQUEST":
                    req_content += fcontent
                elif ftype == "THINK":
                    think_content += fcontent
                elif ftype == "RESPONSE":
                    resp_content += fcontent
                    
            if req_content:
                current_request = req_content
                
            if think_content or resp_content:
                if current_request is not None:
                    turns.append({
                        "request": current_request,
                        "think": think_content,
                        "response": resp_content
                    })
                    
        total_turns = len(turns)
        for turn_idx, t in enumerate(turns, 1):
            # Rich context formulation for M1, M2, M3 inputs
            binary_text = f"Phase gate logical: hash=v31allC, title='{title[:100]}', id={conv_id[:8]}, turn={turn_idx}"
            geometry_text = f"Turn {turn_idx}/{total_turns}. Env=Sandbox. SystemID={system_id}. Title='{title[:100]}'."
            language_text = t["request"]
            
            # Stimulus proportional to thinking length (complexity)
            stimulus = 1.0 + min(len(t["think"]) / 1000.0, 9.0)
            
            rows.append({
                "binary_text": binary_text,
                "geometry_text": geometry_text,
                "language_text": language_text,
                "stimulus": stimulus,
                "is_logical": True,
                "telemetry_stable": True,
                # Extra metadata to save in JSONL
                "meta_extra": {
                    "conversation_id": conv_id,
                    "conversation_title": title,
                    "turn_index": turn_idx,
                    "total_turns": total_turns,
                    "think_len": len(t["think"]),
                    "response_len": len(t["response"]),
                }
            })
            
    print(f"Extracted {len(rows)} turns across all conversations.")
    return rows


def run_ingestion():
    print("=" * 70)
    print("[START] DEEPSEEK CORPORA INGESTION")
    print("=" * 70)
    
    # Process PC Corpus
    if PATH_PC.exists():
        pc_rows = parse_conversations_to_rows(PATH_PC, "DEEPSEEK_PC")
        if pc_rows:
            print(f"Processing {len(pc_rows)} PC turns through RecordBuilder...")
            builder_pc = RecordBuilder(
                total_entries=len(pc_rows),
                mission="deepseek-pc-ingest",
                codebase_hash="v31allC",
                system_id="DEEPSEEK_PC"
            )
            
            # Use GPU-accelerated batch build
            t0 = time.time()
            records = builder_pc.build_batch(pc_rows)
            
            # Incorporate extra metadata into records
            for rec, row in zip(records, pc_rows):
                rec["_meta"].update(row["meta_extra"])
                
            # Write ingested JSONL
            OUT_PC.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
            print(f"[OK] Successfully wrote {len(records)} ingested records -> {OUT_PC} ({time.time()-t0:.2f}s)")
    else:
        print(f"[ERROR] PC raw corpus {PATH_PC} not found!")

    print("-" * 70)

    # Process iPhone Corpus
    if PATH_IPHONE.exists():
        iphone_rows = parse_conversations_to_rows(PATH_IPHONE, "DEEPSEEK_IPHONE")
        if iphone_rows:
            print(f"Processing {len(iphone_rows)} iPhone turns through RecordBuilder...")
            builder_iphone = RecordBuilder(
                total_entries=len(iphone_rows),
                mission="deepseek-iphone-ingest",
                codebase_hash="v31allC",
                system_id="DEEPSEEK_IPHONE"
            )
            
            # Use GPU-accelerated batch build
            t0 = time.time()
            records = builder_iphone.build_batch(iphone_rows)
            
            # Incorporate extra metadata into records
            for rec, row in zip(records, iphone_rows):
                rec["_meta"].update(row["meta_extra"])
                
            # Write ingested JSONL
            OUT_IPHONE.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
            print(f"[OK] Successfully wrote {len(records)} ingested records -> {OUT_IPHONE} ({time.time()-t0:.2f}s)")
    else:
        print(f"[ERROR] iPhone raw corpus {PATH_IPHONE} not found!")

    print("=" * 70)
    print("[SUCCESS] ALL INGESTIONS COMPLETED!")
    print("=" * 70)


if __name__ == '__main__':
    run_ingestion()
