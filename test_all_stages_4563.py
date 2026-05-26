# -*- coding: utf-8 -*-
"""test_all_stages_4563.py — V31all Consolidated Run

This script runs the entire observer model pipeline in a single unified run:
  Stage 1 (Ingestion) -> Stage 2 (Training Loss Calculation) -> Stage 3 (Inference Testing)

Workflow:
  1. Ingests raw data from 4563-fixed.json into corpus_4563.jsonl (using GPU-accelerated RecordBuilder).
  2. Runs stateful multi-turn inference over the entire corpus.
  3. Computes 4-component triangulation loss at each turn.
  4. Outputs a consolidated telemetry report including loss statistics, gate status ratio,
     and a side-by-side text trajectory plot of Value vs Loss.
"""

import os
import sys
import json
import time
from pathlib import Path

# Insert project root into python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage1.core.record_builder import RecordBuilder
from stage1.core.triangulation import ACTION_VOCAB
from stage2.training_engine.triangulation_loss import total_triangulation_loss
from stage3.inference_engine.main import V31InferenceEngine
from gui.stage1.ingest import Stage1Ingest

# Configuration
RAW_SRC = Path('/mnt/d/NextAura/L2EVAL/OpenSourceWorldModel-headless/v2L2/origin/4563-fixed.json')
INGESTED_CORPUS = Path(__file__).resolve().parent / 'output' / 'corpus_4563.jsonl'
OUTPUT_RESULTS = Path(__file__).resolve().parent / 'output' / 'replay_results_losses_4563.jsonl'

INGESTED_CORPUS.parent.mkdir(parents=True, exist_ok=True)


def run_ingestion():
    print("=" * 70)
    print("🚀 STAGE 1: RUNNING CORPUS INGESTION")
    print("=" * 70)
    
    if not RAW_SRC.exists():
        print(f"⚠ Raw source dataset {RAW_SRC} not found.")
        if INGESTED_CORPUS.exists():
            print(f"✅ Found existing ingested corpus at {INGESTED_CORPUS}. Skipping raw ingestion.")
            return True
        else:
            print("❌ No raw source or existing corpus found. Cannot proceed!")
            return False
            
    print(f"Ingesting from: {RAW_SRC}")
    try:
        raw = RAW_SRC.read_text(encoding='utf-8', errors='replace')
        data = Stage1Ingest.parse_json(raw)
        if not data:
            print("❌ Failed to parse any records from raw JSON.")
            return False
        
        print(f"Successfully parsed {len(data)} raw records via Stage1Ingest.")
        
        # Normalize and extract fields
        def extract_fields(rec):
            lang = rec.get('description') or rec.get('summary') or rec.get('recommendation') or ''
            geo  = rec.get('analysis') or rec.get('phase') or rec.get('action') or ''
            bin_ = str(rec.get('step') or rec.get('entry_id') or rec.get('key_insight') or '')
            return bin_[:300], geo[:400], lang[:600]

        builder = RecordBuilder(total_entries=len(data), mission='4563-corpus-ingest', codebase_hash='v31allC')
        lines = []
        skipped = 0
        
        for i, rec in enumerate(data):
            b, g, l = extract_fields(rec)
            if not l and not g:
                skipped += 1
                continue
            record = builder.build(binary_text=b, geometry_text=g, language_text=l)
            lines.append(json.dumps(record))
            
        INGESTED_CORPUS.write_text('\n'.join(lines), encoding='utf-8')
        print(f"✅ Wrote {len(lines)} records ({skipped} skipped) -> {INGESTED_CORPUS}")
        return True
    except Exception as ex:
        print(f"❌ Ingestion failed with error: {ex}")
        return False


def run_training_and_inference():
    print("\n" + "=" * 70)
    print("⚙ STAGES 2 & 3: RUNNING TRAINING LOSS & INFERENCE TESTING")
    print("=" * 70)
    
    if not INGESTED_CORPUS.exists():
        print(f"❌ Ingested corpus file {INGESTED_CORPUS} is missing!")
        return
        
    print(f"Loading corpus from: {INGESTED_CORPUS}")
    try:
        # Load ingested records
        records = []
        with open(INGESTED_CORPUS, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        
        total_records = len(records)
        print(f"Successfully loaded {total_records} records.")
        
        # Initialize stateful inference engine
        engine = V31InferenceEngine()
        results_with_losses = []
        
        # Accumulated loss metrics
        loss_sums = {
            "total": 0.0,
            "v_prediction": 0.0,
            "action_alignment": 0.0,
            "invariant": 0.0,
            "m3_budget": 0.0
        }
        
        v_history = []
        gate_statuses = []
        
        print(f"Replaying records through stateful engine & computing triangulation losses...")
        
        for idx, rec in enumerate(records):
            si = rec.get("scalar_inputs", {})
            binary_text   = si.get("binary", "")   or str(rec.get("M1", ""))
            geometry_text = si.get("geometry", "") or str(rec.get("M2", ""))
            language_text = si.get("language", "") or str(rec.get("M3", ""))
            
            # Enrich sparse data for invariant stability
            if len(geometry_text.strip()) < 20:
                geometry_text = (
                    f"Entry {idx+1}/{total_records}. Env=Sandbox. SystemID=CASEBELIZE. "
                    f"TelemetryStable=True. {geometry_text}".strip()
                )
            math_rel = si.get("mathematical_relationship", "")
            if len(language_text.strip()) < 20 and math_rel:
                language_text = math_rel
                
            # Stateful Inference Turn
            result = engine.infer(
                binary_text=binary_text,
                geometry_text=geometry_text,
                language_text=language_text,
                stimulus=float(rec.get("stimulus", 1.0)),
                is_logical=bool(rec.get("is_logical", True)),
                telemetry_stable=bool(rec.get("telemetry_stable", True)),
                entry_index=idx + 1,
                total_entries=total_records,
            )
            
            # Map parameters for training loss calculation
            predicted_v = result["V"]
            target_reward = rec["_meta"]["V"]  # The expert value function computed during ingestion
            predicted_action = result["language_token_output"]
            expert_action = rec["output_tokens"]["language_token_output"]
            m3_multiplier = result["M3"]
            elapsed_fraction = idx / max(total_records - 1, 1)
            
            # Calculate stage 2 Triangulation Loss
            losses = total_triangulation_loss(
                predicted_v=predicted_v,
                target_reward=target_reward,
                predicted_action=predicted_action,
                expert_action=expert_action,
                action_vocab=ACTION_VOCAB,
                v_history=v_history,
                m3_multiplier=m3_multiplier,
                elapsed_fraction=elapsed_fraction
            )
            
            # Record historical V
            v_history.append(predicted_v)
            gate_statuses.append(result["gate_status"])
            
            # Accumulate loss metrics
            for k in loss_sums:
                loss_sums[k] += losses[k]
                
            # Enrich inference result with computed losses
            result["losses"] = losses
            result["target_V"] = target_reward
            result["expert_action"] = expert_action
            results_with_losses.append(result)
            
        # Write output snapshots to disk
        OUTPUT_RESULTS.write_text(
            '\n'.join(json.dumps(r) for r in results_with_losses),
            encoding='utf-8'
        )
        print(f"✅ Replay logs and computed losses written to {OUTPUT_RESULTS}")
        
        # Calculate summary statistics
        avg_losses = {k: v / total_records for k, v in loss_sums.items()}
        open_count = gate_statuses.count("OPEN")
        open_ratio = open_count / total_records
        
        print("\n" + "=" * 70)
        print("📊 CONSOLIDATED Telemetry & PERFORMANCE REPORT")
        print("=" * 70)
        print(f"Total Iterations Replayed: {total_records}")
        print(f"Phase Gate Ratio (OPEN)  : {open_ratio:.2%} ({open_count}/{total_records})")
        print("-" * 70)
        print("AVERAGE TRIANGULATION LOSSES (lower is better):")
        print(f"  • Total Loss             : {avg_losses['total']:.5f}")
        print(f"  • V-Prediction MSE Loss  : {avg_losses['v_prediction']:.5f}")
        print(f"  • Action Alignment Loss  : {avg_losses['action_alignment']:.5f}")
        print(f"  • Invariant Consistency  : {avg_losses['invariant']:.5f}")
        print(f"  • M3 Time Budget Penalty : {avg_losses['m3_budget']:.5f}")
        print("-" * 70)
        
        print("\n" + "📈 STATEFUL TRAJECTORY PLOT (Value vs Invariant Loss)")
        print("-" * 70)
        print("  [idx]  Target_V -> Pred_V  [Trajectory Visualization]           Loss")
        print("-" * 70)
        
        # Print a neat text-based bar trajectory of value function & losses
        for i, r in enumerate(results_with_losses):
            pred_v = r["V"]
            target_v = r["target_V"]
            loss_val = r["losses"]["total"]
            
            # Build bar representation of V(s)
            bar_len = int(pred_v * 30)
            bar = "#" * bar_len + " " * (30 - bar_len)
            gate_mk = " [OPEN]" if r["gate_status"] == "OPEN" else "       "
            
            print(f"  [{i+1:>3}]  {target_v:.2f} -> {pred_v:.2f}  |{bar}|{gate_mk}  Loss: {loss_val:.4f}")
            
        print("=" * 70)
        print("CONSOLIDATED SYSTEM PIPELINE RUN COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as ex:
        print(f"❌ Training and inference test failed with error: {ex}")


if __name__ == '__main__':
    success = run_ingestion()
    if success:
        run_training_and_inference()
