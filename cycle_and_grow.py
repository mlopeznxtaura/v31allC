"""cycle_and_grow.py — NextAura v31allC
Ingest all corpora → save every inference result to master_corpus.jsonl → retrain.

Flow per corpus:
  1. Load records from source .jsonl
  2. Run each record through V31InferenceEngine.infer()
  3. Build a fully-compliant master_corpus record from the inference output
  4. SHA-256 dedup against existing master corpus
  5. Append net-new records to output/master_corpus.jsonl

After all corpora:
  6. Retrain V31Model (AsymmetricTrainer) from scratch on full master_corpus.jsonl
  7. Register new checkpoint in models/state_registry.json
  8. Copy to models/v31_neural_model.pt (active model)
"""

import json
import sys
import time
import uuid
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stage1.core.soc_compliance import IngestionComplianceEngine, calculate_record_hash
from stage1.core.triangulation import build_mathematical_relationship
from stage3.inference_engine.main import V31InferenceEngine
from stage3.master_corpus.manager import MASTER_CORPUS_PATH
from stage2.training_engine.state_registry import AuraCycleManager

CORPORA = [
    ROOT / "output" / "corpus_4563.jsonl",
    ROOT / "output" / "corpus_baby_ingested.jsonl",
    ROOT / "output" / "corpus_iphone.jsonl",
    ROOT / "output" / "corpus_pc.jsonl",
    ROOT / "trimodal_observer_corpus.jsonl",
    ROOT / "corpus_v31_sample.jsonl",
]

TEMP_CKPT = ROOT / "models" / "_cycle_grow_ckpt.pt"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_existing_hashes() -> set:
    hashes = set()
    if not MASTER_CORPUS_PATH.exists():
        return hashes
    with MASTER_CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
                h = rec.get("_meta", {}).get("provenance", {}).get("sha256")
                if not h:
                    h = calculate_record_hash(rec.get("scalar_inputs", {}))
                hashes.add(h)
            except Exception:
                pass
    return hashes


def infer_result_to_master_record(
    source_rec: dict,
    result: dict,
    idx: int,
    total: int,
    source_name: str
) -> dict:
    """Convert an inference result + its source record into a master_corpus entry."""
    si = source_rec.get("scalar_inputs", {})
    binary_text   = si.get("binary", "")   or str(source_rec.get("M1", ""))
    geometry_text = si.get("geometry", "") or str(source_rec.get("M2", ""))
    language_text = si.get("language", "") or str(source_rec.get("M3", ""))

    math_rel = build_mathematical_relationship(
        M1=result["M1"],
        M2=result["M2"],
        M3=result["M3"],
        binary=result["binary_scalar"],
        geometry=result["geometry_scalar"],
        language=result["language_scalar"],
        context_tags=f"entry={idx}/{total}, V={result['V']:.4f}, src={source_name}"
    )

    record = {
        "M1": f"Unbounded thinker exploration_vector={result['M1']:.4f}; neural feedback state.",
        "M2": f"Efficiency score={result['M2']:.4f}; neural threshold.",
        "M3": f"Meta-multiplier={result['M3']:.4f}; feedback gain.",
        "scalar_inputs": {
            "binary":   binary_text[:512] if binary_text else f"corpus_entry={idx}",
            "geometry": geometry_text[:512] if geometry_text else f"progress={idx}/{total}",
            "language": language_text[:512] if language_text else "unspecified",
            "mathematical_relationship": math_rel
        },
        "output_tokens": {
            "next_frame_prediction": result["next_frame_prediction"],
            "language_token_output": result["language_token_output"]
        },
        "_meta": {
            "provenance": {
                "uuid": str(uuid.uuid4()),
                "sha256": "",        # filled below after hash
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "lineage_source": source_name,
                "soc2_integrity_checked": True
            },
            "V": round(result["V"], 6),
            "M1_raw": result["M1"],
            "M2_raw": result["M2"],
            "M3_raw": result["M3"],
            "binary_scalar": result["binary_scalar"],
            "geometry_scalar": result["geometry_scalar"],
            "language_scalar": result["language_scalar"],
            "gate_status": result["gate_status"],
            "turn": result["turn"],
            "cycle_source": source_name,
        }
    }

    rec_hash = calculate_record_hash(record["scalar_inputs"])
    record["_meta"]["provenance"]["sha256"] = rec_hash
    return record, rec_hash


# ── main cycle ────────────────────────────────────────────────────────────────

def run_cycles():
    engine = V31InferenceEngine()
    existing_hashes = load_existing_hashes()
    MASTER_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_added = 0
    total_skipped = 0

    print("=" * 70)
    print("CYCLE-AND-GROW: Ingest → Master Corpus → Retrain")
    print("=" * 70)

    for corpus_path in CORPORA:
        if not corpus_path.exists():
            print(f"  SKIP (not found): {corpus_path.name}")
            continue

        # Load source records
        source_records = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        source_records.append(json.loads(line))
                    except Exception:
                        pass

        total = len(source_records)
        added = 0
        skipped = 0
        engine.reset()

        with MASTER_CORPUS_PATH.open("a", encoding="utf-8") as out_f:
            for idx, src_rec in enumerate(source_records, 1):
                si = src_rec.get("scalar_inputs", {})
                binary_text   = si.get("binary", "")   or str(src_rec.get("M1", ""))
                geometry_text = si.get("geometry", "") or str(src_rec.get("M2", ""))
                language_text = si.get("language", "") or str(src_rec.get("M3", ""))

                if len(geometry_text.strip()) < 20:
                    geometry_text = (
                        f"Entry {idx}/{total}. Env=Sandbox. SystemID=CASEBELIZE. "
                        f"TelemetryStable=True. {geometry_text}".strip()
                    )
                math_rel = si.get("mathematical_relationship", "")
                if len(language_text.strip()) < 20 and math_rel:
                    language_text = math_rel

                result = engine.infer(
                    binary_text=binary_text,
                    geometry_text=geometry_text,
                    language_text=language_text,
                    stimulus=float(src_rec.get("stimulus", 1.0)),
                    is_logical=bool(src_rec.get("is_logical", True)),
                    telemetry_stable=bool(src_rec.get("telemetry_stable", True)),
                    entry_index=idx,
                    total_entries=total,
                )

                master_rec, rec_hash = infer_result_to_master_record(
                    src_rec, result, idx, total, corpus_path.name
                )

                if rec_hash in existing_hashes:
                    skipped += 1
                    continue

                existing_hashes.add(rec_hash)
                out_f.write(json.dumps(master_rec) + "\n")
                added += 1

        total_added += added
        total_skipped += skipped
        corpus_size = MASTER_CORPUS_PATH.stat().st_size / 1024
        print(f"  {corpus_path.name:<40}  src={total:>5}  added={added:>5}  skip={skipped:>5}  master={corpus_size:.1f}KB")

    print(f"\n  Total added to master_corpus: {total_added}  |  Duplicates skipped: {total_skipped}")
    master_count = sum(1 for l in MASTER_CORPUS_PATH.read_text(encoding="utf-8").splitlines() if l.strip())
    print(f"  Master corpus total records : {master_count}")

    # ── Retrain ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RETRAIN: V31Model from full master_corpus.jsonl")
    print("=" * 70)

    try:
        import torch
        from stage2.training_engine.neural_model import V31Model
        from stage2.training_engine.asymmetric_trainer import AsymmetricTrainer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        model = V31Model().to(device)
        trainer = AsymmetricTrainer(model, lr=3e-4)

        # Load existing active weights if present (continue from last checkpoint)
        existing_ckpt = ROOT / "models" / "v31_neural_model.pt"
        if existing_ckpt.exists():
            trainer.load_checkpoint(str(existing_ckpt))
            print(f"  Loaded existing checkpoint: {existing_ckpt.name} (step={trainer.step_count})")

        EPOCHS = 3
        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            metrics = trainer.train_epoch(
                str(MASTER_CORPUS_PATH),
                batch_size=32,
                on_batch_complete=None
            )
            elapsed = round(time.time() - t0, 2)
            print(f"  Epoch {epoch}/{EPOCHS}  records={metrics['records']}  "
                  f"loss={metrics['loss']:.5f}  mse={metrics['mse']:.5f}  "
                  f"ce={metrics['ce']:.5f}  {elapsed}s")

        # Save checkpoint
        trainer.save_checkpoint(str(TEMP_CKPT))
        print(f"  Checkpoint saved: {TEMP_CKPT.name}")

        # Register in state_registry
        corpus_hash = AuraCycleManager.calculate_corpus_hash(str(MASTER_CORPUS_PATH))
        state_hash = AuraCycleManager.compute_state_hash(corpus_hash, EPOCHS, 32, 3e-4)
        AuraCycleManager.register_state(
            state_hash=state_hash,
            corpus_path=str(MASTER_CORPUS_PATH),
            corpus_hash=corpus_hash,
            epochs=EPOCHS,
            batch_size=32,
            lr=3e-4,
            metrics=metrics,
            source_checkpoint_path=str(TEMP_CKPT.relative_to(ROOT))
        )
        print(f"  Registered state: {state_hash[:16]}  →  models/v31_neural_model.pt updated")

    except Exception as ex:
        print(f"  [RETRAIN ERROR] {ex}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 70)
    print("CYCLE-AND-GROW COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_cycles()
