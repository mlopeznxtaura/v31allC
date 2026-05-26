"""run_all_corpora.py - Ingest and run inference cycles on all available corpora."""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage3.inference_engine.main import V31InferenceEngine

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"

CORPORA = [
    OUTPUT / "corpus_baby_ingested.jsonl",
    OUTPUT / "corpus_iphone.jsonl",
    OUTPUT / "corpus_pc.jsonl",
    OUTPUT / "corpus_4563.jsonl",
    ROOT  / "trimodal_observer_corpus.jsonl",
    ROOT  / "corpus_v31_sample.jsonl",
]

def run_corpus(path: Path, engine: V31InferenceEngine):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    total = len(records)
    v_vals = []
    open_ct = 0
    t0 = time.time()

    engine.reset()
    for idx, rec in enumerate(records):
        si = rec.get("scalar_inputs", {})
        binary_text   = si.get("binary", "")   or str(rec.get("M1", ""))
        geometry_text = si.get("geometry", "") or str(rec.get("M2", ""))
        language_text = si.get("language", "") or str(rec.get("M3", ""))
        if len(geometry_text.strip()) < 20:
            geometry_text = f"Entry {idx+1}/{total}. Env=Sandbox. SystemID=CASEBELIZE. TelemetryStable=True. {geometry_text}".strip()
        math_rel = si.get("mathematical_relationship", "")
        if len(language_text.strip()) < 20 and math_rel:
            language_text = math_rel

        result = engine.infer(
            binary_text=binary_text,
            geometry_text=geometry_text,
            language_text=language_text,
            stimulus=float(rec.get("stimulus", 1.0)),
            is_logical=bool(rec.get("is_logical", True)),
            telemetry_stable=bool(rec.get("telemetry_stable", True)),
            entry_index=idx + 1,
            total_entries=total,
        )
        v_vals.append(result["V"])
        if result["gate_status"] == "OPEN":
            open_ct += 1

    elapsed = round(time.time() - t0, 2)
    v_mean  = round(sum(v_vals) / len(v_vals), 4) if v_vals else 0.0
    v_min   = round(min(v_vals), 4) if v_vals else 0.0
    v_max   = round(max(v_vals), 4) if v_vals else 0.0
    return total, v_mean, v_min, v_max, open_ct, elapsed


if __name__ == "__main__":
    engine = V31InferenceEngine()

    print("=" * 70)
    print("MULTI-CORPUS INFERENCE CYCLE RUN")
    print("=" * 70)
    print(f"{'Corpus':<40} {'Recs':>5}  {'V_mean':>7}  {'V_min':>7}  {'V_max':>7}  {'OPEN':>6}  {'Sec':>5}")
    print("-" * 70)

    for corpus_path in CORPORA:
        if not corpus_path.exists():
            print(f"  SKIP (not found): {corpus_path.name}")
            continue
        try:
            total, v_mean, v_min, v_max, open_ct, elapsed = run_corpus(corpus_path, engine)
            print(f"  {corpus_path.name:<38} {total:>5}  {v_mean:>7.4f}  {v_min:>7.4f}  {v_max:>7.4f}  {open_ct:>4}/{total}  {elapsed:>5}s")
        except Exception as ex:
            print(f"  ERROR on {corpus_path.name}: {ex}")

    print("=" * 70)
    print("ALL CYCLES COMPLETE")
    print("=" * 70)
