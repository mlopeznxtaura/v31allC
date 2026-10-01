"""ingest_badas.py — BADAS dashcam dataset ingestor
Ingests screen_rec_1779777342_badas_all.mp4 + _badas.json twice:
  OF_1: primary pass — frame + BADAS reasoning as corpus records
  OF_2: orthogonal pass — same frames reparameterized as opposite vector

For each BADAS window:
  - Extract the video frame at that timestamp (64x64 RGB, base64)
  - Build scalar_inputs from risk, probability, action, reasoning
  - Run through V31InferenceEngine
  - Emit record with pixel_frame_base64, IF7 fields, orthogonal_vector
  - Emit second record with orthogonal_reparameterize applied (OF_2)
"""

import sys, json, time, uuid, base64, subprocess
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BADAS_JSON = Path("/mnt/d/Users/archi/Downloads/screen_rec_1779777342_badas.json")
BADAS_MP4  = Path("/mnt/d/Users/archi/Downloads/screen_rec_1779777342_badas_all.mp4")
FRAMES_DIR = ROOT / "output" / "_badas_frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

from stage3.master_corpus.manager import MASTER_CORPUS_PATH
from stage3.inference_engine.main import V31InferenceEngine
from stage1.core.soc_compliance import calculate_record_hash
from stage1.core.triangulation import build_mathematical_relationship, orthogonal_reparameterize

def load_existing_hashes():
    hashes = set()
    if not MASTER_CORPUS_PATH.exists(): return hashes
    with open(MASTER_CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                h = json.loads(s).get("_meta",{}).get("provenance",{}).get("sha256")
                if h: hashes.add(h)
            except: pass
    return hashes

def extract_frame_at(ts: float, idx: int) -> str:
    """Extract single frame at timestamp ts from mp4, return base64 RGB bytes."""
    out_path = FRAMES_DIR / f"badas_{idx:04d}.png"
    if not out_path.exists():
        cmd = [
            "ffmpeg", "-y", "-ss", str(ts),
            "-i", str(BADAS_MP4),
            "-vframes", "1",
            "-vf", "scale=64:64",
            str(out_path)
        ]
        subprocess.run(cmd, capture_output=True)
    if out_path.exists():
        from PIL import Image
        import numpy as np
        img = Image.open(out_path).convert("RGB").resize((64,64))
        arr = np.array(img, dtype="uint8")
        return base64.b64encode(arr.tobytes()).decode("ascii")
    return ""

def build_record(window: dict, result: dict, frame_b64: str,
                 pass_name: str, idx: int, total: int,
                 rec_hash: str) -> dict:
    meta = result.get("_meta_extra", {})
    return {
        "M1": f"exploration_vector={result['M1']:.4f}",
        "M2": f"efficiency={result['M2']:.4f}",
        "M3": f"meta_multiplier={result['M3']:.4f}",
        "scalar_inputs": {
            "binary":   f"badas pass={pass_name} ts={window['timestamp']:.3f} risk={window['risk_level']} idx={idx}",
            "geometry": f"Entry {idx}/{total}. SystemID=CASEBELIZE. BADAS=True. Risk={window['risk_level'].upper()}. PhaseReady=True. Convergence={0.4+idx/total:.2f}.",
            "language": meta.get("language_text", ""),
            "mathematical_relationship": build_mathematical_relationship(
                result["M1"], result["M2"], result["M3"],
                result["binary_scalar"], result["geometry_scalar"], result["language_scalar"],
                f"badas_{pass_name},ts={window['timestamp']:.3f},risk={window['risk_level']},V={result['V']:.4f},if7={result['if7_policy_fusion']:.4f}"
            )
        },
        "output_tokens": {
            "next_frame_prediction": result["next_frame_prediction"],
            "language_token_output":  result["language_token_output"],
        },
        "_meta": {
            "provenance": {
                "uuid":                 str(uuid.uuid4()),
                "sha256":               rec_hash,
                "timestamp":            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "lineage_source":       f"badas:{pass_name}",
                "soc2_integrity_checked": True,
            },
            "V":                    round(result["V"], 6),
            "M1_raw":               result["M1"],
            "M2_raw":               result["M2"],
            "M3_raw":               result["M3"],
            "binary_scalar":        result["binary_scalar"],
            "geometry_scalar":      result["geometry_scalar"],
            "language_scalar":      result["language_scalar"],
            "if7_policy_fusion":    result["if7_policy_fusion"],
            "if7_coherence":        result["if7_coherence"],
            "if7_stability_loss":   result["if7_stability_loss"],
            "if7_uncertainty":      result["if7_uncertainty"],
            "orthogonal_vector":    result["orthogonal_vector"],
            "gate_status":          result["gate_status"],
            "cycle_source":         f"badas:{pass_name}",
            # BADAS-specific
            "badas_timestamp":      window["timestamp"],
            "badas_probability":    window.get("probability", 0.0),
            "badas_risk_level":     window.get("risk_level", ""),
            "badas_action":         window.get("action", ""),
            "badas_reasoning":      window.get("reasoning", ""),
            "badas_collision_risk": window.get("collision_risk_percent", 0.0),
            "badas_pass":           pass_name,
            "pixel_frame_base64":   frame_b64,
        }
    }

def main():
    print("=" * 64)
    print("BADAS INGEST — 2x passes (OF_1 primary + OF_2 orthogonal)")
    print("=" * 64)

    # Load BADAS JSON
    with open(BADAS_JSON, encoding="utf-8") as f:
        badas = json.load(f)
    windows = badas.get("windows", [])
    print(f"  BADAS model : {badas.get('model','?')}")
    print(f"  Exported at : {badas.get('exported_at','?')}")
    print(f"  Windows     : {len(windows)}")

    # Risk distribution
    risk_counts = {}
    for w in windows:
        risk_counts[w.get("risk_level","?")] = risk_counts.get(w.get("risk_level","?"),0) + 1
    print(f"  Risk dist   : {risk_counts}")

    engine = V31InferenceEngine()
    existing = load_existing_hashes()
    print(f"  Existing hashes: {len(existing)}")

    MASTER_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_added = 0

    with open(MASTER_CORPUS_PATH, "a", encoding="utf-8") as out_f:

        for pass_name in ["OF_1", "OF_2"]:
            print(f"\n  --- {pass_name} ---")
            engine.reset()
            added = skipped = 0
            total = len(windows)

            for idx, window in enumerate(windows, 1):
                ts    = window.get("timestamp", 0.0)
                risk  = window.get("risk_level", "low")
                prob  = window.get("probability", 0.0)
                action    = window.get("action", "")
                reasoning = window.get("reasoning", "")
                collision = window.get("collision_risk_percent", 0.0)

                # Build rich language text from BADAS fields
                if reasoning:
                    lang_text = (f"BADAS-Reasoning. Action: {action}. "
                                 f"Reasoning: {reasoning} "
                                 f"CollisionRisk: {collision:.1f}%. Risk: {risk}. "
                                 f"Probability: {prob:.4f}. t={ts:.3f}s.")
                else:
                    lang_text = (f"BADAS window. Risk={risk}. Probability={prob:.4f}. "
                                 f"t={ts:.3f}s. No trigger reasoning.")

                geo_text = (f"Entry {idx}/{total}. BADAS dashcam. SystemID=CASEBELIZE. "
                            f"Risk={risk.upper()}. PhaseReady=True. "
                            f"Convergence={0.4 + idx/total * 0.4:.2f}. "
                            f"CollisionProb={prob:.4f}.")

                bin_text = (f"badas pass={pass_name} ts={ts:.3f} "
                            f"risk={risk} prob={prob:.4f} idx={idx}/{total}")

                # For OF_2: invert risk signal in geometry (orthogonal context)
                if pass_name == "OF_2":
                    inv_prob = round(1.0 - prob, 4)
                    inv_risk = {"high": "low", "medium": "low", "low": "high"}.get(risk, "medium")
                    geo_text = (f"Entry {idx}/{total}. BADAS orthogonal. SystemID=CASEBELIZE. "
                                f"Risk={inv_risk.upper()}. InvertedSignal=True. "
                                f"Convergence={0.9 - idx/total * 0.4:.2f}. "
                                f"CollisionProb={inv_prob:.4f}.")
                    bin_text = f"badas pass=OF_2_orthogonal ts={ts:.3f} inv_risk={inv_risk}"

                result = engine.infer(
                    binary_text=bin_text,
                    geometry_text=geo_text,
                    language_text=lang_text[:512],
                    stimulus=1.0 + prob * 2.0,       # high-risk = high stimulus
                    is_logical=True,
                    telemetry_stable=(risk != "high"),
                    entry_index=idx,
                    total_entries=total,
                )
                # Attach language_text for record building
                result["_meta_extra"] = {"language_text": lang_text[:512]}

                # Build scalar_inputs for hash
                si = {
                    "binary":   bin_text[:512],
                    "geometry": geo_text[:512],
                    "language": lang_text[:512],
                }
                h = calculate_record_hash(si)
                if h in existing:
                    skipped += 1
                    continue
                existing.add(h)

                # Extract real video frame
                frame_b64 = extract_frame_at(ts, idx + (1000 if pass_name == "OF_2" else 0))

                rec = build_record(window, result, frame_b64, pass_name, idx, total, h)
                out_f.write(json.dumps(rec) + "\n")
                added += 1

                if idx % 50 == 0:
                    print(f"    {pass_name} [{idx}/{total}] V={result['V']:.4f} "
                          f"gate={result['gate_status']} risk={risk}")

            print(f"  {pass_name}: added={added}  skipped={skipped}")
            total_added += added

    corpus_total = sum(1 for l in open(MASTER_CORPUS_PATH, encoding="utf-8") if l.strip())
    size_mb = MASTER_CORPUS_PATH.stat().st_size / 1024 / 1024
    print(f"\n  Total added: {total_added}")
    print(f"  Master corpus: {corpus_total} records  |  {size_mb:.1f} MB")
    print("=" * 64)

if __name__ == "__main__":
    main()
