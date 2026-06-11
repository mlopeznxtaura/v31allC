"""ingest_physicalai.py — safe streaming ingest of nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes
Streams scene metadata rows via the HuggingFace datasets-server rows API.
NEVER downloads the 15 TB of video payload — only per-scene JSON metadata
(camera intrinsics, world2cam poses, usd transforms, frame counts).
Each row is converted to a single master_corpus record (Golias scalar format).
Safe limits: ROWS_LIMIT rows max, dedup gate, batch fetch with backoff.

Usage:
  python ingest_physicalai.py            # ingest default ROWS_LIMIT rows
  python ingest_physicalai.py 2000       # ingest up to 2000 rows
  HF_TOKEN env var used if set (dataset is public; token optional).
"""
import sys, os, json, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stage3.master_corpus.manager import MASTER_CORPUS_PATH
from stage3.inference_engine.main import V31InferenceEngine
from stage1.core.soc_compliance import calculate_record_hash
from stage1.core.triangulation import build_mathematical_relationship

HF_BASE  = "https://datasets-server.huggingface.co/rows"
DATASET  = "nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes"
CONFIG   = "default"
SPLIT    = "train"
ROWS_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
BATCH    = 100  # rows per API call (API max)


def fetch_rows(offset: int, length: int) -> list:
    import urllib.request
    url = (f"{HF_BASE}?dataset={DATASET}&config={CONFIG}&split={SPLIT}"
           f"&offset={offset}&length={length}")
    headers = {"User-Agent": "NextAura/1.0"}
    token = os.environ.get("HF_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data.get("rows", [])
    except Exception as ex:
        print(f"  [FETCH ERR] @{offset}: {ex}")
        return []


def extract_text(row: dict) -> tuple:
    """Extract (binary, geometry, language) Golias scalars from a scene row.

    binary   <- raw substrate: scene key, frame count (real env constraints)
    geometry <- live telemetry: camera pose/intrinsics/transform
    language <- rich synthesis of the prior scalars
    """
    r = row.get("row", {})
    meta = r.get("json", {}) or {}
    key  = r.get("__key__", "unknown")
    cam  = meta.get("camera", {}) or {}

    frame_count = meta.get("frame_count", 0)
    cam_name    = meta.get("camera_name", "unknown")

    focal = cam.get("focal_length") or []
    principal = cam.get("principal_point") or []
    pose = cam.get("pose_world2cam") or []
    skew = cam.get("skew", 0.0)
    distortion = cam.get("distortion") or []
    usd = meta.get("usd_transform") or []

    def flat(x, n=6):
        out = []
        def walk(v):
            if isinstance(v, (list, tuple)):
                for i in v: walk(i)
            elif isinstance(v, (int, float)):
                out.append(round(float(v), 4))
        walk(x)
        return out[:n]

    binary_t = (f"physicalai scene={key} frames={frame_count} camera={cam_name} "
                f"distortion_terms={len(distortion)}")
    geometry_t = (f"World2cam pose={flat(pose, 8)}. Focal={flat(focal, 4)}. "
                  f"Principal={flat(principal, 4)}. Skew={round(float(skew), 5)}. "
                  f"USD transform={flat(usd, 8)}. SystemID=PHYSICALAI_NVIDIA.")
    language_t = (f"Synthetic physical-interaction scene '{key}' captured by {cam_name} "
                  f"over {frame_count} frames. Camera geometry and world transform define "
                  f"the invariant substrate; next-frame dynamics are the prediction target.")
    return binary_t[:512], geometry_t[:512], language_t[:512]


def load_existing_hashes() -> set:
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


def main():
    print("=" * 64)
    print(f"PHYSICALAI WORLD-MODEL INGEST (streaming, limit={ROWS_LIMIT})")
    print("=" * 64)

    engine = V31InferenceEngine()
    existing = load_existing_hashes()
    print(f"  Existing hashes: {len(existing)}")

    added = skipped = fetch_err = 0
    MASTER_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MASTER_CORPUS_PATH, "a", encoding="utf-8") as out_f:
        offset = 0
        processed = 0
        while processed < ROWS_LIMIT:
            length = min(BATCH, ROWS_LIMIT - processed)
            batch = fetch_rows(offset, length)
            if not batch:
                fetch_err += 1
                if fetch_err > 3:
                    print("  Too many fetch errors; stopping.")
                    break
                time.sleep(2.0)
                continue
            fetch_err = 0

            for row in batch:
                processed += 1
                bin_t, geo_t, lng_t = extract_text(row)

                result = engine.infer(
                    binary_text=bin_t,
                    geometry_text=geo_t,
                    language_text=lng_t,
                    entry_index=processed,
                    total_entries=ROWS_LIMIT,
                )

                scalar_inputs = {
                    "binary": bin_t,
                    "geometry": geo_t,
                    "language": lng_t,
                    "mathematical_relationship": build_mathematical_relationship(
                        result["M1"], result["M2"], result["M3"],
                        result["binary_scalar"], result["geometry_scalar"],
                        result["language_scalar"],
                        f"physicalai,V={result['V']:.4f},if7={result['if7_policy_fusion']:.4f}"
                    )
                }
                h = calculate_record_hash(scalar_inputs)
                if h in existing:
                    skipped += 1
                    continue
                existing.add(h)

                rec = {
                    "M1": f"exploration_vector={result['M1']:.4f}",
                    "M2": f"efficiency={result['M2']:.4f}",
                    "M3": f"meta_multiplier={result['M3']:.4f}",
                    "scalar_inputs": scalar_inputs,
                    "output_tokens": {
                        "next_frame_prediction": result["next_frame_prediction"],
                        "language_token_output": result["language_token_output"],
                    },
                    "_meta": {
                        "provenance": {
                            "uuid": str(uuid.uuid4()),
                            "sha256": h,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "lineage_source": "physicalai:worldmodel-synthetic",
                            "soc2_integrity_checked": True,
                        },
                        "V": round(result["V"], 6),
                        "M1_raw": result["M1"], "M2_raw": result["M2"], "M3_raw": result["M3"],
                        "binary_scalar": result["binary_scalar"],
                        "geometry_scalar": result["geometry_scalar"],
                        "language_scalar": result["language_scalar"],
                        "if7_policy_fusion": result["if7_policy_fusion"],
                        "if7_coherence": result["if7_coherence"],
                        "if7_stability_loss": result["if7_stability_loss"],
                        "orthogonal_vector": result["orthogonal_vector"],
                        "gate_status": result["gate_status"],
                        "cycle_source": "physicalai:worldmodel-synthetic",
                    }
                }
                out_f.write(json.dumps(rec) + "\n")
                added += 1

            offset += len(batch)
            print(f"  progress: {processed}/{ROWS_LIMIT}  added={added}  skipped={skipped}")
            time.sleep(0.4)

    total = sum(1 for l in open(MASTER_CORPUS_PATH, encoding="utf-8") if l.strip())
    kb = MASTER_CORPUS_PATH.stat().st_size / 1024
    print(f"\n  Total added: {added}  (skipped dup: {skipped})")
    print(f"  Master total: {total} records  |  {kb:.1f} KB")
    print("=" * 64)

if __name__ == "__main__":
    main()
