"""ingest_nemotron.py — safe streaming ingest of Nemotron-Cascade-2-SFT-Data
Streams 500 rows from math + science + swe subsets via HuggingFace datasets API.
No auth required (public dataset). Converts SFT chat format to master_corpus records.
Safe limits: 500 rows/subset, dedup gate, no full dataset pull.
"""
import sys, json, time, uuid, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stage3.master_corpus.manager import MASTER_CORPUS_PATH
from stage3.inference_engine.main import V31InferenceEngine
from stage1.core.soc_compliance import calculate_record_hash
from stage1.core.triangulation import build_mathematical_relationship

SUBSETS = [
    ("math",    500),
    ("science", 500),
    ("swe",     300),
]
ROWS_PER_SUBSET = 500
HF_BASE = "https://datasets-server.huggingface.co/rows"
DATASET = "nvidia/Nemotron-Cascade-2-SFT-Data"

def fetch_rows(subset: str, offset: int, length: int) -> list:
    import urllib.request
    url = (f"{HF_BASE}?dataset={DATASET}&config={subset}&split=train"
           f"&offset={offset}&length={length}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NextAura/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data.get("rows", [])
    except Exception as ex:
        print(f"  [FETCH ERR] {subset}@{offset}: {ex}")
        return []

def extract_text(row: dict) -> tuple:
    """Extract (binary, geometry, language) from SFT chat format."""
    msgs = row.get("row", {}).get("messages", [])
    domain = row.get("row", {}).get("domain", "unknown")
    source = row.get("row", {}).get("source", "nemotron")

    system_txt = next((m["content"] for m in msgs if m.get("role") == "system"), "")
    user_txt   = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    asst_txt   = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")

    binary_t   = f"nemotron domain={domain} source={source} msg_count={len(msgs)}"
    geometry_t = f"SFT entry. Domain={domain}. Source={source}. SystemID=CASEBELIZE. PhaseReady=True."
    language_t = f"{system_txt[:100]} | USER: {user_txt[:300]} | ASST: {asst_txt[:200]}"
    return binary_t, geometry_t, language_t[:512]

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
    print("NEMOTRON-CASCADE-2 SFT INGEST (safe streaming)")
    print("=" * 64)

    engine = V31InferenceEngine()
    existing = load_existing_hashes()
    print(f"  Existing hashes: {len(existing)}")

    total_added = 0
    MASTER_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MASTER_CORPUS_PATH, "a", encoding="utf-8") as out_f:
        for subset, limit in SUBSETS:
            print(f"\n  [{subset}] fetching up to {limit} rows ...")
            engine.reset()
            added = skipped = fetch_err = 0

            # Fetch in batches of 100
            batch_size = 100
            rows_fetched = []
            for offset in range(0, limit, batch_size):
                batch = fetch_rows(subset, offset, min(batch_size, limit - offset))
                if not batch:
                    fetch_err += 1
                    if fetch_err > 3: break
                    time.sleep(1.0)
                    continue
                rows_fetched.extend(batch)
                time.sleep(0.4)

            print(f"  Fetched {len(rows_fetched)} rows")

            for idx, row in enumerate(rows_fetched, 1):
                bin_t, geo_t, lng_t = extract_text(row)

                result = engine.infer(
                    binary_text=bin_t,
                    geometry_text=geo_t,
                    language_text=lng_t,
                    entry_index=idx,
                    total_entries=len(rows_fetched),
                )

                scalar_inputs = {
                    "binary": bin_t[:512],
                    "geometry": geo_t[:512],
                    "language": lng_t[:512],
                    "mathematical_relationship": build_mathematical_relationship(
                        result["M1"], result["M2"], result["M3"],
                        result["binary_scalar"], result["geometry_scalar"],
                        result["language_scalar"],
                        f"nemotron_{subset},V={result['V']:.4f},if7={result['if7_policy_fusion']:.4f}"
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
                            "lineage_source": f"nemotron:{subset}",
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
                        "cycle_source": f"nemotron:{subset}",
                    }
                }
                out_f.write(json.dumps(rec) + "\n")
                added += 1

            total_added += added
            print(f"  {subset}: added={added}  skipped={skipped}")

    total = sum(1 for l in open(MASTER_CORPUS_PATH, encoding="utf-8") if l.strip())
    kb = MASTER_CORPUS_PATH.stat().st_size / 1024
    print(f"\n  Total added: {total_added}")
    print(f"  Master total: {total} records  |  {kb:.1f} KB")
    print("=" * 64)

if __name__ == "__main__":
    main()
