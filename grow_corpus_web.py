"""grow_corpus_web.py — NextAura v31allC
Scrapes real diverse web content and injects it as new master_corpus records.
Sources: arXiv AI papers, HN, Wikipedia AI topics, GitHub trending.
Each page → chunked into ~300-token passages → each passage becomes one record
with full scalar_inputs + _meta provenance.
"""

import sys, json, time, uuid, hashlib, re
from pathlib import Path
import urllib.request
import urllib.error

ROOT   = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stage1.core.soc_compliance import calculate_record_hash
from stage1.core.triangulation import build_mathematical_relationship
from stage3.inference_engine.main import V31InferenceEngine

MASTER = ROOT / "output" / "master_corpus.jsonl"

# Diverse sources — publicly accessible, no auth
SOURCES = [
    ("arxiv_attention",    "https://arxiv.org/abs/1706.03762"),
    ("arxiv_rl_survey",    "https://arxiv.org/abs/2201.02135"),
    ("wiki_reinforcement", "https://en.wikipedia.org/wiki/Reinforcement_learning"),
    ("wiki_transformer",   "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)"),
    ("wiki_neural_net",    "https://en.wikipedia.org/wiki/Neural_network_(machine_learning)"),
    ("wiki_llm",           "https://en.wikipedia.org/wiki/Large_language_model"),
    ("wiki_world_model",   "https://en.wikipedia.org/wiki/World_model_(robotics)"),
    ("wiki_moe",           "https://en.wikipedia.org/wiki/Mixture_of_experts"),
    ("wiki_knowledge",     "https://en.wikipedia.org/wiki/Knowledge_representation_and_reasoning"),
    ("wiki_autonomous",    "https://en.wikipedia.org/wiki/Autonomous_agent"),
]

CHUNK_SIZE = 300   # tokens per passage (approx chars/4)
MAX_CHUNKS = 40    # max passages per source


def fetch_text(url: str) -> str:
    headers = {"User-Agent": "NextAura-Corpus-Builder/1.0 (research; mlopez@nextaura.fit)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as ex:
        print(f"    [FETCH ERR] {url}: {ex}")
        return ""


def chunk_text(text: str, chunk_chars: int = CHUNK_SIZE * 4) -> list:
    words = text.split()
    chunks = []
    buf = []
    length = 0
    for w in words:
        buf.append(w)
        length += len(w) + 1
        if length >= chunk_chars:
            chunks.append(" ".join(buf))
            buf = []
            length = 0
    if buf:
        chunks.append(" ".join(buf))
    return chunks[:MAX_CHUNKS]


def load_existing_hashes() -> set:
    hashes = set()
    if not MASTER.exists():
        return hashes
    with MASTER.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
                h = rec.get("_meta", {}).get("provenance", {}).get("sha256")
                if h:
                    hashes.add(h)
            except Exception:
                pass
    return hashes


def passage_to_record(passage: str, source_name: str, idx: int, total: int,
                       engine: V31InferenceEngine) -> tuple:
    """Convert a text passage through inference into a full master_corpus record."""
    # Build geometry from source context
    geo_text = f"Entry {idx}/{total}. Source={source_name}. WebCorpus=True. SystemID=CASEBELIZE."
    binary_text = f"web_ingest corpus_chunk={idx} source={source_name} len={len(passage)}"

    result = engine.infer(
        binary_text=binary_text,
        geometry_text=geo_text,
        language_text=passage[:400],
        entry_index=idx,
        total_entries=total,
    )

    math_rel = build_mathematical_relationship(
        M1=result["M1"], M2=result["M2"], M3=result["M3"],
        binary=result["binary_scalar"],
        geometry=result["geometry_scalar"],
        language=result["language_scalar"],
        context_tags=f"web_source={source_name}, chunk={idx}, V={result['V']:.4f}"
    )

    scalar_inputs = {
        "binary":   binary_text[:512],
        "geometry": geo_text[:512],
        "language": passage[:512],
        "mathematical_relationship": math_rel,
    }
    rec_hash = calculate_record_hash(scalar_inputs)

    record = {
        "M1": f"Unbounded thinker exploration_vector={result['M1']:.4f}; web corpus ingestion.",
        "M2": f"Efficiency score={result['M2']:.4f}; neural threshold.",
        "M3": f"Meta-multiplier={result['M3']:.4f}; feedback gain.",
        "scalar_inputs": scalar_inputs,
        "output_tokens": {
            "next_frame_prediction": result["next_frame_prediction"],
            "language_token_output": result["language_token_output"],
        },
        "_meta": {
            "provenance": {
                "uuid": str(uuid.uuid4()),
                "sha256": rec_hash,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "lineage_source": f"web:{source_name}",
                "soc2_integrity_checked": True,
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
            "cycle_source": f"web:{source_name}",
            "web_source_url": source_name,
        },
    }
    return record, rec_hash


def main():
    print("=" * 64)
    print("CORPUS WEB GROW — diverse real-world text ingestion")
    print("=" * 64)

    engine = V31InferenceEngine()
    existing = load_existing_hashes()
    MASTER.parent.mkdir(parents=True, exist_ok=True)

    total_added = 0
    total_skipped = 0

    with MASTER.open("a", encoding="utf-8") as out_f:
        for source_name, url in SOURCES:
            print(f"\n  [{source_name}] {url}")
            text = fetch_text(url)
            if not text:
                print("    → empty, skip")
                continue

            chunks = chunk_text(text)
            print(f"    → {len(chunks)} chunks extracted")

            engine.reset()
            added = skipped = 0

            for i, chunk in enumerate(chunks, 1):
                record, rec_hash = passage_to_record(
                    chunk, source_name, i, len(chunks), engine
                )
                if rec_hash in existing:
                    skipped += 1
                    continue
                existing.add(rec_hash)
                out_f.write(json.dumps(record) + "\n")
                added += 1

            print(f"    added={added}  skipped={skipped}")
            total_added  += added
            total_skipped += skipped
            time.sleep(0.5)   # be polite

    corpus_lines = sum(1 for l in MASTER.read_text(encoding="utf-8").splitlines() if l.strip())
    size_kb = MASTER.stat().st_size / 1024
    print(f"\n{'='*64}")
    print(f"  Total new records: {total_added}  |  Skipped: {total_skipped}")
    print(f"  Master corpus    : {corpus_lines} records, {size_kb:.1f} KB")
    print("=" * 64)


if __name__ == "__main__":
    main()
