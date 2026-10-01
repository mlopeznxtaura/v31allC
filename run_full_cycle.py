"""run_full_cycle.py — grow + patch frames + deep train + live inference Q&A"""
import sys, os, json, time, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ["PYTHONUNBUFFERED"] = "1"

def log(msg):
    print(msg, flush=True)

# ── STEP 1: Grow corpus ───────────────────────────────────────────────────────
log("\n" + "="*60)
log("STEP 1: CYCLE AND GROW")
log("="*60)

from stage3.master_corpus.manager import MASTER_CORPUS_PATH
from stage3.inference_engine.main import V31InferenceEngine
from stage1.core.soc_compliance import calculate_record_hash
from stage1.core.triangulation import build_mathematical_relationship
import uuid, hashlib

CORPORA = [
    ROOT / "output" / "corpus_4563.jsonl",
    ROOT / "output" / "corpus_baby_ingested.jsonl",
    ROOT / "output" / "corpus_iphone.jsonl",
    ROOT / "output" / "corpus_pc.jsonl",
    ROOT / "trimodal_observer_corpus.jsonl",
    ROOT / "corpus_v31_sample.jsonl",
]

def load_hashes():
    hashes = set()
    if not MASTER_CORPUS_PATH.exists():
        return hashes
    with open(MASTER_CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                h = json.loads(s).get("_meta",{}).get("provenance",{}).get("sha256")
                if h: hashes.add(h)
            except: pass
    return hashes

engine = V31InferenceEngine()
existing = load_hashes()
log(f"  Existing master records: {len(existing)}")
total_added = 0

with open(MASTER_CORPUS_PATH, "a", encoding="utf-8") as out_f:
    for cp in CORPORA:
        if not cp.exists():
            log(f"  SKIP: {cp.name}"); continue
        recs = [json.loads(l) for l in cp.read_text(encoding="utf-8").splitlines() if l.strip()]
        engine.reset()
        added = 0
        for idx, src in enumerate(recs, 1):
            si = src.get("scalar_inputs", {})
            bin_t = si.get("binary","") or str(src.get("M1",""))
            geo_t = si.get("geometry","") or str(src.get("M2",""))
            lng_t = si.get("language","") or str(src.get("M3",""))
            if len(geo_t.strip()) < 20:
                geo_t = f"Entry {idx}/{len(recs)}. SystemID=CASEBELIZE. {geo_t}".strip()
            result = engine.infer(
                binary_text=bin_t, geometry_text=geo_t, language_text=lng_t,
                stimulus=float(src.get("stimulus",1.0)),
                is_logical=bool(src.get("is_logical",True)),
                telemetry_stable=bool(src.get("telemetry_stable",True)),
                entry_index=idx, total_entries=len(recs),
            )
            scalar_inputs = {
                "binary": bin_t[:512], "geometry": geo_t[:512], "language": lng_t[:512],
                "mathematical_relationship": build_mathematical_relationship(
                    result["M1"], result["M2"], result["M3"],
                    result["binary_scalar"], result["geometry_scalar"], result["language_scalar"],
                    f"src={cp.name},V={result['V']:.4f}"
                )
            }
            h = calculate_record_hash(scalar_inputs)
            if h in existing: continue
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
                    "provenance": {"uuid": str(uuid.uuid4()), "sha256": h,
                                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                   "lineage_source": cp.name, "soc2_integrity_checked": True},
                    "V": round(result["V"],6), "M1_raw": result["M1"],
                    "M2_raw": result["M2"], "M3_raw": result["M3"],
                    "binary_scalar": result["binary_scalar"],
                    "geometry_scalar": result["geometry_scalar"],
                    "language_scalar": result["language_scalar"],
                    "gate_status": result["gate_status"],
                    "cycle_source": cp.name,
                }
            }
            out_f.write(json.dumps(rec) + "\n")
            added += 1
        total_added += added
        log(f"  {cp.name:<40} added={added}")

corpus_total = sum(1 for l in open(MASTER_CORPUS_PATH, encoding="utf-8") if l.strip())
log(f"  Total new: {total_added}  |  Master total: {corpus_total}")

# ── STEP 2: Re-inject real mp4 frames ────────────────────────────────────────
log("\n" + "="*60)
log("STEP 2: RE-INJECT MP4 PIXEL FRAMES")
log("="*60)

import base64, numpy as np
from PIL import Image

FRAMES_DIR = ROOT / "output" / "_mp4_frames"
frame_paths = sorted(FRAMES_DIR.glob("frame_*.png"))
log(f"  Frame files: {len(frame_paths)}")

if frame_paths:
    b64s = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB").resize((64,64))
        arr = np.array(img, dtype="uint8")
        b64s.append(base64.b64encode(arr.tobytes()).decode("ascii"))

    records = []
    with open(MASTER_CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try: records.append(json.loads(s))
                except: pass

    n = len(b64s)
    patched = 0
    for idx, rec in enumerate(records):
        if not rec.get("_meta",{}).get("pixel_frame_base64"):
            rec.setdefault("_meta",{})["pixel_frame_base64"] = b64s[idx % n]
            patched += 1

    with open(MASTER_CORPUS_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    log(f"  Patched {patched} new records with real mp4 frames")
    log(f"  Corpus size: {MASTER_CORPUS_PATH.stat().st_size/1024:.1f} KB")
else:
    log("  No frame files found, skipping.")

# ── STEP 3: Deep train ────────────────────────────────────────────────────────
log("\n" + "="*60)
log("STEP 3: DEEP TRAIN (5 cycles x 15 epochs)")
log("="*60)

import torch, math, shutil
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from stage2.training_engine.neural_model import V31Model, count_params
from stage2.training_engine.asymmetric_trainer import get_deterministic_token_id, decode_and_pool_frame
from stage2.training_engine.state_registry import AuraCycleManager

CKPT_ACTIVE = ROOT / "models" / "v31_neural_model.pt"
CYCLES=5; EPOCHS=15; BATCH=32; LR=1e-4; WEIGHT_DECAY=5e-5; GRAD_CLIP=1.0

def load_frames(path):
    frames = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                rec=json.loads(line); meta=rec.get("_meta",{}); si=rec.get("scalar_inputs",{}); ot=rec.get("output_tokens",{})
                frames.append({
                    "m1": float(meta.get("M1_raw",1.0)), "m2": float(meta.get("M2_raw",1.0)),
                    "m3": float(meta.get("M3_raw",1.0)),
                    "geo": torch.ones(32)*float(meta.get("geometry_scalar",0.0)),
                    "bin": torch.ones(32)*float(meta.get("binary_scalar",0.0)),
                    "lng": torch.ones(128)*float(meta.get("language_scalar",0.0)),
                    "tri": torch.ones(32)*float(meta.get("V",0.5)),
                    "next_frame_target": decode_and_pool_frame(meta.get("pixel_frame_base64","")),
                    "next_token_id": get_deterministic_token_id(ot.get("language_token_output","")),
                })
            except: continue
    return frames

def make_sched(opt, total, warmup):
    def lr_fn(e):
        if e < warmup: return (e+1)/max(1,warmup)
        p = (e-warmup)/max(1,total-warmup)
        return 0.5*(1+math.cos(math.pi*p))
    return LambdaLR(opt, lr_fn)

def train_epoch(model, frames, opt, device, batch_size):
    model.train()
    tl=tm=tc=nb=0
    mse_fn=torch.nn.MSELoss(); ce_fn=torch.nn.CrossEntropyLoss()
    for i in range(0, len(frames), batch_size):
        b=frames[i:i+batch_size]
        m1=torch.tensor([f["m1"] for f in b],dtype=torch.float32).to(device)
        m2=torch.tensor([f["m2"] for f in b],dtype=torch.float32).to(device)
        m3=torch.tensor([f["m3"] for f in b],dtype=torch.float32).to(device)
        geo=torch.stack([f["geo"] for f in b]).to(device)
        bn=torch.stack([f["bin"] for f in b]).to(device)
        lng=torch.stack([f["lng"] for f in b]).to(device)
        tri=torch.stack([f["tri"] for f in b]).to(device)
        tf=torch.stack([f["next_frame_target"] for f in b]).to(device)
        tt=torch.tensor([f["next_token_id"] for f in b],dtype=torch.long).to(device)
        opt.zero_grad()
        pf,pt=model(m1,m2,m3,geo,bn,lng,tri)
        lm=mse_fn(pf,tf); lc=ce_fn(pt,tt); loss=lm+0.1*lc
        if torch.isnan(loss): opt.zero_grad(); continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        tl+=loss.item(); tm+=lm.item(); tc+=lc.item(); nb+=1
    n=max(nb,1)
    return tl/n, tm/n, tc/n

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"  Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
model = V31Model().to(device)
log(f"  {count_params(model)}")

if CKPT_ACTIVE.exists():
    try:
        ckpt=torch.load(str(CKPT_ACTIVE), map_location=device, weights_only=True)
        sd=ckpt.get("model_state_dict", ckpt)
        miss,unex=model.load_state_dict(sd, strict=False)
        log(f"  Loaded checkpoint  missing={len(miss)} unexpected={len(unex)}")
    except Exception as ex:
        log(f"  Fresh start ({ex})")

opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

for cycle in range(1, CYCLES+1):
    log(f"\n  --- CYCLE {cycle}/{CYCLES} ---")
    frames = load_frames(MASTER_CORPUS_PATH)
    log(f"  Frames: {len(frames):,}  batches/epoch: {len(frames)//BATCH}")
    sched = make_sched(opt, EPOCHS, 1)
    best = float("inf")
    best_ckpt = ROOT / "models" / f"v31_cycle{cycle}_best.pt"
    for ep in range(1, EPOCHS+1):
        t0=time.time()
        loss,mse,ce = train_epoch(model, frames, opt, device, BATCH)
        sched.step()
        lr_now=sched.get_last_lr()[0]
        mark=" *" if loss<best else ""
        if loss<best:
            best=loss
            torch.save({"model_state_dict": model.state_dict()}, str(best_ckpt))
        log(f"  C{cycle} E{ep:02d}/{EPOCHS}  loss={loss:.4f}  mse={mse:.4f}  ce={ce:.4f}  lr={lr_now:.2e}  {round(time.time()-t0,1)}s{mark}")
    shutil.copy(str(best_ckpt), str(CKPT_ACTIVE))
    ch=AuraCycleManager.calculate_corpus_hash(str(MASTER_CORPUS_PATH))
    sh=AuraCycleManager.compute_state_hash(ch, EPOCHS*cycle, BATCH, LR)
    AuraCycleManager.register_state(sh,str(MASTER_CORPUS_PATH),ch,EPOCHS*cycle,BATCH,LR,
        {"loss":round(best,6),"mse":mse,"ce":ce,"records":len(frames),"cycle":cycle},
        str(best_ckpt.relative_to(ROOT)))
    log(f"  Cycle {cycle} done  best={best:.4f}  registered {sh[:16]}")

log("\n" + "="*60)
log("TRAINING COMPLETE")
log("="*60)

# ── STEP 4: Live inference Q&A loop ──────────────────────────────────────────
log("\n" + "="*60)
log("STEP 4: LIVE INFERENCE Q&A")
log("="*60)

model.eval()
engine2 = V31InferenceEngine()

questions = [
    ("What is the role of attention in transformer models?",
     "Attention allows the model to weigh the importance of different tokens."),
    ("How does a mixture of experts model route tokens?",
     "A router network assigns each token to a subset of expert networks."),
    ("What is reinforcement learning?",
     "An agent learns to maximize cumulative reward through environment interaction."),
    ("Describe a world model in robotics.",
     "An internal representation of the environment used for planning and prediction."),
    ("What is the purpose of layer normalization?",
     "It stabilizes training by normalizing activations across features."),
    ("How does the V31 system represent knowledge?",
     "Via trimodal scalar vectors: binary, geometry, and language dimensions."),
    ("What is the Agent Trust Protocol?",
     "Identity, credentialing, and audit infrastructure for autonomous AI agents."),
    ("How do SOC2 provenance blocks secure corpus records?",
     "By attaching SHA-256 hashes and timestamps to each ingested record."),
    ("What drives MSE loss in the V31 visual head?",
     "Reconstruction of 64x64 RGB pixel frames extracted from screen recordings."),
    ("How should the master corpus grow over time?",
     "Each inference cycle appends deduplicated records with increasing V values."),
]

log(f"\n  Running {len(questions)} probe questions through V31InferenceEngine...\n")

results_log = []
for i, (q, expected) in enumerate(questions, 1):
    engine2.reset()
    result = engine2.infer(
        binary_text=f"question_id={i} probe=True",
        geometry_text=f"Inference probe {i}/{len(questions)}. SystemID=CASEBELIZE. LiveQA=True.",
        language_text=q,
        entry_index=i,
        total_entries=len(questions),
    )
    entry = {
        "question": q,
        "expected_theme": expected,
        "V": round(result["V"], 4),
        "M1": round(result["M1"], 4),
        "M2": round(result["M2"], 4),
        "M3": round(result["M3"], 4),
        "gate": result["gate_status"],
        "token_out": result["language_token_output"][:60],
    }
    results_log.append(entry)
    log(f"  Q{i:02d} V={entry['V']:.4f} gate={entry['gate']:<6} M1={entry['M1']:.3f} M2={entry['M2']:.3f} M3={entry['M3']:.3f}")
    log(f"       Q: {q[:70]}")
    log(f"       T: {entry['token_out']}")
    log("")

# Save Q&A log as new corpus records for next cycle
qa_corpus = ROOT / "output" / "corpus_live_qa.jsonl"
with open(qa_corpus, "w", encoding="utf-8") as f:
    for r in results_log:
        rec = {
            "scalar_inputs": {
                "binary": f"qa_probe question_id={results_log.index(r)+1}",
                "geometry": "LiveQA probe. SystemID=CASEBELIZE.",
                "language": r["question"],
                "mathematical_relationship": r["expected_theme"],
            },
            "_meta": {
                "provenance": {"sha256": "", "lineage_source": "live_qa",
                               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                "V": r["V"], "M1_raw": r["M1"], "M2_raw": r["M2"], "M3_raw": r["M3"],
                "gate_status": r["gate"], "cycle_source": "live_qa",
            },
            "output_tokens": {"language_token_output": r["token_out"]},
        }
        f.write(json.dumps(rec) + "\n")

log(f"  Q&A corpus saved: {qa_corpus.name}  ({len(results_log)} records)")
log("  -> Add to CORPORA list for next cycle_and_grow run to feed feedback back in.")
log("\nRUN COMPLETE.")
