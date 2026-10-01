"""run_golias_full.py - GOLIAS Full Cycle w/ Qwen M1/M3 + V31Model M2"""
import sys, os, json, time, math, shutil
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONUNBUFFERED"] = "1"

OLLAMA_URL  = "http://localhost:11434/api/generate"
QWEN_MODEL  = "hf.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M"
ENRICH_MAX  = 60
CYCLES=5; EPOCHS=15; BATCH=32; LR=1e-4; WEIGHT_DECAY=5e-5; GRAD_CLIP=1.0

def log(msg): print(msg, flush=True)

def qwen_enrich(text, role, timeout=12):
    if not text or len(text.strip()) < 5: return text
    prompts = {"M1": f"Binary logic 1 sentence: {text[:120]}", "M3": f"Semantic synthesis 1 sentence: {text[:120]}"}
    try:
        r = requests.post(OLLAMA_URL, json={"model": QWEN_MODEL, "prompt": prompts.get(role, text[:120]),
            "stream": False, "options": {"num_predict": 40, "temperature": 0.2}}, timeout=timeout)
        if r.status_code == 200:
            out = r.json().get("response","").strip()
            return out[:200] if out else text
    except Exception as ex:
        log(f"  [qwen] {role} skip: {ex}")
    return text

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=4)
        models = [m["name"] for m in r.json().get("models",[])]
        log(f"  [ollama] online  models={models[:3]}")
        return True
    except Exception as ex:
        log(f"  [ollama] OFFLINE - {ex}"); return False

log("="*60); log("GOLIAS FULL CYCLE - Qwen M1/M3 + V31Model M2"); log("="*60)
ollama_ok = check_ollama()
if not ollama_ok: log("  WARNING: ollama offline, raw text mode")

log("="*60); log("STEP 1: CORPUS GROW"); log("="*60)
from stage3.master_corpus.manager import MASTER_CORPUS_PATH
from stage3.inference_engine.main import V31InferenceEngine
from stage1.core.soc_compliance import calculate_record_hash
from stage1.core.triangulation import build_mathematical_relationship
import uuid

CORPORA = [ROOT/"output"/"corpus_4563.jsonl", ROOT/"output"/"corpus_baby_ingested.jsonl",
    ROOT/"output"/"corpus_iphone.jsonl", ROOT/"output"/"corpus_pc.jsonl",
    ROOT/"trimodal_observer_corpus.jsonl", ROOT/"corpus_v31_sample.jsonl"]

def load_hashes():
    hashes = set()
    if not MASTER_CORPUS_PATH.exists(): return hashes
    with open(MASTER_CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            try:
                h=json.loads(s).get("_meta",{}).get("provenance",{}).get("sha256")
                if h: hashes.add(h)
            except: pass
    return hashes

engine=V31InferenceEngine(); existing=load_hashes(); total_added=0
log(f"  Existing: {len(existing)}")

with open(MASTER_CORPUS_PATH,"a",encoding="utf-8") as out_f:
    for cp in CORPORA:
        if not cp.exists(): log(f"  SKIP: {cp.name}"); continue
        recs=[json.loads(l) for l in cp.read_text(encoding="utf-8").splitlines() if l.strip()]
        sample=recs[:ENRICH_MAX]; engine.reset(); added=0
        log(f"  {cp.name} total={len(recs)} enriching={len(sample)}")
        for idx,src in enumerate(sample,1):
            si=src.get("scalar_inputs",{})
            bin_t=si.get("binary","") or str(src.get("M1",""))
            geo_t=si.get("geometry","") or str(src.get("M2",""))
            lng_t=si.get("language","") or str(src.get("M3",""))
            if len(geo_t.strip())<20: geo_t=f"Entry {idx}. SystemID=CASEBELIZE. {geo_t}".strip()
            bin_t_e=qwen_enrich(bin_t,"M1") if ollama_ok else bin_t
            lng_t_e=qwen_enrich(lng_t,"M3") if ollama_ok else lng_t
            result=engine.infer(binary_text=bin_t_e,geometry_text=geo_t,language_text=lng_t_e,
                stimulus=float(src.get("stimulus",1.0)),entry_index=idx,total_entries=len(sample))
            scalar_inputs={"binary":bin_t_e[:512],"geometry":geo_t[:512],"language":lng_t_e[:512],
                "mathematical_relationship":build_mathematical_relationship(
                    result["M1"],result["M2"],result["M3"],
                    result["binary_scalar"],result["geometry_scalar"],result["language_scalar"],
                    f"src={cp.name},V={result['V']:.4f},qwen={ollama_ok}")}
            h=calculate_record_hash(scalar_inputs)
            if h in existing: continue
            existing.add(h); added+=1
            rec={"M1":f"exploration_vector={result['M1']:.4f}","M2":f"efficiency={result['M2']:.4f}",
                "M3":f"meta_multiplier={result['M3']:.4f}","scalar_inputs":scalar_inputs,
                "output_tokens":{"next_frame_prediction":result["next_frame_prediction"],
                    "language_token_output":result["language_token_output"]},
                "_meta":{"provenance":{"uuid":str(uuid.uuid4()),"sha256":h,
                    "timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                    "lineage_source":cp.name,"qwen_m1":ollama_ok,"qwen_m3":ollama_ok},
                    "V":round(result["V"],6),"M1_raw":result["M1"],"M2_raw":result["M2"],
                    "M3_raw":result["M3"],"binary_scalar":result["binary_scalar"],
                    "geometry_scalar":result["geometry_scalar"],"language_scalar":result["language_scalar"],
                    "gate_status":result["gate_status"],"cycle_source":cp.name}}
            out_f.write(json.dumps(rec)+"\n")
        total_added+=added; log(f"    added={added}")

corpus_total=sum(1 for l in open(MASTER_CORPUS_PATH,encoding="utf-8") if l.strip())
log(f"\n  NEW: {total_added}  Master total: {corpus_total}")

log("="*60); log("STEP 2: MP4 FRAME INJECTION"); log("="*60)
import base64, numpy as np
from PIL import Image
FRAMES_DIR=ROOT/"output"/"_mp4_frames"; frame_paths=sorted(FRAMES_DIR.glob("frame_*.png"))
log(f"  Frame files: {len(frame_paths)}")
if frame_paths:
    b64s=[base64.b64encode(np.array(Image.open(p).convert("RGB").resize((64,64)),dtype="uint8").tobytes()).decode("ascii") for p in frame_paths]
    records=[]
    with open(MASTER_CORPUS_PATH,encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if s:
                try: records.append(json.loads(s))
                except: pass
    patched=0
    for idx,rec in enumerate(records):
        if not rec.get("_meta",{}).get("pixel_frame_base64"):
            rec.setdefault("_meta",{})["pixel_frame_base64"]=b64s[idx%len(b64s)]; patched+=1
    with open(MASTER_CORPUS_PATH,"w",encoding="utf-8") as f:
        for rec in records: f.write(json.dumps(rec)+"\n")
    log(f"  Patched {patched}  Corpus: {MASTER_CORPUS_PATH.stat().st_size/1024:.1f} KB")
else: log("  No frames, skip")

log("="*60); log(f"STEP 3: DEEP TRAIN {CYCLES}x{EPOCHS} - V31Model M2"); log("="*60)
import torch
from torch.optim import AdamW; from torch.optim.lr_scheduler import LambdaLR
from stage2.training_engine.neural_model import V31Model, count_params
from stage2.training_engine.asymmetric_trainer import get_deterministic_token_id, decode_and_pool_frame
from stage2.training_engine.state_registry import AuraCycleManager

CKPT_ACTIVE=ROOT/"models"/"v31_neural_model.pt"

def load_frames(path):
    frames=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                rec=json.loads(line); meta=rec.get("_meta",{}); ot=rec.get("output_tokens",{})
                frames.append({"m1":float(meta.get("M1_raw",1.0)),"m2":float(meta.get("M2_raw",1.0)),
                    "m3":float(meta.get("M3_raw",1.0)),"geo":torch.ones(32)*float(meta.get("geometry_scalar",0.0)),
                    "bin":torch.ones(32)*float(meta.get("binary_scalar",0.0)),
                    "lng":torch.ones(128)*float(meta.get("language_scalar",0.0)),
                    "tri":torch.ones(32)*float(meta.get("V",0.5)),
                    "next_frame_target":decode_and_pool_frame(meta.get("pixel_frame_base64","")),
                    "next_token_id":get_deterministic_token_id(ot.get("language_token_output",""))})
            except: continue
    return frames

def make_sched(opt,total,warmup):
    def lr_fn(e):
        if e<warmup: return (e+1)/max(1,warmup)
        p=(e-warmup)/max(1,total-warmup); return 0.5*(1+math.cos(math.pi*p))
    return LambdaLR(opt,lr_fn)

def train_epoch(model,frames,opt,device):
    model.train(); mse_fn=torch.nn.MSELoss(); ce_fn=torch.nn.CrossEntropyLoss(); tl=tm=tc=nb=0
    for i in range(0,len(frames),BATCH):
        b=frames[i:i+BATCH]
        m1=torch.tensor([f["m1"] for f in b],dtype=torch.float32).to(device)
        m2=torch.tensor([f["m2"] for f in b],dtype=torch.float32).to(device)
        m3=torch.tensor([f["m3"] for f in b],dtype=torch.float32).to(device)
        geo=torch.stack([f["geo"] for f in b]).to(device); bn=torch.stack([f["bin"] for f in b]).to(device)
        lng=torch.stack([f["lng"] for f in b]).to(device); tri=torch.stack([f["tri"] for f in b]).to(device)
        tf=torch.stack([f["next_frame_target"] for f in b]).to(device)
        tt=torch.tensor([f["next_token_id"] for f in b],dtype=torch.long).to(device)
        opt.zero_grad(); pf,pt=model(m1,m2,m3,geo,bn,lng,tri)
        lm=mse_fn(pf,tf); lc=ce_fn(pt,tt); loss=lm+0.1*lc
        if torch.isnan(loss): opt.zero_grad(); continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step()
        tl+=loss.item(); tm+=lm.item(); tc+=lc.item(); nb+=1
    n=max(nb,1); return tl/n,tm/n,tc/n

device="cuda" if torch.cuda.is_available() else "cpu"
log(f"  Device: {device}  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
model=V31Model().to(device); log(f"  {count_params(model)}")
if CKPT_ACTIVE.exists():
    try:
        ckpt=torch.load(str(CKPT_ACTIVE),map_location=device,weights_only=True); sd=ckpt.get("model_state_dict",ckpt)
        miss,unex=model.load_state_dict(sd,strict=False); log(f"  Loaded ckpt miss={len(miss)} unex={len(unex)}")
    except Exception as ex: log(f"  Fresh start ({ex})")
opt=AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY); prev_best=float("inf")

for cycle in range(1,CYCLES+1):
    log(f"\n  -- CYCLE {cycle}/{CYCLES} --")
    frames=load_frames(MASTER_CORPUS_PATH)
    log(f"  Frames: {len(frames):,}")
    sched=make_sched(opt,EPOCHS,1); best=float("inf"); best_ckpt=ROOT/"models"/f"v31_cycle{cycle}_best.pt"
    for ep in range(1,EPOCHS+1):
        t0=time.time(); loss,mse,ce=train_epoch(model,frames,opt,device); sched.step()
        lr_now=sched.get_last_lr()[0]; mark=" *" if loss<best else ""
        if loss<best: best=loss; torch.save({"model_state_dict":model.state_dict()},str(best_ckpt))
        log(f"  C{cycle} E{ep:02d}/{EPOCHS} loss={loss:.4f} mse={mse:.4f} ce={ce:.4f} lr={lr_now:.2e} {round(time.time()-t0,1)}s{mark}")
    shutil.copy(str(best_ckpt),str(CKPT_ACTIVE))
    delta=prev_best-best
    ch=AuraCycleManager.calculate_corpus_hash(str(MASTER_CORPUS_PATH))
    sh=AuraCycleManager.compute_state_hash(ch,EPOCHS*cycle,BATCH,LR)
    AuraCycleManager.register_state(sh,str(MASTER_CORPUS_PATH),ch,EPOCHS*cycle,BATCH,LR,
        {"loss":round(best,6),"mse":mse,"ce":ce,"records":len(frames),"delta":round(delta,6),"qwen":ollama_ok},
        str(best_ckpt.relative_to(ROOT)))
    log(f"  Cycle {cycle} best={best:.4f} delta={delta:+.4f} registered {sh[:16]}"); prev_best=best

log("TRAINING COMPLETE")

log("="*60); log("STEP 4: INFERENCE PROBES"); log("="*60)
model.eval(); engine2=V31InferenceEngine()
questions=["What is attention in transformers?","How does MoE routing work?",
    "What is RLHF?","Describe world model in robotics.","Purpose of layer normalization?",
    "How does V31 represent knowledge?","What is Agent Trust Protocol?",
    "How do SOC2 provenance blocks work?","What drives MSE in V31 visual head?",
    "How should master corpus grow over time?"]
results_log=[]
for i,q in enumerate(questions,1):
    bin_p=f"question_id={i} probe=True qwen={ollama_ok}"
    lng_p=qwen_enrich(q,"M3") if ollama_ok else q
    bin_p=qwen_enrich(bin_p,"M1") if ollama_ok else bin_p
    engine2.reset()
    result=engine2.infer(binary_text=bin_p,geometry_text=f"Probe {i}. SystemID=CASEBELIZE.",
        language_text=lng_p,entry_index=i,total_entries=len(questions))
    results_log.append({"q":q,"V":round(result["V"],4),"M1":round(result["M1"],4),
        "M2":round(result["M2"],4),"M3":round(result["M3"],4),
        "gate":result["gate_status"],"token":result["language_token_output"][:60]})
    log(f"  Q{i:02d} V={result['V']:.4f} gate={result['gate_status']:<8} M1={result['M1']:.3f} M2={result['M2']:.3f} M3={result['M3']:.3f}")
    log(f"       {q}"); log(f"       -> {result['language_token_output']}"); log("")

qa_corpus=ROOT/"output"/"corpus_live_qa.jsonl"
with open(qa_corpus,"w",encoding="utf-8") as f:
    for r in results_log:
        rec={"scalar_inputs":{"binary":f"qa_probe qwen={ollama_ok}",
            "geometry":"LiveQA. SystemID=CASEBELIZE.","language":r["q"],
            "mathematical_relationship":r["token"]},
            "_meta":{"provenance":{"sha256":"","lineage_source":"live_qa",
                "timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                "qwen_m1":ollama_ok,"qwen_m3":ollama_ok},
                "V":r["V"],"M1_raw":r["M1"],"M2_raw":r["M2"],"M3_raw":r["M3"],
                "gate_status":r["gate"],"cycle_source":"live_qa"},
            "output_tokens":{"language_token_output":r["token"]}}
        f.write(json.dumps(rec)+"\n")
log(f"  Q&A saved: {len(results_log)} records -> corpus_live_qa.jsonl")
log("="*60); log("GOLIAS FULL CYCLE COMPLETE")
log(f"  Master corpus : {corpus_total} records")
log(f"  Qwen M1/M3    : {ollama_ok}")
log(f"  Final M2 loss : {prev_best:.4f}")
log(f"  Next run      : python run_golias_full.py")
log("="*60)
