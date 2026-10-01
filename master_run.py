"""master_run.py - NextAura v31allC full orchestration
  M1/M3 -> qwen (ollama)
  M2    -> custom .pt (V31Model via neural_inference)
  Flow  -> full_cycle_v2 -> cycle_and_grow (next iteration)
  All print() -> stdout -> piped to C:/golias/cycle.log -> GUI terminal mirror
"""
import sys, os, json, time, runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONUNBUFFERED"] = "1"

QWEN_MODEL = "hf.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M"
OLLAMA_URL  = "http://localhost:11434/api/generate"

def log(msg): print(msg, flush=True)

# ── STEP 0: verify ollama + qwen, patch M1/M3 language output ─────────────────
log("="*60)
log("STEP 0: LOAD QWEN -> M1/M3 LANGUAGE OUTPUT")
log("="*60)

_qwen_ok = False
try:
    import requests
    r = requests.post(OLLAMA_URL, json={
        "model": QWEN_MODEL, "prompt": "ping", "stream": False,
        "options": {"num_predict": 4}
    }, timeout=30)
    tok = r.json().get("response","").strip()
    log(f"  Qwen M1/M3 ONLINE: response='{tok[:40]}'")
    _qwen_ok = True
except Exception as e:
    log(f"  WARNING Qwen offline ({e}) -- scalar fallback active")

if _qwen_ok:
    import requests
    from stage3.inference_engine.main import V31InferenceEngine
    _orig_infer = V31InferenceEngine.infer.__wrapped__ if hasattr(V31InferenceEngine.infer, '__wrapped__') else V31InferenceEngine.infer

    def _m1m3_qwen_infer(self, binary_text, geometry_text, language_text, **kw):
        result = _orig_infer(self, binary_text, geometry_text, language_text, **kw)
        try:
            prompt = (
                f"[M1={result['M1']:.4f} M3={result['M3']:.4f} "
                f"V={result['V']:.4f} gate={result['gate_status']}] "
                f"context: {language_text[:180]} | next_token:"
            )
            rr = requests.post(OLLAMA_URL, json={
                "model": QWEN_MODEL, "prompt": prompt,
                "stream": False, "options": {"num_predict": 20}
            }, timeout=15)
            tok = rr.json().get("response","").strip()[:80]
            if tok:
                result["language_token_output"] = tok
                log(f"  [M1/M3 qwen] V={result['V']:.4f} tok='{tok[:40]}'")
        except:
            pass
        return result

    _m1m3_qwen_infer.__wrapped__ = _orig_infer
    V31InferenceEngine.infer = _m1m3_qwen_infer
    log("  M1/M3 qwen patch ACTIVE on V31InferenceEngine")

# ── STEP 1: load .pt M2 weights (neural inference) ────────────────────────────
log("\n" + "="*60)
log("STEP 1: LOAD .PT MODEL -> M2 (V31NeuralInferenceEngine)")
log("="*60)
from stage3.inference_engine.neural_inference import V31NeuralInferenceEngine
_ckpt = str(ROOT / "models" / "v31_neural_model.pt")
_neural = V31NeuralInferenceEngine(checkpoint_path=_ckpt)
if _neural.is_loaded:
    log(f"  M2 .pt LOADED: {_ckpt}")
    log(f"  V31Model params: {sum(p.numel() for p in _neural.model.parameters()):,}")
else:
    log(f"  WARNING: M2 .pt failed to load -- mathematical fallback")

# ── STEP 2: full_cycle_v2 (ingest all corpora + deep train) ───────────────────
log("\n" + "="*60)
log("STEP 2: FULL_CYCLE_V2 -- ingest + grow + deep train")
log("="*60)
_t0 = time.time()
runpy.run_path(str(ROOT / "full_cycle_v2.py"), run_name="__main__")
log(f"\n  full_cycle_v2 done in {time.time()-_t0:.1f}s")

# ── STEP 3: cycle_and_grow (next iteration on fresh weights) ──────────────────
log("\n" + "="*60)
log("STEP 3: CYCLE_AND_GROW -- next iteration")
log("="*60)
_t1 = time.time()
runpy.run_path(str(ROOT / "cycle_and_grow.py"), run_name="__main__")
log(f"\n  cycle_and_grow done in {time.time()-_t1:.1f}s")

log("\n" + "="*60)
log("ALL CYCLES COMPLETE")
log("="*60)

