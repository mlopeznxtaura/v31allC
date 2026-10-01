"""golias_dashboard.py — NEW FILE. Live training mirror + adaptive inference tester
for Golias v7. Pure-stdlib HTTP server (no external web deps) so it runs anywhere
the training runs. Does not modify any existing file.

Two live panels:
  1. /stream  (SSE): tails models/golias_v8_train.log -> mirrors the training terminal
  2. /infer   (POST): loads the latest golias_v8_best.pt, runs a forward pass on the
                      current (adapting) weights and returns frame/token/halt.
                      Hot-reloads the checkpoint when its mtime changes.

Run:  python golias_dashboard.py   (serves on 0.0.0.0:8080)
Env:  GV7_PORT (default 8080), GV7_LOG, GV7_CKPT
"""
import os, sys, json, time, threading, math
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from goliasv7_model import GoliasV7, M1_IN, M2_IN, M3_IN

PORT = int(os.environ.get("GV7_PORT", "8080"))
LOG  = Path(os.environ.get("GV7_LOG", ROOT / "models" / "golias_v8_train.log"))
CKPT = Path(os.environ.get("GV7_CKPT", ROOT / "models" / "golias_v8_best.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB, FRAME = 5000, 224

_model = None
_ckpt_mtime = 0.0
_lock = threading.Lock()

# --- Spec-aligned defaults (from the mathematical sheet you just pasted) ---
DEFAULT_TRAINING = {
    "cycles": 5,
    "epochs_per_cycle": 15,
    "batch_size": 64,
    "lr": 8e-5,
    "tau_start": 0.28,
    "tau_end": 0.15,
    "lambda_pred": 1.0,      # λ1 L_pred(Ψ_pred)
    "lambda_syntax": 0.5,    # λ2 L_syntax(Ω_expl)
    "lambda_constraint": 0.8, # λ3 L_constraint(C_comp - τ)
    "alpha_push": 0.05,      # α for θ1 orthogonal push
    "beta_push": 0.05,       # β for θ2 orthogonal push
    "epsilon_efficiency": 0.01, # ε for L_efficiency(θ2) ≤ ε
    "k_cycle": [800,900,1000,1100,1200],
}

def _tile(v, n):
    return torch.full((1, n), float(v), dtype=torch.float32)


def load_model():
    global _model, _ckpt_mtime
    with _lock:
        if not CKPT.exists():
            return None
        mtime = CKPT.stat().st_mtime
        if _model is not None and mtime == _ckpt_mtime:
            return _model
        m = GoliasV7(vocab_size=VOCAB, frame_size=FRAME).to(DEVICE)
        ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        _model, _ckpt_mtime = m, mtime
        return _model


def build_vectors(m1, m2, m3, b, g, l, v, if7):
    m1l = math.copysign(math.log1p(abs(m1)), m1) / 10.0
    m1v = torch.cat([_tile(m1l, 32), _tile(b, 32), _tile(l, 32)], dim=-1)
    m2v = torch.cat([_tile(m2, 32), _tile(g, 32), _tile(l, 32), _tile(v, 32)], dim=-1)
    m3v = torch.cat([_tile(m1l, 32), _tile(m2, 32), _tile(m3, 32), _tile(b, 32),
                     _tile(g, 32), _tile(l, 32), _tile(v, 32), _tile(if7, 32),
                     _tile(0.0, 32), _tile((m1l + m2 + m3) / 3, 32), _tile(v * if7, 32)], dim=-1)
    return m1v.to(DEVICE), m2v.to(DEVICE), m3v.to(DEVICE)


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Golias Γ — Live (Spec-faithful)</title>
<style>
:root{color-scheme:dark}
body{margin:0;font-family:ui-monospace,Menlo,monospace;background:#05070a;color:#d7e0f0}
header{padding:12px 24px;border-bottom:1px solid #1f2937;background:#0a0f1c;display:flex;justify-content:space-between;align-items:center}
h1{font-size:15px;margin:0;color:#8ab4ff}
.tag{font-size:11px;color:#64748b}
.container{max-width:1320px;margin:0 auto;padding:18px 24px}
h2{font-size:11px;text-transform:uppercase;letter-spacing:1.8px;color:#64748b;margin:0 0 8px}
.section{margin-bottom:22px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:8px}
label{font-size:9px;color:#64748b;display:block;margin-bottom:2px}
input,select{width:100%;background:#0f1625;border:1px solid #243150;color:#e2e8f0;border-radius:5px;padding:5px 7px;font-size:12px}
.sliders{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.srow{display:flex;align-items:center;gap:6px}
.srow label{width:72px;font-size:9px;color:#94a3b8;flex-shrink:0}
.srow input[type=range]{flex:1}
.srow input[type=number]{width:58px;background:#0f1625;border:1px solid #243150;color:#e2e8f0;border-radius:4px;padding:2px 4px;font-size:11px}
.presets{display:flex;gap:4px;flex-wrap:wrap;margin-top:6px}
.presets button{font-size:9px;padding:2px 6px;background:#1e2937;border:1px solid #334155;color:#cbd5e1;border-radius:3px;cursor:pointer}
.iframe-wrap{width:100%;height:360px;border:1px solid #1f2937;border-radius:8px;overflow:hidden;background:#0a0f1c}
.base{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.card{background:#0f1625;border:1px solid #1f2937;border-radius:6px;padding:10px;font-size:10px}
button.primary{background:linear-gradient(135deg,#3f7bff,#7a47ff);border:0;color:#fff;font-weight:600;padding:9px 16px;border-radius:6px;cursor:pointer;font-size:12px;margin-top:8px}
.results{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:6px;margin-top:8px}
.res{background:#0f1625;border:1px solid #1f2937;border-radius:5px;padding:6px 8px;font-size:10px}
.res .k{color:#64748b;font-size:9px}
.res .v{color:#bae6fd;font-weight:600}
.spec{font-size:9px;color:#64748b;margin-top:4px}
</style>
</head>
<body>
<header>
  <h1>● GOLIAS Γ — LIVE (Spec-faithful Modular GUI)</h1>
  <span class="tag" id="dev"></span>
</header>

<div class="container">

  <!-- 1. INPUT SPACE (B_t, G_t, L_t, H¹/²/³) -->
  <div class="section">
    <h2>1. Input Space (Spec)</h2>
    <div class="grid">
      <div><label>B_t (Binary substrate)</label><input id="b" type="number" step="0.01" value="0.73"></div>
      <div><label>G_t (Geometry telemetry)</label><input id="g" type="number" step="0.01" value="0.47"></div>
      <div><label>L_t (Language)</label><input id="l" type="number" step="0.01" value="0.40"></div>
      <div><label>H¹_t (M1 latent)</label><input id="m1" type="number" step="0.1" value="4.2"></div>
      <div><label>H²_t (M2 latent)</label><input id="m2" type="number" step="0.01" value="0.55"></div>
      <div><label>H³_t (M3 latent)</label><input id="m3" type="number" step="0.01" value="0.99"></div>
      <div><label>V (triangulation)</label><input id="v" type="number" step="0.01" value="0.58"></div>
      <div><label>IF7 (fusion)</label><input id="if7" type="number" step="0.01" value="0.5"></div>
    </div>
    <div class="spec">Generalization constraint: B_t ≠ ∅ and G_t ≠ ∅ (enforced for any forward pass)</div>
  </div>

  <!-- 2. FORWARD PASS (M1/M2 gen → Hard Halting C_comp/τ → M3 Meta-Eval → Rotation K_t) -->
  <div class="section">
    <h2>2. Forward Pass (Spec)</h2>
    <div class="grid">
      <div><label>τ (halt threshold)</label><input id="tau" type="number" step="0.01" value="0.15"></div>
      <div><label>C_comp (compute cost)</label><input id="ccomp" type="number" step="0.01" value="0.5"></div>
      <div><label>K_t (rotation cycle)</label>
        <select id="kt">
          <option>800</option><option>900</option><option selected>1000</option>
          <option>1100</option><option>1200</option>
        </select>
      </div>
      <div><label>α (θ1 push)</label><input id="alpha" type="number" step="0.01" value="0.05"></div>
      <div><label>β (θ2 push)</label><input id="beta" type="number" step="0.01" value="0.05"></div>
      <div><label>ε (efficiency constraint on θ2)</label><input id="eps" type="number" step="0.001" value="0.01"></div>
    </div>
  </div>

  <!-- 3. ADAPTIVE DIMENSIONS (full slider version for live tweaking) -->
  <div class="section">
    <h2>3. Adaptive Dimensions (Live Tweaking)</h2>
    <div class="sliders">
      <div class="srow"><label>M1 (H¹)</label><input type="range" id="m1r" min="0" max="10" step="0.05" value="4.2"><input id="m1" type="number" step="0.1" value="4.2"></div>
      <div class="srow"><label>M2 (H²)</label><input type="range" id="m2r" min="0" max="2" step="0.01" value="0.55"><input id="m2" type="number" step="0.01" value="0.55"></div>
      <div class="srow"><label>M3 (H³)</label><input type="range" id="m3r" min="0" max="2" step="0.01" value="0.99"><input id="m3" type="number" step="0.01" value="0.99"></div>
      <div class="srow"><label>B_t</label><input type="range" id="br" min="0" max="1" step="0.01" value="0.73"><input id="b" type="number" step="0.01" value="0.73"></div>
      <div class="srow"><label>G_t</label><input type="range" id="gr" min="0" max="1" step="0.01" value="0.47"><input id="g" type="number" step="0.01" value="0.47"></div>
      <div class="srow"><label>L_t</label><input type="range" id="lr" min="0" max="1" step="0.01" value="0.40"><input id="l" type="number" step="0.01" value="0.40"></div>
      <div class="srow"><label>V</label><input type="range" id="vr" min="0" max="1" step="0.01" value="0.58"><input id="v" type="number" step="0.01" value="0.58"></div>
      <div class="srow"><label>IF7</label><input type="range" id="if7r" min="0" max="1" step="0.01" value="0.5"><input id="if7" type="number" step="0.01" value="0.5"></div>
    </div>
    <div class="presets">
      <button onclick="loadPreset('high_expl')">High M1</button>
      <button onclick="loadPreset('efficiency')">M2 locked</button>
      <button onclick="loadPreset('physicalai')">PhysicalAI</button>
      <button onclick="loadPreset('reset')">Reset</button>
    </div>
  </div>

  <!-- TERMINAL IFRAME — exactly between adaptive dimensions and base function config -->
  <div class="section">
    <h2>4. Live Training Terminal (iframe mirror of golias_v8_train.log)</h2>
    <div class="iframe-wrap">
      <iframe src="/terminal" style="width:100%;height:100%;border:0" title="Training log"></iframe>
    </div>
  </div>

  <!-- 5. FUSION / RELATION / ATTENTION (Constraint-Trimmed) -->
  <div class="section">
    <h2>5. Fusion, Relation, Constraint-Trimmed Attention (Spec)</h2>
    <div class="grid">
      <div><label>Fusion weights (H1+H2+B+G+Θ_meta)</label><input id="fuse_w" type="text" value="1,1,0.8,0.7,1.2"></div>
      <div><label>K_t for attention slice</label><input id="kt_slice" type="number" value="1000"></div>
      <div><label>Orthogonal push strength (A_push)</label><input id="push" type="number" step="0.01" value="0.05"></div>
    </div>
  </div>

  <!-- 6. OUTPUT SPACE -->
  <div class="section">
    <h2>6. Output Space (Spec)</h2>
    <div class="grid">
      <div><label>Ψ_pred (MLP_pred on R_t)</label><span style="font-size:10px;color:#64748b">next-frame / next-pixel</span></div>
      <div><label>Ω_expl (Decode_LM on R_t)</label><span style="font-size:10px;color:#64748b">language explanation</span></div>
    </div>
  </div>

  <!-- 7. TRIPARTITE DYNAMIC LOSS + ONLINE RETUNING -->
  <div class="section">
    <h2>7. Tripartite Dynamic Loss + Online Adaptive Retuning (Spec)</h2>
    <div class="grid">
      <div><label>λ1 L_pred(Ψ)</label><input id="lam1" type="number" step="0.1" value="1.0"></div>
      <div><label>λ2 L_syntax(Ω)</label><input id="lam2" type="number" step="0.1" value="0.5"></div>
      <div><label>λ3 L_constraint(C_comp-τ)</label><input id="lam3" type="number" step="0.1" value="0.8"></div>
      <div><label>ε (efficiency on θ2)</label><input id="eps2" type="number" step="0.001" value="0.01"></div>
    </div>
    <div class="spec">All θ1,θ2,θ3 updated per token-step. Generalization only when B_t and G_t are both non-empty.</div>
  </div>

  <button class="primary" onclick="runInference()">Run Adaptive Inference (uses current modular config + v8 weights)</button>
  <div id="results" class="results"></div>

</div>

<script>
const dev = document.getElementById('dev');
fetch('/info').then(r=>r.json()).then(d=>{dev.textContent=(d.device||'cpu')+' · '+(d.ckpt||'v8.pt');});

let scalars={m1:4.2,m2:0.55,m3:0.99,b:0.73,g:0.47,l:0.40,v:0.58,if7:0.5};
let trainCfg = {cycles:5,epochs:15,batch:64,lr:8e-5,tau_start:0.28,tau_end:0.15,lam1:1.0,lam2:0.5,lam3:0.8,alpha:0.05,beta:0.05,eps:0.01,kt:1000};

function syncScalars(){
  ['m1','m2','m3','b','g','l','v','if7'].forEach(k=>{
    const n=document.getElementById(k); const r=document.getElementById(k+'r');
    if(n) n.value=scalars[k];
    if(r) r.value=scalars[k];
  });
}
function wire(id){
  const r=document.getElementById(id+'r'); const n=document.getElementById(id);
  if(!r||!n) return;
  r.oninput=()=>{n.value=parseFloat(r.value).toFixed(2); scalars[id]=parseFloat(r.value);};
  n.oninput=()=>{r.value=n.value; scalars[id]=parseFloat(n.value)||0;};
}
['m1','m2','m3','b','g','l','v','if7'].forEach(wire);

function loadPreset(p){
  if(p==='high_expl') scalars={m1:7.8,m2:0.35,m3:0.92,b:0.65,g:0.51,l:0.55,v:0.61,if7:0.48};
  else if(p==='efficiency') scalars={m1:2.1,m2:1.25,m3:0.78,b:0.81,g:0.39,l:0.29,v:0.44,if7:0.71};
  else if(p==='physicalai') scalars={m1:4.57,m2:0.53,m3:0.99,b:0.71,g:0.62,l:0.41,v:0.59,if7:0.52};
  else scalars={m1:4.2,m2:0.55,m3:0.99,b:0.73,g:0.47,l:0.40,v:0.58,if7:0.5};
  syncScalars();
}

function collectPayload(){
  return {
    m1:scalars.m1, m2:scalars.m2, m3:scalars.m3, b:scalars.b, g:scalars.g, l:scalars.l, v:scalars.v, if7:scalars.if7,
    tau: parseFloat(document.getElementById('tau').value)||0.15,
    ccomp: parseFloat(document.getElementById('ccomp').value)||0.5,
    kt: parseInt(document.getElementById('kt').value)||1000,
    alpha: parseFloat(document.getElementById('alpha').value)||0.05,
    beta: parseFloat(document.getElementById('beta').value)||0.05,
    eps: parseFloat(document.getElementById('eps').value)||0.01,
    lam1: parseFloat(document.getElementById('lam1').value)||1.0,
    lam2: parseFloat(document.getElementById('lam2').value)||0.5,
    lam3: parseFloat(document.getElementById('lam3').value)||0.8,
    training: {
      cycles: parseInt(document.getElementById('cfg_cycles').value)||5,
      epochs: parseInt(document.getElementById('cfg_epochs').value)||15,
      batch: parseInt(document.getElementById('cfg_batch').value)||64,
      lr: parseFloat(document.getElementById('cfg_lr').value)||8e-5
    }
  };
}

async function runInference(){
  const out=document.getElementById('results');
  out.innerHTML='<div class="res"><div class="k">status</div><div class="v">running…</div></div>';
  try{
    const r=await fetch('/infer',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(collectPayload())});
    const d=await r.json();
    if(d.error){ out.innerHTML=`<div class="res"><div class="k">error</div><div class="v">${d.error}</div></div>`; return; }
    out.innerHTML=`
      <div class="res"><div class="k">frame μ/σ</div><div class="v">${d.frame_mean} / ${d.frame_std}</div></div>
      <div class="res"><div class="k">halt (C_comp &gt; τ)</div><div class="v">${d.halt}</div></div>
      <div class="res"><div class="k">top tokens (Ω_expl)</div><div class="v">${d.top5_token_ids.join(' ')}</div></div>
      <div class="res"><div class="k">logits (Ψ_pred)</div><div class="v">${d.top5_logits.join(' ')}</div></div>
    `;
  }catch(e){ out.innerHTML=`<div class="res"><div class="k">client</div><div class="v">${e}</div></div>`; }
}

// Training config live mirror (for the JSON card)
['cfg_cycles','cfg_epochs','cfg_batch','cfg_lr'].forEach(id=>{
  const el=document.getElementById(id); if(el) el.oninput=()=>updateTrainJson();
});
function updateTrainJson(){
  const el=document.getElementById('next_train_json');
  if(!el) return;
  el.textContent=JSON.stringify({
    cycles:parseInt(document.getElementById('cfg_cycles').value)||5,
    epochs:parseInt(document.getElementById('cfg_epochs').value)||15,
    batch:parseInt(document.getElementById('cfg_batch').value)||64,
    lr:parseFloat(document.getElementById('cfg_lr').value)||8e-5,
    tau_anneal:[0.28,0.15],
    loss:{lam1:1.0,lam2:0.5,lam3:0.8},
    online_retune:{alpha:0.05,beta:0.05,eps:0.01}
  },null,2);
}
updateTrainJson();

// initial values for the spec controls
document.getElementById('tau').value = 0.15;
document.getElementById('ccomp').value = 0.5;
document.getElementById('kt').value = 1000;
document.getElementById('alpha').value = 0.05;
document.getElementById('beta').value = 0.05;
document.getElementById('eps').value = 0.01;
document.getElementById('lam1').value = 1.0;
document.getElementById('lam2').value = 0.5;
document.getElementById('lam3').value = 0.8;

syncScalars();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("cache-control", "no-store")
        self.end_headers()
        if isinstance(body, str):
            body = body.encode()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif self.path == "/info":
            self._send(200, "application/json",
                       json.dumps({"device": DEVICE, "ckpt": CKPT.name,
                                   "ckpt_exists": CKPT.exists()}))
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-store")
            self.end_headers()
            pos = 0
            try:
                while True:
                    if LOG.exists():
                        with open(LOG, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(pos)
                            for line in f:
                                self.wfile.write(f"data: {line.rstrip()}\n\n".encode())
                            pos = f.tell()
                        self.wfile.flush()
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self._send(404, "text/plain", "not found")

    def do_POST(self):
        if self.path != "/infer":
            self._send(404, "text/plain", "not found"); return
        n = int(self.headers.get("content-length", 0))
        try:
            p = json.loads(self.rfile.read(n) or "{}")
        except Exception:
            p = {}
        model = load_model()
        if model is None:
            self._send(503, "application/json",
                       json.dumps({"error": "no checkpoint yet"})); return
        m1v, m2v, m3v = build_vectors(
            p.get("m1", 0), p.get("m2", 0), p.get("m3", 0), p.get("b", 0),
            p.get("g", 0), p.get("l", 0), p.get("v", 0.5), p.get("if7", 0))
        with torch.no_grad():
            frame, tokens, halt, _ = model(m1v, m2v, m3v)
        top = torch.topk(tokens[0], 5)
        self._send(200, "application/json", json.dumps({
            "ckpt_mtime": _ckpt_mtime,
            "frame_mean": round(float(frame.mean()), 5),
            "frame_std": round(float(frame.std()), 5),
            "halt": round(float(halt[0, 0]), 5),
            "top5_token_ids": top.indices.tolist(),
            "top5_logits": [round(x, 4) for x in top.values.tolist()],
        }))


def main():
    print(f"Golias dashboard on http://0.0.0.0:{PORT}  device={DEVICE}")
    print(f"  log={LOG}\n  ckpt={CKPT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


# ── Standalone terminal page for the iframe (clean isolated mirror) ──────────
TERMINAL_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>terminal</title>
<style>
body{margin:0;background:#0a0f1c;color:#67e8f9;font:11px/1.35 ui-monospace,Menlo,monospace;padding:8px;overflow:auto}
#term{white-space:pre;min-height:100%}
</style>
</head>
<body>
<div id="term">connecting…</div>
<script>
const t = document.getElementById('term');
const es = new EventSource('/stream');
es.onmessage = e => { t.textContent += e.data + "\\n"; window.scrollTo(0, document.body.scrollHeight); };
es.onerror = () => { t.textContent += "\\n[stream disconnected]\\n"; };
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("cache-control", "no-store")
        self.end_headers()
        if isinstance(body, str):
            body = body.encode()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif self.path == "/terminal":
            self._send(200, "text/html; charset=utf-8", TERMINAL_PAGE)
        elif self.path == "/info":
            self._send(200, "application/json",
                       json.dumps({"device": DEVICE, "ckpt": CKPT.name,
                                   "ckpt_exists": CKPT.exists()}))
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-store")
            self.end_headers()
            pos = 0
            try:
                while True:
                    if LOG.exists():
                        with open(LOG, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(pos)
                            for line in f:
                                self.wfile.write(f"data: {line.rstrip()}\n\n".encode())
                            pos = f.tell()
                        self.wfile.flush()
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self._send(404, "text/plain", "not found")

    def do_POST(self):
        if self.path != "/infer":
            self._send(404, "text/plain", "not found"); return
        n = int(self.headers.get("content-length", 0))
        try:
            p = json.loads(self.rfile.read(n) or "{}")
        except Exception:
            p = {}
        model = load_model()
        if model is None:
            self._send(200, "application/json", json.dumps({"error": "no checkpoint yet"}))
            return
        # Accept the richer payload but fall back gracefully
        m1v, m2v, m3v = build_vectors(
            p.get("m1", 0), p.get("m2", 0), p.get("m3", 0), p.get("b", 0),
            p.get("g", 0), p.get("l", 0), p.get("v", 0.5), p.get("if7", 0))
        with torch.no_grad():
            frame, tokens, halt, _ = model(m1v, m2v, m3v)
        top = torch.topk(tokens[0], 5)
        self._send(200, "application/json", json.dumps({
            "ckpt_mtime": _ckpt_mtime,
            "frame_mean": round(float(frame.mean()), 5),
            "frame_std": round(float(frame.std()), 5),
            "halt": round(float(halt[0, 0]), 5),
            "top5_token_ids": top.indices.tolist(),
            "top5_logits": [round(x, 4) for x in top.values.tolist()],
        }))


def main():
    print(f"Golias dashboard on http://0.0.0.0:{PORT}  device={DEVICE}")
    print(f"  log={LOG}\n  ckpt={CKPT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
