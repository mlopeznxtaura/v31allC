"""deep_train_goliasv7.py — NEW FILE. Trains Golias-Γ v7 (warm-started from
golias v6.pt) on master_corpus.jsonl and writes v8 checkpoints.

Does NOT modify any existing training script. Reuses the existing corpus and
the existing scalar fields already present in master_corpus.jsonl.

Observer input vectors (built from each corpus record's scalars):
  M1 vec (96)  : exploration/creative channel  — m1_raw + binary + language scalars
  M2 vec (128) : efficiency channel            — m2_raw + geometry + language scalars
  M3 vec (352) : meta channel                  — m3_raw + all scalars + V + if7

Targets:
  frame  : decoded/pooled next-frame proxy (from V + scalars; corpus has no raw pixels)
  token  : deterministic id from language_token_output

Output: models/golias_v8_*.pt  (+ golias_v8_best.pt + registry entry)
"""

import sys, os, json, time, math, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from stage2.training_engine.goliasv7_model import (
    GoliasV7, load_v6_warmstart, count_params, M1_IN, M2_IN, M3_IN,
)

MASTER     = ROOT / "output" / "master_corpus.jsonl"
MODELS_DIR = ROOT / "models"
V6_PATH    = r"E:\NextAuraMonth3(corefocuses)\RAWgolias\golias v6.pt"

CYCLES = int(os.environ.get("GV7_CYCLES", "5"))
EPOCHS = int(os.environ.get("GV7_EPOCHS", "15"))
BATCH  = int(os.environ.get("GV7_BATCH", "64"))
LR     = float(os.environ.get("GV7_LR", "8e-5"))
WEIGHT_DECAY = 5e-5
WARMUP_EPOCHS = 1
GRAD_CLIP = 1.0
VOCAB = 5000
FRAME = 224
SMOKE = "--smoke" in sys.argv


def det_token_id(s: str, vocab: int = VOCAB) -> int:
    if not s:
        return 0
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % vocab


def tile(val, n):
    return torch.full((n,), float(val), dtype=torch.float32)


def build_vectors(meta, si):
    """Construct (m1_vec, m2_vec, m3_vec, frame_target, token_id) from a record."""
    m1 = float(meta.get("M1_raw", 1.0))
    m2 = float(meta.get("M2_raw", 1.0))
    m3 = float(meta.get("M3_raw", 1.0))
    b  = float(meta.get("binary_scalar", 0.0))
    g  = float(meta.get("geometry_scalar", 0.0))
    l  = float(meta.get("language_scalar", 0.0))
    v  = float(meta.get("V", 0.5))
    if7 = float(meta.get("if7_policy_fusion", 0.0))
    orth = float(meta.get("orthogonal_vector", 0.0)) if isinstance(meta.get("orthogonal_vector", 0.0), (int, float)) else 0.0

    # M1 explorer: log-compressed m1 + binary + language texture
    m1_log = math.copysign(math.log1p(abs(m1)), m1) / 10.0
    m1_vec = torch.cat([tile(m1_log, 32), tile(b, 32), tile(l, 32)])           # 96

    # M2 efficiency: m2 + geometry + language + V
    m2_vec = torch.cat([tile(m2, 32), tile(g, 32), tile(l, 32), tile(v, 32)])  # 128

    # M3 meta: sees everything
    m3_vec = torch.cat([
        tile(m1_log, 32), tile(m2, 32), tile(m3, 32), tile(b, 32),
        tile(g, 32), tile(l, 32), tile(v, 32), tile(if7, 32),
        tile(orth, 32), tile((m1_log + m2 + m3) / 3.0, 32), tile(v * if7, 32),
    ])                                                                          # 352

    frame_target = tile(v, FRAME) + 0.01 * tile(g, FRAME)
    return m1_vec, m2_vec, m3_vec, frame_target


def load_frames(corpus_path, limit=None):
    frames = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                meta = rec.get("_meta", {})
                si = rec.get("scalar_inputs", {})
                ot = rec.get("output_tokens", {})
                m1v, m2v, m3v, ft = build_vectors(meta, si)
                frames.append((m1v, m2v, m3v, ft,
                               det_token_id(ot.get("language_token_output", ""))))
            except Exception:
                continue
            if limit and len(frames) >= limit:
                break
    return frames


def make_scheduler(opt, total, warmup):
    def lr_lambda(epoch):
        if epoch < warmup:
            return float(epoch + 1) / float(max(1, warmup))
        prog = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return LambdaLR(opt, lr_lambda)


def train_epoch(model, frames, opt, mse, ce, device, bs):
    model.train()
    tot = tmse = tce = 0.0
    nb = 0
    for i in range(0, len(frames), bs):
        batch = frames[i:i + bs]
        m1b = torch.stack([f[0] for f in batch]).to(device)
        m2b = torch.stack([f[1] for f in batch]).to(device)
        m3b = torch.stack([f[2] for f in batch]).to(device)
        ftb = torch.stack([f[3] for f in batch]).to(device)
        ttb = torch.tensor([f[4] for f in batch], dtype=torch.long).to(device)

        opt.zero_grad()
        frame, tokens, halt, _ = model(m1b, m2b, m3b)
        loss_mse = mse(frame, ftb)
        loss_ce = ce(tokens, ttb)
        loss = loss_mse + 0.1 * loss_ce
        if torch.isnan(loss):
            opt.zero_grad()
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        tot += loss.item(); tmse += loss_mse.item(); tce += loss_ce.item(); nb += 1
    nb = max(nb, 1)
    return tot / nb, tmse / nb, tce / nb


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"GOLIAS v7 TRAINING (warm-start v6 -> output v8)  device={device}")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = GoliasV7(vocab_size=VOCAB, frame_size=FRAME).to(device)
    rep = load_v6_warmstart(model, V6_PATH)
    print(f"  {count_params(model)}  | warm-start copied {rep['copied_tensors']} v6 tensors")

    limit = 200 if SMOKE else None
    frames = load_frames(MASTER, limit=limit)
    print(f"  Loaded {len(frames)} frames from master_corpus")
    if not frames:
        print("  No frames — abort.")
        return

    cycles = 1 if SMOKE else CYCLES
    epochs = 1 if SMOKE else EPOCHS

    mse = nn.MSELoss(); ce = nn.CrossEntropyLoss()
    best = float("inf")
    for c in range(1, cycles + 1):
        opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = make_scheduler(opt, epochs, WARMUP_EPOCHS)
        for e in range(1, epochs + 1):
            t0 = time.time()
            loss, lmse, lce = train_epoch(model, frames, opt, mse, ce, device, BATCH)
            sched.step()
            lr_now = sched.get_last_lr()[0]
            print(f"  C{c} E{e:02d}/{epochs}  loss={loss:.5f}  mse={lmse:.5f}  "
                  f"ce={lce:.5f}  lr={lr_now:.2e}  {time.time()-t0:.2f}s")
            if loss < best:
                best = loss
                torch.save({"state_dict": model.state_dict(),
                            "_metadata": {"arch": "Golias-v7", "parent": "Golias-v6",
                                          "warm_start": True, "best_loss": best},
                            "metrics": {"cycle": c, "epoch": e, "loss": loss}},
                           MODELS_DIR / "golias_v8_best.pt")
        torch.save({"state_dict": model.state_dict(),
                    "_metadata": {"arch": "Golias-v7", "cycle": c}},
                   MODELS_DIR / f"golias_v8_cycle{c}.pt")
        print(f"  [OK] Cycle {c} done  best={best:.5f}")

    print("=" * 70)
    print(f"DONE -> models/golias_v8_best.pt  best_loss={best:.5f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
