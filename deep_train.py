"""deep_train.py — NextAura v31allC deep training cycle
Trains V31Model v2 (31.5M params) from scratch on full master_corpus.jsonl.
Schedule: linear warmup 1 epoch → cosine decay over remaining epochs.
Runs CYCLES passes — each cycle re-reads the (growing) master corpus.
After each cycle: register checkpoint in state_registry + print full metrics.
"""

import sys, time, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from stage2.training_engine.neural_model import V31Model, count_params
from stage2.training_engine.asymmetric_trainer import AsymmetricTrainer, get_deterministic_token_id, decode_and_pool_frame
from stage2.training_engine.state_registry import AuraCycleManager

MASTER     = ROOT / "output" / "master_corpus.jsonl"
MODELS_DIR = ROOT / "models"
CKPT_ACTIVE= MODELS_DIR / "v31_neural_model.pt"

# ── Hyperparams ───────────────────────────────────────────────────────────────
CYCLES     = 5      # full corpus passes (each one re-reads growing corpus)
EPOCHS     = 15     # gradient epochs per cycle
BATCH      = 32
LR         = 1e-4
WEIGHT_DECAY = 5e-5
WARMUP_EPOCHS = 1
GRAD_CLIP  = 1.0


def make_scheduler(optimizer, total_epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def load_frames(corpus_path):
    """Load corpus into frame dicts (reuse AsymmetricTrainer logic)."""
    frames = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec  = json.loads(line)
                meta = rec.get("_meta", {})
                si   = rec.get("scalar_inputs", {})
                ot   = rec.get("output_tokens", {})
                frames.append({
                    "m1":  float(meta.get("M1_raw", 1.0)),
                    "m2":  float(meta.get("M2_raw", 1.0)),
                    "m3":  float(meta.get("M3_raw", 1.0)),
                    "geo": torch.ones(32)  * float(meta.get("geometry_scalar", 0.0)),
                    "bin": torch.ones(32)  * float(meta.get("binary_scalar",   0.0)),
                    "lng": torch.ones(128) * float(meta.get("language_scalar", 0.0)),
                    "tri": torch.ones(32)  * float(meta.get("V", 0.5)),
                    "next_frame_target": decode_and_pool_frame(meta.get("pixel_frame_base64", "")),
                    "next_token_id":     get_deterministic_token_id(ot.get("language_token_output", "")),
                })
            except Exception:
                continue
    return frames


def train_one_epoch(model, frames, optimizer, criterion_mse, criterion_ce, device, batch_size):
    model.train()
    total_loss = total_mse = total_ce = 0.0
    batches = 0
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        m1_b  = torch.tensor([f["m1"] for f in batch], dtype=torch.float32).to(device)
        m2_b  = torch.tensor([f["m2"] for f in batch], dtype=torch.float32).to(device)
        m3_b  = torch.tensor([f["m3"] for f in batch], dtype=torch.float32).to(device)
        geo_b = torch.stack([f["geo"] for f in batch]).to(device)
        bin_b = torch.stack([f["bin"] for f in batch]).to(device)
        lng_b = torch.stack([f["lng"] for f in batch]).to(device)
        tri_b = torch.stack([f["tri"] for f in batch]).to(device)
        tgt_f = torch.stack([f["next_frame_target"] for f in batch]).to(device)
        tgt_t = torch.tensor([f["next_token_id"] for f in batch], dtype=torch.long).to(device)

        optimizer.zero_grad()
        pred_f, pred_t = model(m1_b, m2_b, m3_b, geo_b, bin_b, lng_b, tri_b)

        loss_mse = criterion_mse(pred_f, tgt_f)
        loss_ce  = criterion_ce(pred_t, tgt_t)
        loss     = loss_mse + 0.1 * loss_ce

        if torch.isnan(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        total_mse  += loss_mse.item()
        total_ce   += loss_ce.item()
        batches    += 1

    n = max(batches, 1)
    return total_loss/n, total_mse/n, total_ce/n


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print("V31allC DEEP TRAINING")
    print(f"  Device : {device}  |  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Master : {MASTER.name}")

    model = V31Model().to(device)
    print(f"  Model  : {count_params(model)}")

    # Load active checkpoint if exists (continue learning)
    if CKPT_ACTIVE.exists():
        try:
            ckpt = torch.load(str(CKPT_ACTIVE), map_location=device)
            # Handle both raw state_dict and wrapped checkpoint
            sd = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"  Checkpoint loaded: {CKPT_ACTIVE.name}  missing={len(missing)}  unexpected={len(unexpected)}")
        except Exception as ex:
            print(f"  Checkpoint load failed ({ex}), starting fresh.")

    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion_mse = nn.MSELoss()
    criterion_ce  = nn.CrossEntropyLoss()

    print("=" * 72)

    for cycle in range(1, CYCLES + 1):
        print(f"\n{'─'*72}")
        print(f"  CYCLE {cycle}/{CYCLES}  — loading frames …")
        frames = load_frames(MASTER)
        if not frames:
            print("  No frames — corpus empty. Abort.")
            return
        print(f"  Frames: {len(frames):,}  |  Epochs: {EPOCHS}  |  Batches/epoch: {len(frames)//BATCH}")

        scheduler = make_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)

        best_loss  = float("inf")
        best_ckpt  = MODELS_DIR / f"v31_cycle{cycle}_best.pt"

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            loss, mse, ce = train_one_epoch(
                model, frames, optimizer, criterion_mse, criterion_ce, device, BATCH
            )
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
            elapsed = round(time.time() - t0, 2)

            marker = ""
            if loss < best_loss:
                best_loss = loss
                torch.save({"model_state_dict": model.state_dict()}, str(best_ckpt))
                marker = " ★"

            print(
                f"  C{cycle} E{epoch:02d}/{EPOCHS}  "
                f"loss={loss:.5f}  mse={mse:.5f}  ce={ce:.5f}  "
                f"lr={lr_now:.2e}  {elapsed}s{marker}"
            )

        # End of cycle: promote best checkpoint to active
        import shutil
        shutil.copy(str(best_ckpt), str(CKPT_ACTIVE))

        # Register in state_registry
        ch = AuraCycleManager.calculate_corpus_hash(str(MASTER))
        sh = AuraCycleManager.compute_state_hash(ch, EPOCHS * cycle, BATCH, LR)
        metrics = {"loss": round(best_loss, 6), "mse": mse, "ce": ce, "records": len(frames), "cycle": cycle}
        rel_ckpt = str(best_ckpt.relative_to(ROOT))
        AuraCycleManager.register_state(sh, str(MASTER), ch, EPOCHS*cycle, BATCH, LR, metrics, rel_ckpt)
        print(f"  ✓ Cycle {cycle} done  best_loss={best_loss:.5f}  registered {sh[:16]}")

    print("\n" + "=" * 72)
    print("DEEP TRAINING COMPLETE  →  models/v31_neural_model.pt updated")
    print("=" * 72)


if __name__ == "__main__":
    main()
