"""retrain_master.py — retrain V31Model from full master_corpus.jsonl"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from stage2.training_engine.neural_model import V31Model
from stage2.training_engine.asymmetric_trainer import AsymmetricTrainer
from stage2.training_engine.state_registry import AuraCycleManager

MASTER = Path("output/master_corpus.jsonl")
CKPT   = Path("models/_cycle_grow_ckpt.pt")
EPOCHS = 3
BATCH  = 32
LR     = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model   = V31Model().to(device)
trainer = AsymmetricTrainer(model, lr=LR)

existing = Path("models/v31_neural_model.pt")
if existing.exists():
    trainer.load_checkpoint(str(existing))
    print(f"Loaded checkpoint step={trainer.step_count}")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    m  = trainer.train_epoch(str(MASTER), batch_size=BATCH)
    print(
        f"Epoch {epoch}/{EPOCHS}  records={m['records']}  "
        f"loss={m['loss']:.5f}  mse={m['mse']:.5f}  ce={m['ce']:.5f}  "
        f"{round(time.time()-t0,1)}s"
    )

trainer.save_checkpoint(str(CKPT))
ch = AuraCycleManager.calculate_corpus_hash(str(MASTER))
sh = AuraCycleManager.compute_state_hash(ch, EPOCHS, BATCH, LR)
AuraCycleManager.register_state(
    sh, str(MASTER), ch, EPOCHS, BATCH, LR, m,
    str(CKPT.relative_to(Path(".")))
)
print(f"Registered {sh[:16]} -> models/v31_neural_model.pt updated")
