import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2.training_engine.asymmetric_trainer import AsymmetricTrainer
from stage2.training_engine.neural_model import V31Model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model = V31Model().to(device)
trainer = AsymmetricTrainer(model, lr=3e-4)

# Let's inspect the weights before training
print("Weight stats before training:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}, max={param.max().item():.4f}")

frames = trainer.load_corpus_data("output/master_corpus.jsonl")
print(f"Loaded {len(frames)} frames")

batch_size = 32
model.train()

for batch_idx in range(10):
    batch = frames[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    m1_b = torch.tensor([f["m1"] for f in batch], dtype=torch.float32).to(device)
    m2_b = torch.tensor([f["m2"] for f in batch], dtype=torch.float32).to(device)
    m3_b = torch.tensor([f["m3"] for f in batch], dtype=torch.float32).to(device)
    geo_b = torch.stack([f["geo"] for f in batch]).to(device)
    bin_b = torch.stack([f["bin"] for f in batch]).to(device)
    lng_b = torch.stack([f["lng"] for f in batch]).to(device)
    tri_b = torch.stack([f["tri"] for f in batch]).to(device)

    target_frame_b = torch.stack([f["next_frame_target"] for f in batch]).to(device)
    target_token_b = torch.tensor([f["next_token_id"] for f in batch], dtype=torch.long).to(device)

    trainer.optimizer.zero_grad()
    pred_frame, pred_token_logits = model(m1_b, m2_b, m3_b, geo_b, bin_b, lng_b, tri_b)
    
    loss_mse = trainer.criterion_mse(pred_frame, target_frame_b)
    loss_ce = trainer.criterion_ce(pred_token_logits, target_token_b)
    loss = loss_mse + 0.1 * loss_ce
    
    print(f"Batch {batch_idx}: loss={loss.item():.4f} mse={loss_mse.item():.4f} ce={loss_ce.item():.4f}")
    
    if torch.isnan(loss):
        print("Loss is NaN!")
        break
        
    loss.backward()
    
    # Print gradient norms before clipping
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"  Gradient norm before clipping: {total_norm:.4f}")
    
    # Clip grads
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Check if any grad is NaN
    nan_grads = False
    for p in model.parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            nan_grads = True
            break
    if nan_grads:
        print("  Found NaN in gradients!")
        break
        
    trainer.optimizer.step()
    
    # Check if any weight is NaN
    nan_weights = False
    for p in model.parameters():
        if torch.isnan(p).any():
            nan_weights = True
            break
    if nan_weights:
        print("  Found NaN in weights after step!")
        break
