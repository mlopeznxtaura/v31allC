import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2.training_engine.asymmetric_trainer import AsymmetricTrainer
from stage2.training_engine.neural_model import V31Model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model = V31Model().to(device)
trainer = AsymmetricTrainer(model, lr=3e-4)
frames = trainer.load_corpus_data("output/master_corpus.jsonl")
print(f"Loaded {len(frames)} frames")

batch = frames[:32]
m1_b = torch.tensor([f["m1"] for f in batch], dtype=torch.float32).to(device)
m2_b = torch.tensor([f["m2"] for f in batch], dtype=torch.float32).to(device)
m3_b = torch.tensor([f["m3"] for f in batch], dtype=torch.float32).to(device)
geo_b = torch.stack([f["geo"] for f in batch]).to(device)
bin_b = torch.stack([f["bin"] for f in batch]).to(device)
lng_b = torch.stack([f["lng"] for f in batch]).to(device)
tri_b = torch.stack([f["tri"] for f in batch]).to(device)

target_frame_b = torch.stack([f["next_frame_target"] for f in batch]).to(device)
target_token_b = torch.tensor([f["next_token_id"] for f in batch], dtype=torch.long).to(device)

print("m1_b isnan:", torch.isnan(m1_b).any().item())
print("m2_b isnan:", torch.isnan(m2_b).any().item())
print("m3_b isnan:", torch.isnan(m3_b).any().item())
print("geo_b isnan:", torch.isnan(geo_b).any().item())
print("bin_b isnan:", torch.isnan(bin_b).any().item())
print("lng_b isnan:", torch.isnan(lng_b).any().item())
print("tri_b isnan:", torch.isnan(tri_b).any().item())

# Register hooks or print intermediate activations
# Let's run forward step by step manually or print layers:
x = torch.cat([m1_b.view(-1, 1), m2_b.view(-1, 1), m3_b.view(-1, 1), geo_b, bin_b, lng_b, tri_b], dim=-1)
print("x shape:", x.shape, "x isnan:", torch.isnan(x).any().item())

h = torch.nn.functional.relu(model.input_proj(x))
print("h isnan:", torch.isnan(h).any().item())

tau = torch.clamp(tri_b.mean(dim=-1, keepdim=True), 0.0, 1.0)
print("tau min/max/isnan:", tau.min().item(), tau.max().item(), torch.isnan(tau).any().item())

# inside moe.router
router_logits = model.moe.router.gate(h)
print("router_logits isnan:", torch.isnan(router_logits).any().item())
temp = 1.0 / (tau + 1e-6)
print("temp min/max/isnan:", temp.min().item(), temp.max().item(), torch.isnan(temp).any().item())
scaled_logits = router_logits * temp
print("scaled_logits isnan:", torch.isnan(scaled_logits).any().item())
probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
print("probs isnan:", torch.isnan(probs).any().item(), "probs values:", probs[0])

expert_outputs = torch.stack([expert(h) for expert in model.moe.experts], dim=1)
print("expert_outputs isnan:", torch.isnan(expert_outputs).any().item())

h_moe = torch.sum(probs.unsqueeze(-1) * expert_outputs, dim=1)
print("h_moe isnan:", torch.isnan(h_moe).any().item())

next_frame = model.next_frame_head(h_moe)
token_logits = model.token_head(h_moe)
print("next_frame isnan:", torch.isnan(next_frame).any().item())
print("token_logits isnan:", torch.isnan(token_logits).any().item())

loss_mse = trainer.criterion_mse(next_frame, target_frame_b)
loss_ce = trainer.criterion_ce(token_logits, target_token_b)
print("loss_mse:", loss_mse.item(), "loss_ce:", loss_ce.item())
