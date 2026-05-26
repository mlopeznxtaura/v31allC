"""debug_frames.py"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2.training_engine.asymmetric_trainer import AsymmetricTrainer
from stage2.training_engine.neural_model import V31Model

device = "cuda"
model = V31Model().to(device)
trainer = AsymmetricTrainer(model, lr=3e-4)
frames = trainer.load_corpus_data("output/master_corpus.jsonl")
print(f"Frames loaded: {len(frames)}")
if frames:
    f = frames[0]
    print("m1:", f["m1"], "m2:", f["m2"], "m3:", f["m3"])
    print("geo:", f["geo"][:3], "nfp sum:", f["next_frame_target"].sum().item())
    all_m1 = [x["m1"] for x in frames[:50]]
    all_m2 = [x["m2"] for x in frames[:50]]
    import math
    nan_m1 = sum(1 for v in all_m1 if math.isnan(v))
    nan_m2 = sum(1 for v in all_m2 if math.isnan(v))
    print(f"nan m1 in first 50: {nan_m1}  nan m2: {nan_m2}")
    nfp_sums = [x["next_frame_target"].sum().item() for x in frames[:10]]
    print("nfp_sums first 10:", nfp_sums)
