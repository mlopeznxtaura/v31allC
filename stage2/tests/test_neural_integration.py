"""stage2/tests/test_neural_integration.py — V31all Neural Integration Tests
Tests the PyTorch V31Model, AsymmetricTrainer, and V31NeuralInferenceEngine.
Verifies data collation, forward outputs, optimization updates, and state loading.
"""
import sys
import os
import json
import tempfile
import torch
from pathlib import Path

# Insert project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage2.training_engine.neural_model import V31Model
from stage2.training_engine.asymmetric_trainer import AsymmetricTrainer, decode_and_pool_frame, get_deterministic_token_id
from stage3.inference_engine.neural_inference import V31NeuralInferenceEngine
from stage1.core.triangulation import ACTION_VOCAB


def test_decode_and_pool_frame():
    # Empty string should fallback to 224-dim zero tensor
    t_empty = decode_and_pool_frame("")
    assert isinstance(t_empty, torch.Tensor)
    assert t_empty.shape == (224,)
    assert (t_empty == 0.0).all()

    # Test invalid base64
    t_invalid = decode_and_pool_frame("not-base-64!!")
    assert t_invalid.shape == (224,)
    assert (t_invalid == 0.0).all()


def test_deterministic_token_id():
    token_id_1 = get_deterministic_token_id("explore")
    token_id_2 = get_deterministic_token_id("explore")
    token_id_3 = get_deterministic_token_id("deploy")

    assert isinstance(token_id_1, int)
    assert 0 <= token_id_1 < 5000
    assert token_id_1 == token_id_2  # Must be deterministic
    assert token_id_1 != token_id_3  # Different tokens should map to different values (in most cases due to MD5)


def test_model_forward():
    model = V31Model(vocab_size=5000, num_experts=4)
    model.eval()

    # Input dimensions
    # m1, m2, m3: batch size (e.g. 2)
    # geo, bin, tri: (batch, 32)
    # lng: (batch, 128)
    batch_size = 2
    m1 = torch.rand(batch_size, 1)
    m2 = torch.rand(batch_size, 1)
    m3 = torch.rand(batch_size, 1)
    geo = torch.rand(batch_size, 32)
    bin_v = torch.rand(batch_size, 32)
    lng = torch.rand(batch_size, 128)
    tri = torch.rand(batch_size, 32)

    with torch.no_grad():
        next_frame, token_logits = model(m1, m2, m3, geo, bin_v, lng, tri)

    assert next_frame.shape == (batch_size, 224)
    assert token_logits.shape == (batch_size, 5000)


def test_trainer_and_inference_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        corpus_path = td_path / "mock_corpus.jsonl"
        ckpt_path = td_path / "v31_neural_model.pt"

        # Create a mock corpus file matching the expected structure of load_corpus_data
        # with meta and output_tokens
        mock_records = []
        for i in range(5):
            rec = {
                "M1": f"m1_desc_{i}",
                "M2": f"m2_desc_{i}",
                "M3": f"m3_desc_{i}",
                "_meta": {
                    "M1_raw": 1.2 + i,
                    "M2_raw": 0.95 - i * 0.05,
                    "M3_raw": 1.0,
                    "geometry_scalar": 0.05 * i,
                    "binary_scalar": 0.769,
                    "language_scalar": 0.425,
                    "V": 0.5 + i * 0.05,
                    "pixel_frame_base64": "" # empty for simple fallback
                },
                "scalar_inputs": {
                    "binary": f"bin_prompt_{i}",
                    "geometry": f"geo_prompt_{i}",
                    "language": f"lng_prompt_{i}"
                },
                "output_tokens": {
                    "language_token_output": "explore" if i % 2 == 0 else "converge"
                }
            }
            mock_records.append(rec)

        with open(corpus_path, "w", encoding="utf-8") as f:
            for r in mock_records:
                f.write(json.dumps(r) + "\n")

        # 1. Initialize Model
        model = V31Model(vocab_size=5000, num_experts=4)
        initial_proj_weight = model.input_proj.weight.clone()

        # 2. Train using AsymmetricTrainer
        trainer = AsymmetricTrainer(model, lr=0.01)
        res = trainer.train_epoch(str(corpus_path), batch_size=2)

        assert res["records"] == 5
        assert "loss" in res
        assert "mse" in res
        assert "ce" in res
        assert res["loss"] > 0.0

        # Verify optimization occurred (weights changed)
        assert not torch.equal(model.input_proj.weight, initial_proj_weight)

        # 3. Save checkpoint
        trainer.save_checkpoint(str(ckpt_path))
        assert ckpt_path.exists()

        # 4. Initialize Neural Inference Engine and load checkpoint
        engine = V31NeuralInferenceEngine(checkpoint_path=str(ckpt_path))
        assert engine.is_loaded is True

        # 5. Run inference and check result
        inf_res = engine.infer(
            binary_text="Phase gate: logical.",
            geometry_text="Entry 1. Sandbox=True.",
            language_text="Running model test.",
            stimulus=1.0,
        )

        assert inf_res["is_neural"] is True
        assert isinstance(inf_res["neural_frame"], list)
        assert len(inf_res["neural_frame"]) == 224
        assert inf_res["language_token_output"] in ACTION_VOCAB


if __name__ == "__main__":
    print("Running Neural Integration Unit Tests...")
    test_decode_and_pool_frame()
    print("PASS: test_decode_and_pool_frame")
    test_deterministic_token_id()
    print("PASS: test_deterministic_token_id")
    test_model_forward()
    print("PASS: test_model_forward")
    test_trainer_and_inference_end_to_end()
    print("PASS: test_trainer_and_inference_end_to_end")
    print("\nAll neural integration tests PASSED successfully!")
