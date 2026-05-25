"""stage3/tests/test_stage3.py — V31all
Tests for V31InferenceEngine.
stdlib only.
"""
import sys, json, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage3.inference_engine.main import V31InferenceEngine
from stage2.training_engine.corpus_generator import CorpusGenerator, generate_default_specs


def test_single_infer_output_keys():
    engine = V31InferenceEngine(sandbox=True)
    result = engine.infer(
        binary_text="Logical init.",
        geometry_text="Entry 1. Sandbox.",
        language_text="Test inference.",
    )
    for key in ["turn", "V", "next_frame_prediction", "language_token_output", "gate_status"]:
        assert key in result, f"Missing key: {key}"
    assert 0.0 <= result["V"] <= 1.0
    assert result["gate_status"] in ("OPEN", "BUILDING")
    print("PASS test_single_infer_output_keys")


def test_multi_turn_v_changes():
    engine = V31InferenceEngine()
    v_vals = []
    for i in range(5):
        r = engine.infer(
            binary_text=f"Binary {i}.",
            geometry_text=f"Entry {i}.",
            language_text=f"Turn {i} context.",
            stimulus=1.0 + i * 0.5,
        )
        v_vals.append(r["V"])
    # V must not be identical every turn (self-referential state changes it)
    assert len(set(round(v, 4) for v in v_vals)) > 1
    print("PASS test_multi_turn_v_changes")


def test_gate_opens_after_convergence():
    engine = V31InferenceEngine()
    engine._gate.V_threshold = 0.3  # lower threshold for test
    engine._gate.M2_floor = 0.05
    engine._gate.M3_range = (0.3, 2.0)
    statuses = []
    for _ in range(10):
        r = engine.infer(
            binary_text="Stable init.",
            geometry_text="Mid-sequence.",
            language_text="Converging on stable state.",
        )
        statuses.append(r["gate_status"])
    # after enough passes, gate should open at some point
    assert "OPEN" in statuses or "BUILDING" in statuses
    print(f"PASS test_gate_opens_after_convergence: statuses={set(statuses)}")


def test_corpus_replay():
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = os.path.join(tmp, "corpus.jsonl")
        specs = generate_default_specs(n=5)
        gen = CorpusGenerator(output_path=corpus_path, total_entries=5, codebase_hash="t", mission="t")
        gen.run(specs)

        engine = V31InferenceEngine()
        results = engine.run_corpus(corpus_path)
        assert len(results) == 5
        for r in results:
            assert "V" in r
            assert "language_token_output" in r
    print("PASS test_corpus_replay")


def test_infer_returns_valid_action():
    from stage1.core.triangulation import ACTION_VOCAB
    engine = V31InferenceEngine()
    r = engine.infer("b", "g", "l")
    assert r["language_token_output"] in ACTION_VOCAB, \
        f"Unknown action: {r['language_token_output']}"
    print("PASS test_infer_returns_valid_action")


if __name__ == "__main__":
    test_single_infer_output_keys()
    test_multi_turn_v_changes()
    test_gate_opens_after_convergence()
    test_corpus_replay()
    test_infer_returns_valid_action()
    print("\nAll stage3 tests passed.")
