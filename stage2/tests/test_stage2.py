"""stage2/tests/test_stage2.py — V31all
Tests for CorpusGenerator, TriangulationLoss, PhaseGate.
stdlib only.
"""
import sys, json, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage2.training_engine.corpus_generator import CorpusGenerator, generate_default_specs
from stage2.training_engine.triangulation_loss import (
    v_prediction_loss, action_alignment_loss, invariant_loss, total_triangulation_loss
)
from stage2.dag_compiler.phase_gate import PhaseGate
from stage1.core.triangulation import ACTION_VOCAB


def test_corpus_generator_produces_correct_count():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "corpus.jsonl")
        specs = generate_default_specs(n=10)
        gen = CorpusGenerator(output_path=out, total_entries=10, codebase_hash="test", mission="t")
        records = gen.run(specs)
        assert len(records) == 10
        # verify file line count
        with open(out) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 10
    print("PASS test_corpus_generator_produces_correct_count")


def test_corpus_generator_schema():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "corpus.jsonl")
        specs = generate_default_specs(n=3)
        gen = CorpusGenerator(output_path=out, total_entries=3, codebase_hash="test", mission="t")
        gen.run(specs)
        with open(out) as f:
            for line in f:
                rec = json.loads(line)
                assert "M1" in rec and "M2" in rec and "M3" in rec
                assert "scalar_inputs" in rec
                assert "output_tokens" in rec
                ot = rec["output_tokens"]
                assert "next_frame_prediction" in ot
                assert "language_token_output" in ot
    print("PASS test_corpus_generator_schema")


def test_corpus_stats():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "corpus.jsonl")
        specs = generate_default_specs(n=20)
        gen = CorpusGenerator(output_path=out, total_entries=20, codebase_hash="test", mission="t")
        gen.run(specs)
        stats = gen.stats()
        assert stats["total_records"] == 20
        assert 0.0 <= stats["V_mean"] <= 1.0
    print("PASS test_corpus_stats")


def test_v_prediction_loss():
    loss = v_prediction_loss(0.8, 0.9)
    assert abs(loss - 0.01) < 1e-6
    loss_zero = v_prediction_loss(0.5, 0.5)
    assert loss_zero == 0.0
    print("PASS test_v_prediction_loss")


def test_action_alignment_loss():
    vocab = ACTION_VOCAB
    # exact match
    assert action_alignment_loss("explore", "explore", vocab) == 0.0
    # different actions — should be > 0
    loss = action_alignment_loss("explore", "deploy", vocab)
    assert loss > 0.0
    print("PASS test_action_alignment_loss")


def test_invariant_loss():
    history = [0.5, 0.5, 0.5]
    loss = invariant_loss(history, 0.5)
    assert loss == 0.0
    loss2 = invariant_loss(history, 0.9)
    assert loss2 > 0.0
    print("PASS test_invariant_loss")


def test_total_triangulation_loss_keys():
    result = total_triangulation_loss(
        predicted_v=0.7,
        target_reward=0.8,
        predicted_action="converge",
        expert_action="converge",
        action_vocab=ACTION_VOCAB,
        v_history=[0.65, 0.68, 0.70],
        m3_multiplier=1.0,
        elapsed_fraction=0.3,
    )
    for key in ["total", "v_prediction", "action_alignment", "invariant", "m3_budget"]:
        assert key in result, f"Missing key: {key}"
    assert result["total"] >= 0.0
    print("PASS test_total_triangulation_loss_keys")


def test_phase_gate_builds_then_opens():
    with tempfile.TemporaryDirectory() as tmp:
        gate = PhaseGate(V_threshold=0.5, M2_floor=0.1, M3_range=(0.5, 2.0))
        gate._log = Path(tmp) / "gate.jsonl"
        # Two consecutive passing checks should open gate
        s1 = gate.check(V=0.8, action="converge", M2=0.5, M3=1.0)
        s2 = gate.check(V=0.8, action="converge", M2=0.5, M3=1.0)
        assert s2 == "OPEN", f"Gate should be OPEN after 2 passes, got {s2}"
    print("PASS test_phase_gate_builds_then_opens")


def test_phase_gate_fails_on_bad_action():
    with tempfile.TemporaryDirectory() as tmp:
        gate = PhaseGate(V_threshold=0.5, M2_floor=0.1, M3_range=(0.5, 2.0))
        gate._log = Path(tmp) / "gate.jsonl"
        gate.check(V=0.8, action="explore", M2=0.5, M3=1.0)  # explore not in convergence
        status = gate.check(V=0.8, action="explore", M2=0.5, M3=1.0)
        assert status == "BUILDING"
    print("PASS test_phase_gate_fails_on_bad_action")


if __name__ == "__main__":
    test_corpus_generator_produces_correct_count()
    test_corpus_generator_schema()
    test_corpus_stats()
    test_v_prediction_loss()
    test_action_alignment_loss()
    test_invariant_loss()
    test_total_triangulation_loss_keys()
    test_phase_gate_builds_then_opens()
    test_phase_gate_fails_on_bad_action()
    print("\nAll stage2 tests passed.")
