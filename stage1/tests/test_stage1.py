"""stage1/tests/test_stage1.py — V31all
Tests for M scalars, Layer-2 scalars, Triangulation, RecordBuilder.
stdlib only. No torch required.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1.core.m_scalars import M1State, M2State, M3State, compute_m_scalars
from stage1.core.layer2_scalars import build_layer2, BinaryScalar, GeometryScalar, LanguageScalar
from stage1.core.triangulation import TriangulationState, triangulate, build_mathematical_relationship
from stage1.core.record_builder import RecordBuilder


def test_m1_unbounded():
    m1 = M1State()
    v1 = m1.step(1.0)
    v2 = m1.step(100.0)
    assert v2 != v1, "M1 must update across steps"
    assert m1.branch_count == 2
    print("PASS test_m1_unbounded")


def test_m2_clips_to_01():
    m1 = M1State()
    m2 = M2State()
    for _ in range(20):
        m1.step(10.0)
    score = m2.observe(m1)
    assert 0.0 <= score <= 1.0, f"M2 must be in [0,1], got {score}"
    print("PASS test_m2_clips_to_01")


def test_m3_range():
    m1 = M1State()
    m2 = M2State()
    m3 = M3State()
    m1.step(5.0)
    m2_score = m2.observe(m1)
    mult = m3.observe(m1, m2_score)
    assert 0.5 <= mult <= 2.0, f"M3 multiplier out of range: {mult}"
    print("PASS test_m3_range")


def test_binary_scalar_range():
    b = BinaryScalar(codebase_hash="abc123", mission="test mission", is_logical=True)
    val = b.compute()
    assert 0.0 <= val <= 1.0
    print("PASS test_binary_scalar_range")


def test_geometry_scalar_progress():
    g = GeometryScalar(entry_index=25, total_entries=50)
    val = g.compute()
    assert 0.0 <= val <= 1.0
    g2 = GeometryScalar(entry_index=50, total_entries=50)
    assert g2.compute() >= g.compute(), "Later entry should have higher geometry scalar"
    print("PASS test_geometry_scalar_progress")


def test_language_scalar_density():
    l = LanguageScalar(text="hello world this is a test with unique words")
    val = l.compute()
    assert 0.0 <= val <= 1.0
    l_empty = LanguageScalar(text="")
    assert l_empty.compute() == 0.0
    print("PASS test_language_scalar_density")


def test_triangulation_7_inputs():
    state = TriangulationState()
    v, nfp, lto = triangulate(
        M1=42.0, M2=0.7, M3=1.1,
        binary=0.8, geometry=0.5, language=0.6,
        state=state,
    )
    assert 0.0 <= v <= 1.0, f"V out of range: {v}"
    assert isinstance(nfp, str) and len(nfp) > 0
    assert isinstance(lto, str) and len(lto) > 0
    print(f"PASS test_triangulation_7_inputs: V={v:.4f}, action={lto}")


def test_triangulation_self_reference():
    state = TriangulationState()
    v_vals = []
    for i in range(10):
        v, _, _ = triangulate(1.0 * i, 0.5, 1.0, 0.7, 0.5, 0.6, state)
        v_vals.append(v)
    # invariant: V should not be identical every step (self-momentum changes it)
    assert len(set(round(v, 4) for v in v_vals)) > 1, "V must vary with self-reference"
    print("PASS test_triangulation_self_reference")


def test_record_builder_schema():
    rb = RecordBuilder(total_entries=10, codebase_hash="test", mission="unit test")
    record = rb.build(
        binary_text="Phase init: logical.",
        geometry_text="Entry 1/10. Sandbox.",
        language_text="Testing record builder schema.",
    )
    assert "M1" in record
    assert "M2" in record
    assert "M3" in record
    assert "scalar_inputs" in record
    si = record["scalar_inputs"]
    assert "binary" in si
    assert "geometry" in si
    assert "language" in si
    assert "mathematical_relationship" in si
    ot = record["output_tokens"]
    assert "next_frame_prediction" in ot
    assert "language_token_output" in ot
    print("PASS test_record_builder_schema")


def test_record_builder_jsonl():
    rb = RecordBuilder(total_entries=3, codebase_hash="hash", mission="jsonl test")
    for i in range(3):
        line = rb.build_jsonl_line(
            binary_text=f"Entry {i+1} binary.",
            geometry_text=f"Entry {i+1} geometry.",
            language_text=f"Entry {i+1} language context.",
        )
        parsed = json.loads(line)
        assert "_meta" not in parsed, "JSONL output must not include _meta"
    print("PASS test_record_builder_jsonl")


if __name__ == "__main__":
    test_m1_unbounded()
    test_m2_clips_to_01()
    test_m3_range()
    test_binary_scalar_range()
    test_geometry_scalar_progress()
    test_language_scalar_density()
    test_triangulation_7_inputs()
    test_triangulation_self_reference()
    test_record_builder_schema()
    test_record_builder_jsonl()
    print("\nAll stage1 tests passed.")
