import json
import subprocess
import tempfile
from pathlib import Path


def test_ingest_v31_and_legacy_mix():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inp1 = td / "a.jsonl"
        inp2 = td / "b.jsonl"
        out = td / "out.jsonl"

        v31 = {
            "M1": "m1", "M2": "m2", "M3": "m3",
            "scalar_inputs": {
                "binary": "b", "geometry": "g", "language": "l", "mathematical_relationship": "m"
            },
            "output_tokens": {
                "next_frame_prediction": "n", "language_token_output": "act"
            },
        }
        legacy = {
            "Binary": "B", "Geometry": "G", "LanguageContext": "L",
            "MathematicalRelationshipTriangulation": "M",
            "NextFramePrediction": "N", "LanguageTokenOutput": "deploy"
        }

        inp1.write_text(json.dumps(v31) + "\n", encoding="utf-8")
        inp2.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

        cmd = [
            "python3",
            "stage2/training_engine/corpus_ingest.py",
            "--inputs", str(inp1), str(inp2),
            "--output", str(out),
            "--strict",
        ]
        subprocess.check_call(cmd, cwd="/mnt/d/NextAura/v31all_1/v31allC")

        rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(rows) == 2
        assert rows[0]["output_tokens"]["language_token_output"] == "act"
        assert rows[1]["scalar_inputs"]["language"] == "L"
        assert rows[1]["output_tokens"]["language_token_output"] == "deploy"
