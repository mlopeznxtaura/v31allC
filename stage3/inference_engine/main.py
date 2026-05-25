"""stage3/inference_engine/main.py — V31all
Inference engine: loads a v31 corpus, runs the 7-input triangulation
pipeline over a prompt, and returns next_frame_prediction + language_token_output.

Replaces: stage3/inference_engine/main.py (v26) which used domain_classifier
and separate semantic/kinematic encoders.

V31 inference is stateful — M1/M2/M3/Triangulation maintain state across
turns (sliding window over prior V values for invariant preservation).
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Dict

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1.core.m_scalars import M1State, M2State, M3State, compute_m_scalars
from stage1.core.layer2_scalars import build_layer2
from stage1.core.triangulation import TriangulationState, triangulate
from stage2.training_engine.triangulation_loss import total_triangulation_loss
from stage2.dag_compiler.phase_gate import PhaseGate


class V31InferenceEngine:
    """
    Stateful v31 inference engine.
    Maintains M1/M2/M3/Triangulation across turns.
    """

    def __init__(
        self,
        system_id: str = "CASEBELIZE",
        sandbox: bool = True,
        codebase_hash: str = "v31all",
        mission: str = "NextAura ATP inference",
        time_budget: float = 300.0,
    ):
        self.system_id = system_id
        self.sandbox = sandbox
        self.codebase_hash = codebase_hash
        self.mission = mission

        self._m1 = M1State()
        self._m2 = M2State()
        self._m3 = M3State(time_budget_seconds=time_budget)
        self._tri = TriangulationState()
        self._gate = PhaseGate()
        self._turn = 0
        self._v_history: List[float] = []

    def infer(
        self,
        binary_text: str,
        geometry_text: str,
        language_text: str,
        stimulus: float = 1.0,
        is_logical: bool = True,
        telemetry_stable: bool = True,
    ) -> Dict:
        """
        Single inference turn.
        Returns full output record.
        """
        self._turn += 1

        # Layer-1
        m_vals = compute_m_scalars(self._m1, self._m2, self._m3, stimulus)

        # Layer-2
        l2 = build_layer2(
            binary_hash=self.codebase_hash,
            mission=self.mission,
            purpose=language_text[:80],
            entry_index=self._turn,
            total_entries=100,
            language_text=language_text,
            sandbox=self.sandbox,
            system_id=self.system_id,
            is_logical=is_logical,
            telemetry_stable=telemetry_stable,
        )

        # Triangulation (7 → 2)
        v, nfp, lto = triangulate(
            M1=m_vals["M1"],
            M2=m_vals["M2"],
            M3=m_vals["M3"],
            binary=l2["binary_scalar"],
            geometry=l2["geometry_scalar"],
            language=l2["language_scalar"],
            state=self._tri,
        )
        self._v_history.append(v)

        # Phase gate check
        gate_status = self._gate.check(
            V=v,
            action=lto,
            M2=m_vals["M2"],
            M3=m_vals["M3"],
        )

        return {
            "turn": self._turn,
            "V": round(v, 6),
            "next_frame_prediction": nfp,
            "language_token_output": lto,
            "gate_status": gate_status,
            "M1": round(m_vals["M1"], 4),
            "M2": round(m_vals["M2"], 4),
            "M3": round(m_vals["M3"], 4),
            "binary_scalar": round(l2["binary_scalar"], 4),
            "geometry_scalar": round(l2["geometry_scalar"], 4),
            "language_scalar": round(l2["language_scalar"], 4),
        }

    def run_corpus(self, corpus_path: str) -> List[Dict]:
        """
        Replay a v31 JSONL corpus through inference.
        For each record: extract text fields → infer → attach result.
        """
        results = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                si = rec.get("scalar_inputs", {})
                result = self.infer(
                    binary_text=si.get("binary", ""),
                    geometry_text=si.get("geometry", ""),
                    language_text=si.get("language", ""),
                )
                result["source_record"] = rec
                results.append(result)
        return results


if __name__ == "__main__":
    engine = V31InferenceEngine()
    result = engine.infer(
        binary_text="Phase gate: logical. Codebase v31all initialized.",
        geometry_text="Entry 1. Sandbox=True. SystemID=CASEBELIZE.",
        language_text="Running v31 inference smoke test. 7-input schema active.",
    )
    print(json.dumps(result, indent=2))
