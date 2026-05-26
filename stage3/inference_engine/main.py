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
        self._v_history = []
    def reset(self):
        """Reset state across turns."""
        self._m1 = M1State()
        self._m2 = M2State()
        self._m3 = M3State()
        self._tri = TriangulationState()
        self._gate = PhaseGate()
        self._turn = 0
        self._v_history = []

    def infer(
        self,
        binary_text: str,
        geometry_text: str,
        language_text: str,
        stimulus: float = 1.0,
        is_logical: bool = True,
        telemetry_stable: bool = True,
        entry_index: Optional[int] = None,
        total_entries: int = 100,
    ) -> Dict:
        """
        Single inference turn.
        entry_index: 1-based position in corpus (used for geometry_scalar progress).
                     Defaults to self._turn if not provided.
        Returns full output record.
        """
        self._turn += 1
        eidx = entry_index if entry_index is not None else self._turn

        # Layer-1
        m_vals = compute_m_scalars(self._m1, self._m2, self._m3, stimulus)

        # Layer-2
        l2 = build_layer2(
            binary_hash=self.codebase_hash,
            mission=self.mission,
            purpose=language_text[:80],
            entry_index=eidx,
            total_entries=total_entries,
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

        Fixes vs original:
        - Passes entry_index (1-based) and total_entries so geometry_scalar
          reflects actual progress through the corpus instead of always 0.0.
        - Extracts stimulus, is_logical, telemetry_stable from record if present.
        - Falls back to top-level M1/M2/M3 descriptor strings when scalar_inputs
          fields are sparse (idle/filler records), so language_scalar is never
          collapsed to near-zero on the 40 idle entries in this corpus.
        - Enriches sparse geometry_text with corpus position so geometry_scalar
          is never stuck at 0.0.
        """
        # Pre-scan to get total count for geometry_scalar denominator
        records = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        total = len(records)
        results = []

        for idx, rec in enumerate(records):
            si = rec.get("scalar_inputs", {})

            # Pull text fields; fall back to top-level M1/M2/M3 descriptor
            # strings when scalar_inputs fields are sparse (idle/filler entries)
            binary_text   = si.get("binary", "")   or str(rec.get("M1", ""))
            geometry_text = si.get("geometry", "") or str(rec.get("M2", ""))
            language_text = si.get("language", "") or str(rec.get("M3", ""))

            # Enrich sparse geometry with corpus position so geometry_scalar
            # is never stuck at 0.0
            if len(geometry_text.strip()) < 20:
                geometry_text = (
                    f"Entry {idx+1}/{total}. Env=Sandbox. SystemID=CASEBELIZE. "
                    f"TelemetryStable=True. {geometry_text}".strip()
                )

            # Enrich sparse language with mathematical_relationship if available
            math_rel = si.get("mathematical_relationship", "")
            if len(language_text.strip()) < 20 and math_rel:
                language_text = math_rel

            stimulus = float(rec.get("stimulus", 1.0))

            result = self.infer(
                binary_text=binary_text,
                geometry_text=geometry_text,
                language_text=language_text,
                stimulus=stimulus,
                is_logical=bool(rec.get("is_logical", True)),
                telemetry_stable=bool(rec.get("telemetry_stable", True)),
                entry_index=idx + 1,
                total_entries=total,
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
