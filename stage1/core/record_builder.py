"""stage1/core/record_builder.py — V31all
Assembles complete JSONL records in the v31 7-input → 2-output schema.

Schema:
{
  "M1": <str description of unbounded thinker state>,
  "M2": <str description of efficiency zealot state>,
  "M3": <str description of meta-observer state>,
  "scalar_inputs": {
    "binary":  <str>,
    "geometry": <str>,
    "language": <str>,
    "mathematical_relationship": <str>   ← Triangulation receives all 6+self
  },
  "output_tokens": {
    "next_frame_prediction": <str>,
    "language_token_output": <str>
  }
}

The mathematical_relationship field encodes Triangulation's full computation
across all 7 inputs (including its own prior state). This is the RL value
function field in human-readable form.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import time

from stage1.core.m_scalars import M1State, M2State, M3State, compute_m_scalars
from stage1.core.layer2_scalars import build_layer2
from stage1.core.triangulation import (
    TriangulationState,
    triangulate,
    build_mathematical_relationship,
)


@dataclass
class RecordBuilder:
    """
    Stateful builder. Maintains M1/M2/M3/Triangulation across entries
    so each record inherits momentum from prior records (self-referential
    invariant preservation per spec).
    """
    total_entries: int = 50
    sandbox: bool = True
    system_id: str = "CASEBELIZE"
    codebase_hash: str = ""
    mission: str = ""

    _m1: M1State = field(default_factory=M1State)
    _m2: M2State = field(default_factory=M2State)
    _m3: M3State = field(default_factory=M3State)
    _tri: TriangulationState = field(default_factory=TriangulationState)
    _entry: int = 0

    def _m_descriptions(self, m_vals: dict) -> tuple:
        """Generate M1/M2/M3 description strings from current state."""
        m1_desc = (
            f"Unbounded thinker; branches={self._m1.branch_count}; "
            f"depth={self._m1.exploration_depth:.2f}; "
            f"exploration_vector={m_vals['M1']:.4f}; no valence signal."
        )
        m2_desc = (
            f"Efficiency zealot; observes M1; "
            f"efficiency_score={m_vals['M2']:.4f}; "
            f"budget_remaining={self._m2.cost_budget:.3f}; "
            f"blind to novel discovery."
        )
        m3_desc = (
            f"Meta-observer; observes M1+M2; "
            f"meta_multiplier={m_vals['M3']:.4f}; "
            f"logic_mult={self._m3.logic_multiplier:.2f}; "
            f"efficiency_mult={self._m3.efficiency_multiplier:.2f}; "
            f"creative_mult={self._m3.creative_multiplier:.2f}; "
            f"manages own time budget."
        )
        return m1_desc, m2_desc, m3_desc

    def build(
        self,
        binary_text: str,
        geometry_text: str,
        language_text: str,
        stimulus: float = 1.0,
        is_logical: bool = True,
        telemetry_stable: bool = True,
    ) -> dict:
        """
        Build one complete record. Advances internal state.
        """
        self._entry += 1

        # Layer-1: compute M scalars
        m_vals = compute_m_scalars(self._m1, self._m2, self._m3, stimulus)

        # Layer-2: compute contextual scalars
        l2 = build_layer2(
            binary_hash=self.codebase_hash,
            mission=self.mission,
            purpose=language_text[:80],
            entry_index=self._entry,
            total_entries=self.total_entries,
            language_text=language_text,
            sandbox=self.sandbox,
            system_id=self.system_id,
            is_logical=is_logical,
            telemetry_stable=telemetry_stable,
        )

        # Triangulation: receives all 7 (6 external + self-momentum)
        v, nfp, lto = triangulate(
            M1=m_vals["M1"],
            M2=m_vals["M2"],
            M3=m_vals["M3"],
            binary=l2["binary_scalar"],
            geometry=l2["geometry_scalar"],
            language=l2["language_scalar"],
            state=self._tri,
        )

        math_rel = build_mathematical_relationship(
            M1=m_vals["M1"],
            M2=m_vals["M2"],
            M3=m_vals["M3"],
            binary=l2["binary_scalar"],
            geometry=l2["geometry_scalar"],
            language=l2["language_scalar"],
            context_tags=f"entry={self._entry}, V={v:.4f}",
        )

        # Override with human-readable text fields from caller
        scalar_binary = binary_text if binary_text else l2["binary_field"]
        scalar_geometry = geometry_text if geometry_text else l2["geometry_field"]
        scalar_language = language_text

        m1_desc, m2_desc, m3_desc = self._m_descriptions(m_vals)

        record = {
            "M1": m1_desc,
            "M2": m2_desc,
            "M3": m3_desc,
            "scalar_inputs": {
                "binary": scalar_binary,
                "geometry": scalar_geometry,
                "language": scalar_language,
                "mathematical_relationship": math_rel,
            },
            "output_tokens": {
                "next_frame_prediction": nfp,
                "language_token_output": lto,
            },
        }

        # attach raw scalar values as metadata (not in spec output but useful)
        record["_meta"] = {
            "entry": self._entry,
            "V": round(v, 6),
            "M1_raw": round(m_vals["M1"], 6),
            "M2_raw": round(m_vals["M2"], 6),
            "M3_raw": round(m_vals["M3"], 6),
            "binary_scalar": round(l2["binary_scalar"], 6),
            "geometry_scalar": round(l2["geometry_scalar"], 6),
            "language_scalar": round(l2["language_scalar"], 6),
            "timestamp": round(time.time(), 3),
        }

        return record

    def build_jsonl_line(self, **kwargs) -> str:
        """Return single JSONL-serialized record (no _meta in output)."""
        record = self.build(**kwargs)
        output = {k: v for k, v in record.items() if k != "_meta"}
        return json.dumps(output)
