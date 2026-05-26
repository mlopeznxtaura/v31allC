"""stage1/core/record_builder.py - V31all
Assembles complete JSONL records in the v31 7-input -> 2-output schema.
build_batch() uses GPU-accelerated language scalar computation for large ingests.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json
import time

from stage1.core.m_scalars import M1State, M2State, M3State, compute_m_scalars
from stage1.core.layer2_scalars import build_layer2, batch_language_scalars, LanguageScalar, BinaryScalar, GeometryScalar
from stage1.core.output_synthesis import OutputSynthesizer
from stage1.core.triangulation import (
    TriangulationState,
    triangulate,
    build_mathematical_relationship,
)
from stage1.core.neural_l1_engine import NeuralL1State


@dataclass
class RecordBuilder:
    """
    Stateful builder. Maintains M1/M2/M3/Triangulation across entries.
    Use build() for single records, build_batch() for large ingests (GPU path).
    """
    total_entries: int = 50
    sandbox: bool = True
    system_id: str = "CASEBELIZE"
    codebase_hash: str = ""
    mission: str = ""
    neural_l1: Optional[NeuralL1State] = None

    _m1: M1State = field(default_factory=M1State)
    _m2: M2State = field(default_factory=M2State)
    _m3: M3State = field(default_factory=M3State)
    _tri: TriangulationState = field(default_factory=TriangulationState)
    _entry: int = 0

    def _m_descriptions(self, m_vals: dict) -> tuple:
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
        self._entry += 1
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

        if self.neural_l1 is not None and self.neural_l1.is_loaded:
            progress = self._entry / max(self.total_entries, 1)
            m1_val, m2_val, m3_val = self.neural_l1.step(
                progress=progress,
                language_scalar=l2["language_scalar"],
                binary_scalar=l2["binary_scalar"],
                stimulus=stimulus
            )
            m_vals = {"M1": m1_val, "M2": m2_val, "M3": m3_val}
            # Also tick simulated objects to keep step/branch counts in sync
            _ = compute_m_scalars(self._m1, self._m2, self._m3, stimulus)
        else:
            m_vals = compute_m_scalars(self._m1, self._m2, self._m3, stimulus)

        v, nfp_tri, lto_tri = triangulate(
            M1=m_vals["M1"], M2=m_vals["M2"], M3=m_vals["M3"],
            binary=l2["binary_scalar"], geometry=l2["geometry_scalar"],
            language=l2["language_scalar"], state=self._tri,
        )
        nfp, lto, pixel_frame = OutputSynthesizer.synthesize(
            M1=m_vals["M1"], M2=m_vals["M2"], M3=m_vals["M3"],
            binary=l2["binary_scalar"], geometry=l2["geometry_scalar"],
            language=l2["language_scalar"], V=v
        )
        math_rel = build_mathematical_relationship(
            M1=m_vals["M1"], M2=m_vals["M2"], M3=m_vals["M3"],
            binary=l2["binary_scalar"], geometry=l2["geometry_scalar"],
            language=l2["language_scalar"],
            context_tags=f"entry={self._entry}, V={v:.4f}",
        )
        scalar_binary = binary_text if binary_text else l2["binary_field"]
        scalar_geometry = geometry_text if geometry_text else l2["geometry_field"]
        m1_desc, m2_desc, m3_desc = self._m_descriptions(m_vals)
        record = {
            "M1": m1_desc, "M2": m2_desc, "M3": m3_desc,
            "scalar_inputs": {
                "binary": scalar_binary,
                "geometry": scalar_geometry,
                "language": language_text,
                "mathematical_relationship": math_rel,
            },
            "output_tokens": {"next_frame_prediction": nfp, "language_token_output": lto},
        }
        record["_meta"] = {
            "entry": self._entry, "V": round(v, 6),
            "M1_raw": round(m_vals["M1"], 6), "M2_raw": round(m_vals["M2"], 6),
            "M3_raw": round(m_vals["M3"], 6),
            "binary_scalar": round(l2["binary_scalar"], 6),
            "geometry_scalar": round(l2["geometry_scalar"], 6),
            "language_scalar": round(l2["language_scalar"], 6),
            "timestamp": round(time.time(), 3),
            "pixel_frame_shape": pixel_frame.shape,
            "pixel_frame_base64": __import__('base64').b64encode(pixel_frame.tobytes()).decode(),
        }
        return record

    def build_batch(self, rows: List[dict]) -> List[dict]:
        """
        GPU-accelerated batch build for large ingests.
        Each row: {"binary_text": str, "geometry_text": str, "language_text": str,
                   "stimulus": float, "is_logical": bool, "telemetry_stable": bool}
        Language scalars are computed in one GPU tensor op, then per-record
        triangulation runs sequentially (stateful M1/M2/M3 must stay serial).
        """
        if not rows:
            return []

        t0 = time.time()
        n = len(rows)

        # 1. GPU batch: language scalars for all rows at once
        texts = [r.get("language_text", "") for r in rows]
        lang_scalars = batch_language_scalars(texts)
        print(f"[RecordBuilder.build_batch] {n} records, lang scalars GPU: {time.time()-t0:.3f}s")

        # 2. Serial loop for stateful M/triangulation (cannot parallelise)
        results = []
        for i, (row, lang_scalar) in enumerate(zip(rows, lang_scalars)):
            self._entry += 1
            binary_text   = row.get("binary_text", "")
            geometry_text = row.get("geometry_text", "")
            language_text = row.get("language_text", "")
            stimulus      = float(row.get("stimulus", 1.0))
            is_logical    = bool(row.get("is_logical", True))
            telemetry_stable = bool(row.get("telemetry_stable", True))

            # Binary + geometry still CPU (fast hash/math)
            b = BinaryScalar(codebase_hash=self.codebase_hash, mission=self.mission,
                             purpose=language_text[:80], is_logical=is_logical)
            g = GeometryScalar(entry_index=self._entry, total_entries=max(self.total_entries, n),
                               sandbox=self.sandbox, system_id=self.system_id,
                               telemetry_stable=telemetry_stable)
            binary_scalar   = b.compute()
            geometry_scalar = g.compute()
            # Use pre-computed GPU language scalar
            language_scalar = lang_scalar

            if self.neural_l1 is not None and self.neural_l1.is_loaded:
                progress = self._entry / max(self.total_entries, 1)
                m1_val, m2_val, m3_val = self.neural_l1.step(
                    progress=progress,
                    language_scalar=language_scalar,
                    binary_scalar=binary_scalar,
                    stimulus=stimulus
                )
                m_vals = {"M1": m1_val, "M2": m2_val, "M3": m3_val}
                # Tick the simulated objects too to keep track of branch count
                _ = compute_m_scalars(self._m1, self._m2, self._m3, stimulus)
            else:
                m_vals = compute_m_scalars(self._m1, self._m2, self._m3, stimulus)

            v, _, _ = triangulate(
                M1=m_vals["M1"], M2=m_vals["M2"], M3=m_vals["M3"],
                binary=binary_scalar, geometry=geometry_scalar,
                language=language_scalar, state=self._tri,
            )
            nfp, lto, pixel_frame = OutputSynthesizer.synthesize(
                M1=m_vals["M1"], M2=m_vals["M2"], M3=m_vals["M3"],
                binary=binary_scalar, geometry=geometry_scalar,
                language=language_scalar, V=v
            )
            math_rel = build_mathematical_relationship(
                M1=m_vals["M1"], M2=m_vals["M2"], M3=m_vals["M3"],
                binary=binary_scalar, geometry=geometry_scalar,
                language=language_scalar,
                context_tags=f"entry={self._entry}, V={v:.4f}",
            )
            m1_desc, m2_desc, m3_desc = self._m_descriptions(m_vals)
            record = {
                "M1": m1_desc, "M2": m2_desc, "M3": m3_desc,
                "scalar_inputs": {
                    "binary": binary_text or b.to_field(),
                    "geometry": geometry_text or g.to_field(),
                    "language": language_text,
                    "mathematical_relationship": math_rel,
                },
                "output_tokens": {"next_frame_prediction": nfp, "language_token_output": lto},
                "_meta": {
                    "entry": self._entry, "V": round(v, 6),
                    "M1_raw": round(m_vals["M1"], 6), "M2_raw": round(m_vals["M2"], 6),
                    "M3_raw": round(m_vals["M3"], 6),
                    "binary_scalar": round(binary_scalar, 6),
                    "geometry_scalar": round(geometry_scalar, 6),
                    "language_scalar": round(language_scalar, 6),
                    "timestamp": round(time.time(), 3),
                    "pixel_frame_shape": pixel_frame.shape,
                    "pixel_frame_base64": __import__('base64').b64encode(pixel_frame.tobytes()).decode(),
                }
            }
            results.append(record)

        print(f"[RecordBuilder.build_batch] total: {time.time()-t0:.3f}s for {n} records")
        return results

    def build_jsonl_line(self, **kwargs) -> str:
        record = self.build(**kwargs)
        output = {k: v for k, v in record.items() if k != "_meta"}
        return json.dumps(output)