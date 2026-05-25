"""stage2/training_engine/corpus_generator.py — V31all
Corpus generator: takes a list of entry spec dicts and produces a
complete JSONL corpus file in the v31 7-input → 2-output schema.

Replaces: asymmetric_train.py, flywheel.py, domain_balanced_replay.
The v31 schema has no domain classifier — domain is implicit in
Triangulation's V output and the chosen language_token_output.

Usage:
  gen = CorpusGenerator(output_path="corpus_v31.jsonl", total_entries=50)
  gen.run(entry_specs)  # entry_specs: list of dicts with keys matching build()

Entry spec keys:
  binary_text    : str — binary field narrative
  geometry_text  : str — geometry field narrative
  language_text  : str — language field narrative
  stimulus       : float (optional, default 1.0)
  is_logical     : bool (optional, default True)
  telemetry_stable: bool (optional, default True)
"""

import json
import time
from pathlib import Path
from typing import List, Optional

from stage1.core.m_scalars import M1State, M2State, M3State
from stage1.core.triangulation import TriangulationState
from stage1.core.record_builder import RecordBuilder


class CorpusGenerator:
    def __init__(
        self,
        output_path: str = "corpus_v31.jsonl",
        total_entries: int = 50,
        sandbox: bool = True,
        system_id: str = "CASEBELIZE",
        codebase_hash: str = "v31all",
        mission: str = "7-input 2-output triangulation corpus generation",
        strip_meta: bool = True,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.total_entries = total_entries
        self.strip_meta = strip_meta

        self.builder = RecordBuilder(
            total_entries=total_entries,
            sandbox=sandbox,
            system_id=system_id,
            codebase_hash=codebase_hash,
            mission=mission,
        )
        self._records: List[dict] = []

    def run(self, entry_specs: List[dict]) -> List[dict]:
        """
        Generate all records from specs. Write to JSONL.
        Returns list of full records (with _meta).
        """
        with open(self.output_path, "w", encoding="utf-8") as f:
            for i, spec in enumerate(entry_specs):
                record = self.builder.build(
                    binary_text=spec.get("binary_text", ""),
                    geometry_text=spec.get("geometry_text", ""),
                    language_text=spec.get("language_text", ""),
                    stimulus=spec.get("stimulus", 1.0 + i * 0.1),
                    is_logical=spec.get("is_logical", True),
                    telemetry_stable=spec.get("telemetry_stable", True),
                )
                self._records.append(record)

                # write without _meta
                output = {k: v for k, v in record.items() if k != "_meta"}
                f.write(json.dumps(output) + "\n")

        print(f"[CorpusGenerator] Wrote {len(self._records)} records → {self.output_path}")
        return self._records

    def stats(self) -> dict:
        """Compute corpus statistics from generated records."""
        if not self._records:
            return {}
        v_values = [r["_meta"]["V"] for r in self._records]
        actions = [r["output_tokens"]["language_token_output"] for r in self._records]
        action_counts: dict = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        return {
            "total_records": len(self._records),
            "V_mean": round(sum(v_values) / len(v_values), 4),
            "V_min": round(min(v_values), 4),
            "V_max": round(max(v_values), 4),
            "action_distribution": action_counts,
        }


def generate_default_specs(n: int = 50, theme: str = "adapter_training") -> List[dict]:
    """
    Generate n default entry specs for a given theme.
    Useful for smoke-testing the corpus generator without a custom spec list.
    """
    phases = [
        "initialization", "schema_design", "data_inspection", "test_red",
        "training_epoch1", "training_epoch5", "training_final",
        "rare_state_eval", "retrain_decision", "deployment",
        "monitoring", "fine_tuning", "production_stable", "documentation",
        "archival",
    ]
    specs = []
    for i in range(n):
        phase = phases[i % len(phases)]
        specs.append({
            "binary_text": (
                f"Phase {phase}: entry {i+1}/{n}. "
                f"Codebase initialized. Mission: {theme}. Logical=True."
            ),
            "geometry_text": (
                f"Entry {i+1}/{n} at t={i*60}s. "
                f"Sandbox=True. SystemID=CASEBELIZE. Telemetry=stable."
            ),
            "language_text": (
                f"Generating v31 corpus entry {i+1}. Theme={theme}. "
                f"Phase={phase}. 7-input schema active. "
                f"Triangulation receiving M1+M2+M3+binary+geometry+language+self."
            ),
            "stimulus": 1.0 + (i % 5) * 0.3,
            "is_logical": True,
            "telemetry_stable": True,
        })
    return specs
