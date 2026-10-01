"""stage2/dag_compiler/phase_gate.py — V31all
Phase gate: replaces v26all cluster_ratio gate with triangulation V gate.

V31 gate conditions (all must hold for 2 consecutive cycles):
  1. V >= V_threshold (default 0.75) — triangulation value above target
  2. language_token_output in CONVERGENCE_ACTIONS — action is stable
  3. M2 > M2_floor (default 0.2) — efficiency not collapsed
  4. M3 in M3_range (default [0.7, 1.5]) — meta multiplier balanced

Replaces: cluster_ratio < 0.3 AND mean_position_loss < 2.0
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

CONVERGENCE_ACTIONS = {"converge", "checkpoint", "deploy", "monitor", "archive"}


@dataclass
class PhaseGate:
    V_threshold: float = 0.72
    M2_floor: float = 0.20
    M3_range: tuple = (0.70, 1.50)
    _history: List[bool] = field(default_factory=list)
    _log: Path = field(default_factory=lambda: Path("telemetry/phase_gate_v31.jsonl"))

    def __post_init__(self):
        self._log.parent.mkdir(parents=True, exist_ok=True)

    def check(
        self,
        V: float,
        action: str,
        M2: float,
        M3: float,
    ) -> str:
        """
        Returns "OPEN" if gate conditions hold for 2+ consecutive cycles.
        Returns "BUILDING" otherwise.
        """
        v_ok = V >= self.V_threshold
        action_ok = action in CONVERGENCE_ACTIONS
        m2_ok = M2 > self.M2_floor
        m3_ok = self.M3_range[0] <= M3 <= self.M3_range[1]

        all_ok = v_ok and action_ok and m2_ok and m3_ok
        self._history.append(all_ok)
        if len(self._history) > 3:
            self._history.pop(0)

        status = "OPEN" if (len(self._history) >= 2 and all(self._history[-2:])) else "BUILDING"

        entry = {
            "status": status,
            "V": V,
            "action": action,
            "M2": M2,
            "M3": M3,
            "v_ok": v_ok,
            "action_ok": action_ok,
            "m2_ok": m2_ok,
            "m3_ok": m3_ok,
            "timestamp": time.time(),
        }
        with open(self._log, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return status

    def is_open(self) -> bool:
        return len(self._history) >= 2 and all(self._history[-2:])
