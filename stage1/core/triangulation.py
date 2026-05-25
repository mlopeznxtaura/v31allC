"""stage1/core/triangulation.py — V31all
Triangulation: the RL value function at the center of the 7-input schema.

Receives all 7 inputs:
  Layer-1: M1 (unbounded), M2 (efficiency), M3 (meta)
  Layer-2: Binary (initialization), Geometry (position), Language (narrative)
  Self: Triangulation's own previous value (invariant preservation)

Outputs 2 tokens:
  next_frame_prediction — what happens next (string narrative)
  language_token_output — RL-applied output token (actual decision/action)

The triangulation value V(s) is computed as a weighted combination of all
6 external inputs plus momentum from its own prior value.
Invariant: V ∈ [0, 1]. The function is a soft gating mechanism that decides
how much weight each scalar dimension carries in the final output tokens.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple


# Weight vector for the 7 inputs (sum = 1.0)
# Order: [M1, M2, M3, binary, geometry, language, self_momentum]
DEFAULT_WEIGHTS = [0.20, 0.15, 0.20, 0.12, 0.13, 0.10, 0.10]

# Action vocabulary: LanguageTokenOutput draws from these
ACTION_VOCAB = [
    "explore",
    "throttle",
    "rebalance",
    "converge",
    "verify",
    "archive",
    "retrain",
    "deploy",
    "monitor",
    "idle",
    "diverge",
    "checkpoint",
]

# State vocabulary: NextFramePrediction draws from these
STATE_VOCAB = [
    "training",
    "evaluation",
    "drift_detected",
    "loop_detected",
    "sandbox_confirmed",
    "convergence",
    "plateau",
    "spike",
    "gate_open",
    "gate_closed",
    "retrain_triggered",
    "deployment_ready",
]


@dataclass
class TriangulationState:
    """
    Maintains the running triangulation value and history.
    Self-referential: prior V feeds into next V (invariant preservation).
    """
    weights: list = field(default_factory=lambda: list(DEFAULT_WEIGHTS))
    _prior_v: float = 0.5
    _history: list = field(default_factory=list)
    _step: int = 0

    def compute(
        self,
        M1: float,
        M2: float,
        M3: float,
        binary: float,
        geometry: float,
        language: float,
    ) -> float:
        """
        V(s) = weighted sum of [M1_norm, M2, M3_norm, binary, geometry, language, self]
        M1 is unbounded → soft-normalize via tanh before weighting.
        M3 may exceed 1.0 → clamp to [0.5, 2.0] then rescale to [0, 1].
        """
        # normalize unbounded inputs
        m1_norm = math.tanh(M1 / max(abs(M1) + 1e-8, 1.0))   # [-1,1] → squeeze
        m1_norm = (m1_norm + 1.0) / 2.0                        # → [0,1]
        m3_norm = (min(max(M3, 0.5), 2.0) - 0.5) / 1.5        # → [0,1]

        inputs = [m1_norm, M2, m3_norm, binary, geometry, language, self._prior_v]

        # weighted sum
        v = sum(w * x for w, x in zip(self.weights, inputs))
        v = min(max(v, 0.0), 1.0)

        self._prior_v = v
        self._step += 1
        self._history.append(v)
        if len(self._history) > 100:
            self._history.pop(0)
        return v

    def momentum(self) -> float:
        """Running mean of last 5 V values."""
        window = self._history[-5:] if len(self._history) >= 5 else self._history
        return sum(window) / max(len(window), 1)

    def next_frame_prediction(self, v: float, geometry: float) -> str:
        """
        Map V + geometry to a STATE_VOCAB string for next_frame_prediction.
        Uses V to select state, geometry to contextualize.
        """
        idx = int(v * (len(STATE_VOCAB) - 1))
        state = STATE_VOCAB[idx]
        step_next = self._step + 1
        progress_str = f"{geometry * 100:.1f}%"
        return f"Entry {step_next} will be in state [{state}] at progress={progress_str}."

    def language_token_output(self, v: float, M2: float, M3: float) -> str:
        """
        RL-applied action token.
        M2 (efficiency) biases toward throttle/converge.
        M3 (meta) biases toward rebalance if multiplier is high.
        V anchors the base action.
        """
        # base action from V
        base_idx = int(v * (len(ACTION_VOCAB) - 1))
        action = ACTION_VOCAB[base_idx]

        # M2 override: if very low efficiency, throttle
        if M2 < 0.15:
            action = "throttle"
        # M3 override: if meta multiplier very high, rebalance
        elif M3 > 1.7:
            action = "rebalance"
        # convergence zone
        elif 0.45 <= v <= 0.55:
            action = "converge"
        # high V = near gate
        elif v > 0.85:
            action = "deploy" if self._step > 10 else "checkpoint"
        # very low V = need more data
        elif v < 0.1:
            action = "explore"

        return action


def triangulate(
    M1: float,
    M2: float,
    M3: float,
    binary: float,
    geometry: float,
    language: float,
    state: Optional[TriangulationState] = None,
) -> Tuple[float, str, str]:
    """
    Top-level triangulation call.
    Returns: (V, next_frame_prediction, language_token_output)
    """
    if state is None:
        state = TriangulationState()
    v = state.compute(M1, M2, M3, binary, geometry, language)
    nfp = state.next_frame_prediction(v, geometry)
    lto = state.language_token_output(v, M2, M3)
    return v, nfp, lto


def build_mathematical_relationship(
    M1: float,
    M2: float,
    M3: float,
    binary: float,
    geometry: float,
    language: float,
    context_tags: str = "",
) -> str:
    """
    Human-readable Triangulation field for JSONL records.
    Format matches example corpus style.
    """
    return (
        f"Triangulation(M1={M1:.3f}, M2={M2:.3f}, M3={M3:.3f}, "
        f"binary={binary:.3f}, geometry={geometry:.3f}, language={language:.3f}"
        + (f", context={context_tags}" if context_tags else "")
        + ") → RL value function."
    )
