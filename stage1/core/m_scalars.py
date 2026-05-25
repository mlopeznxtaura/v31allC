"""stage1/core/m_scalars.py — V31all
M1, M2, M3: the three Layer-1 meta-thinker scalars.

M1 — Unbounded thinker
  Emotionally damaged neural-net simulation. Explores forever.
  Generates infinite logical branches. No internal valence signal.
  Output: exploration_vector (float, unbounded, no clipping)

M2 — Efficiency zealot
  Observes M1. Enforces time-based rules. Strict/extreme efficiency.
  Blind to novel discovery. Kills branches by cost-per-step.
  Output: efficiency_score (float, 0.0–1.0, hard-clipped)

M3 — Meta-observer
  Observes M1 + M2. Determines efficiency/logic/creative multipliers.
  Manages its own time budget. Can throttle or reparameterize both.
  Output: meta_multiplier (float, typically 0.5–2.0)

All three feed into Triangulation as Layer-1 inputs.
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class M1State:
    """Unbounded thinker — no valence ceiling."""
    branch_count: int = 0
    exploration_depth: float = 0.0
    _history: List[float] = field(default_factory=list)

    def step(self, stimulus: float) -> float:
        """Accept stimulus, grow unboundedly."""
        self.branch_count += 1
        self.exploration_depth += abs(stimulus) + math.log1p(self.branch_count)
        val = self.exploration_depth / max(self.branch_count, 1)
        self._history.append(val)
        return val  # no clip — M1 is unbounded

    def vector(self) -> float:
        """Current exploration vector."""
        return self.exploration_depth


@dataclass
class M2State:
    """Efficiency zealot — observes M1, kills cost."""
    cost_budget: float = 1.0
    steps_allowed: float = 100.0
    _step_count: int = 0

    def observe(self, m1: M1State) -> float:
        """
        Penalise excessive branching.
        efficiency_score = budget / (1 + branch_penalty)
        Hard-clipped [0, 1].
        """
        self._step_count += 1
        branch_penalty = m1.branch_count / max(self.steps_allowed, 1)
        score = self.cost_budget / (1.0 + branch_penalty)
        return min(max(score, 0.0), 1.0)

    def throttle(self, m1: M1State) -> None:
        """Reduce M1 branch ceiling (M2 can reparameterize M1)."""
        m1.exploration_depth *= 0.9


@dataclass
class M3State:
    """Meta-observer — manages multipliers and time budget."""
    time_budget_seconds: float = 60.0
    _start: float = field(default_factory=time.time)
    logic_multiplier: float = 1.0
    efficiency_multiplier: float = 1.0
    creative_multiplier: float = 1.0

    def observe(self, m1: M1State, m2_score: float) -> float:
        """
        Compute meta_multiplier from M1 depth and M2 score.
        If M2 is too tight (< 0.2): boost creative, reduce efficiency_mult.
        If M1 is too wild (depth > 1000): throttle logic_mult.
        Returns meta_multiplier in [0.5, 2.0].
        """
        elapsed = time.time() - self._start
        budget_fraction = min(elapsed / max(self.time_budget_seconds, 1), 1.0)

        # adjust multipliers
        if m2_score < 0.2:
            self.creative_multiplier = min(self.creative_multiplier * 1.1, 2.0)
            self.efficiency_multiplier = max(self.efficiency_multiplier * 0.95, 0.5)
        if m1.exploration_depth > 1000:
            self.logic_multiplier = max(self.logic_multiplier * 0.9, 0.5)

        meta = (self.logic_multiplier + self.efficiency_multiplier + self.creative_multiplier) / 3.0
        meta *= (1.0 - 0.3 * budget_fraction)  # decay as budget consumed
        return min(max(meta, 0.5), 2.0)

    def reparameterize(self, m1: M1State, m2: M2State) -> None:
        """M3 can directly modify M1/M2 parameters."""
        if self.creative_multiplier > 1.5:
            m1.exploration_depth *= 1.05  # nudge M1 wider
        if self.efficiency_multiplier < 0.6:
            m2.cost_budget *= 0.95        # tighten M2


def compute_m_scalars(
    m1: M1State,
    m2: M2State,
    m3: M3State,
    stimulus: float = 1.0,
) -> dict:
    """
    Single tick: compute all three M scalars.
    Returns dict ready for Triangulation input.
    """
    m1_val = m1.step(stimulus)
    m2_val = m2.observe(m1)
    m3_val = m3.observe(m1, m2_val)
    m3.reparameterize(m1, m2)
    return {
        "M1": m1_val,
        "M2": m2_val,
        "M3": m3_val,
    }
