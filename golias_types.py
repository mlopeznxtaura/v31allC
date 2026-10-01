"""golias_types.py — IF struct definitions for Golias architecture
Implements the 7 Input Functions from the formal spec.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class IF1_CreativeExploration:
    """if_1 = M1(x_t, theta_1, T_max) — unbounded creative exploration."""
    x_t: Dict[str, Any] = field(default_factory=dict)
    theta_1: Dict[str, Any] = field(default_factory=lambda: {"temperature": 1.4, "max_tokens": 512})
    T_max: float = 1.4
    output: str = ""
    exploration_vector: float = 0.0  # M1 scalar output


@dataclass
class IF2_EfficiencyGate:
    """if_2 = M2(if1, theta_2, tau) — efficiency zealot."""
    if1_ref: Optional[IF1_CreativeExploration] = None
    theta_2: Dict[str, Any] = field(default_factory=lambda: {"epsilon": 0.6})
    tau_ms: float = 5000.0
    epsilon: float = 0.6
    output: str = ""
    halt_triggered: bool = False
    compute_cost: float = 0.0
    latency_ms: float = 0.0
    efficiency_score: float = 1.0  # M2 scalar output


@dataclass
class IF3_DirectionValidator:
    """if_3 = M3(if1, if2, theta_3) — direction validator + adaptive retuner."""
    if1_ref: Optional[IF1_CreativeExploration] = None
    if2_ref: Optional[IF2_EfficiencyGate] = None
    theta_3: Dict[str, Any] = field(default_factory=lambda: {"delta_novelty": 0.1, "delta_coverage": 0.05})
    theta_1_prime: Dict[str, Any] = field(default_factory=dict)
    theta_2_prime: Dict[str, Any] = field(default_factory=dict)
    D_orthogonal: float = 0.0
    G_p: float = 0.0
    output: str = ""
    meta_multiplier: float = 1.0  # M3 scalar output


@dataclass
class IF4_BinarySubstrate:
    """if_4 = B(S_t) — raw substrate/hardware/runtime state."""
    hardware_state: str = ""
    runtime_constraints: str = ""
    memory_available_gb: float = 0.0
    execution_permissions: str = ""
    system_identity: str = ""
    deployment_boundaries: str = ""
    binary_scalar: float = 0.0  # encoded scalar

    def encode(self) -> float:
        """Encode substrate state to [0,1] scalar."""
        parts = [self.hardware_state, self.runtime_constraints,
                 self.system_identity, self.deployment_boundaries]
        text = " ".join(p for p in parts if p)
        if not text:
            return 0.5
        h = sum(ord(c) for c in text[:256])
        self.binary_scalar = (h % 10000) / 10000.0
        return self.binary_scalar


@dataclass
class IF5_GeometryTelemetry:
    """if_5 = G(E_t) — live environment telemetry."""
    spatial_position: str = ""
    temporal_state: str = ""
    network_state: str = ""
    radio_frequency_state: str = ""
    sync_state: str = ""
    external_topology: str = ""
    geometry_scalar: float = 0.0

    def encode(self) -> float:
        text = " ".join(p for p in [self.spatial_position, self.temporal_state,
                                     self.network_state, self.sync_state] if p)
        if not text:
            return 0.5
        h = sum((i+1)*ord(c) for i, c in enumerate(text[:256]))
        self.geometry_scalar = (h % 10000) / 10000.0
        return self.geometry_scalar


@dataclass
class IF6_LanguageFusion:
    """if_6 = L(if1..if5) — multimodal fusion, rotating chunk scheduler."""
    CHUNK_SIZES = [800, 900, 1000, 1100, 1200]
    _chunk_idx: int = 0
    fused_text: str = ""
    language_scalar: float = 0.0

    def fuse(self, if1: IF1_CreativeExploration, if2: IF2_EfficiencyGate,
             if3: IF3_DirectionValidator, if4: IF4_BinarySubstrate,
             if5: IF5_GeometryTelemetry) -> str:
        chunk_size = self.CHUNK_SIZES[self._chunk_idx % len(self.CHUNK_SIZES)]
        self._chunk_idx += 1
        parts = [
            f"[M1_explore] {if1.output[:chunk_size//5]}",
            f"[M2_efficiency] {if2.output[:chunk_size//6]}",
            f"[M3_direction] {if3.output[:chunk_size//6]}",
            f"[binary] hw={if4.hardware_state[:60]} sys={if4.system_identity[:40]}",
            f"[geometry] pos={if5.spatial_position[:60]} net={if5.network_state[:40]}",
        ]
        self.fused_text = " | ".join(p for p in parts if p.split("] ")[1].strip())[:chunk_size]
        h = sum(ord(c) for c in self.fused_text[:512])
        self.language_scalar = (h % 10000) / 10000.0
        return self.fused_text


@dataclass
class IF7_RelationalArbitration:
    """if_7 = Rel(Delta_if1..Delta_if6) — invariant preservation + conflict arbitration."""
    # Deltas
    delta_if1: float = 0.0
    delta_if2: float = 0.0
    delta_if3: float = 0.0
    delta_if4: float = 0.0
    delta_if5: float = 0.0
    delta_if6: float = 0.0

    # Outputs
    invariant_score: float = 1.0
    conflict_weight: float = 0.0
    causal_weight: float = 0.0
    policy_fusion_scalar: float = 0.0
    uncertainty: float = 0.0
    coherence: float = 1.0

    # Stability loss = ||Delta_if7|| (from spec)
    stability_loss: float = 0.0

    def compute(self,
                prev_if1: float, curr_if1: float,
                prev_if2: float, curr_if2: float,
                prev_if3: float, curr_if3: float,
                prev_if4: float, curr_if4: float,
                prev_if5: float, curr_if5: float,
                prev_if6: float, curr_if6: float) -> float:
        """Relational arbitration over deltas. Returns policy_fusion_scalar."""
        self.delta_if1 = curr_if1 - prev_if1
        self.delta_if2 = curr_if2 - prev_if2
        self.delta_if3 = curr_if3 - prev_if3
        self.delta_if4 = curr_if4 - prev_if4
        self.delta_if5 = curr_if5 - prev_if5
        self.delta_if6 = curr_if6 - prev_if6

        deltas = [self.delta_if1, self.delta_if2, self.delta_if3,
                  self.delta_if4, self.delta_if5, self.delta_if6]

        # Stability loss = L2 norm of delta vector
        self.stability_loss = math.sqrt(sum(d**2 for d in deltas))

        # Conflict = variance of deltas (high variance = disagreement)
        mean_d = sum(deltas) / len(deltas)
        self.conflict_weight = math.sqrt(sum((d - mean_d)**2 for d in deltas) / len(deltas))

        # Causal weight = magnitude of M1 delta (exploration drive)
        self.causal_weight = abs(self.delta_if1)

        # Uncertainty = normalized conflict
        self.uncertainty = min(self.conflict_weight / (abs(mean_d) + 1e-8), 1.0)

        # Coherence = 1 - uncertainty
        self.coherence = 1.0 - min(self.uncertainty, 1.0)

        # Invariant score: penalize if any delta exceeds threshold
        max_delta = max(abs(d) for d in deltas)
        self.invariant_score = math.exp(-max_delta)

        # Policy fusion: weighted combination
        self.policy_fusion_scalar = (
            0.30 * self.coherence +
            0.25 * self.invariant_score +
            0.20 * (1.0 - self.uncertainty) +
            0.15 * min(abs(self.causal_weight), 1.0) +
            0.10 * (1.0 - min(self.conflict_weight, 1.0))
        )
        return self.policy_fusion_scalar


@dataclass
class GoliasSample:
    """Full 7-IF sample with outputs and loss components."""
    if1: IF1_CreativeExploration = field(default_factory=IF1_CreativeExploration)
    if2: IF2_EfficiencyGate = field(default_factory=IF2_EfficiencyGate)
    if3: IF3_DirectionValidator = field(default_factory=IF3_DirectionValidator)
    if4: IF4_BinarySubstrate = field(default_factory=IF4_BinarySubstrate)
    if5: IF5_GeometryTelemetry = field(default_factory=IF5_GeometryTelemetry)
    if6: IF6_LanguageFusion = field(default_factory=IF6_LanguageFusion)
    if7: IF7_RelationalArbitration = field(default_factory=IF7_RelationalArbitration)

    of1_prediction: str = ""    # next_frame_prediction
    of2_explanation: str = ""   # language explanation

    # Golias loss components
    L_coherence: float = 0.0
    L_efficiency: float = 0.0
    L_novelty: float = 0.0
    L_stability: float = 0.0
    L_alignment: float = 0.0
    L_prediction: float = 0.0
    L_total: float = 0.0

    # Lambda weights
    LAMBDAS = [0.20, 0.15, 0.20, 0.20, 0.15, 0.10]  # coherence, eff, novelty, stability, align, pred

    def compute_loss(self, Y_target: str = "", Y_future_scalar: float = 0.5):
        """Compute full Golias loss."""
        # L_coherence = 1 - Sim(of1, of2)
        sim = len(set(self.of1_prediction.split()) & set(self.of2_explanation.split())) / \
              max(len(set(self.of1_prediction.split()) | set(self.of2_explanation.split())), 1)
        self.L_coherence = 1.0 - sim

        # L_efficiency = compute_cost + latency_proxy
        self.L_efficiency = min(self.if2.compute_cost + self.if2.latency_ms / 10000.0, 1.0)

        # L_novelty = -delta_semantic_distance (negative: rewards diversity)
        self.L_novelty = -self.if3.G_p  # G_p = max(delta_novelty, delta_coverage)

        # L_stability = ||Delta_if7||
        self.L_stability = self.if7.stability_loss

        # L_alignment = 1 - GoalMatch(of2, Y)
        if Y_target:
            match = len(set(self.of2_explanation.lower().split()) &
                       set(Y_target.lower().split())) / \
                    max(len(set(Y_target.lower().split())), 1)
            self.L_alignment = 1.0 - match
        else:
            self.L_alignment = 0.5

        # L_prediction = ||of1 - Y_future||
        of1_scalar = len(self.of1_prediction) / 1000.0
        self.L_prediction = abs(of1_scalar - Y_future_scalar)

        lams = self.LAMBDAS
        self.L_total = (lams[0]*self.L_coherence + lams[1]*self.L_efficiency +
                        lams[2]*self.L_novelty + lams[3]*self.L_stability +
                        lams[4]*self.L_alignment + lams[5]*self.L_prediction)
        return self.L_total
