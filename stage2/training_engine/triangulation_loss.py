"""stage2/training_engine/triangulation_loss.py — V31all
Triangulation loss: replaces contrastive_loss.py from v26all.

In v31 the learning signal comes from:
  1. V-prediction loss: how well does computed V match target reward?
  2. Action alignment loss: does language_token_output match expert action?
  3. Invariant loss: is V(t) ≈ weighted_mean(V(t-1..t-k))?  (self-consistency)
  4. M3 budget loss: penalize M3 exceeding time budget fraction.

No domain classifier. No contrastive pairs. No cluster ratio.
The triangulation value function IS the domain signal.
"""

import math
from typing import List, Optional


def v_prediction_loss(predicted_v: float, target_reward: float) -> float:
    """MSE between predicted triangulation value and observed reward."""
    return (predicted_v - target_reward) ** 2


def action_alignment_loss(
    predicted_action: str,
    expert_action: str,
    action_vocab: List[str],
) -> float:
    """
    0.0 if exact match.
    Soft penalty based on vocab distance (index difference / vocab_size).
    """
    if predicted_action == expert_action:
        return 0.0
    vocab = action_vocab
    try:
        pi = vocab.index(predicted_action)
        ei = vocab.index(expert_action)
        return abs(pi - ei) / max(len(vocab) - 1, 1)
    except ValueError:
        return 1.0  # unknown token = max penalty


def invariant_loss(v_history: List[float], current_v: float, k: int = 5) -> float:
    """
    Self-consistency: V(t) should not deviate wildly from prior k values.
    Loss = (current_v - mean(history[-k:]))^2
    """
    if not v_history:
        return 0.0
    window = v_history[-k:]
    mean_v = sum(window) / len(window)
    return (current_v - mean_v) ** 2


def m3_budget_loss(
    m3_multiplier: float,
    elapsed_fraction: float,
    target_multiplier: float = 1.0,
) -> float:
    """
    Penalize M3 multiplier drift as time budget is consumed.
    As elapsed_fraction → 1.0, M3 should approach target_multiplier.
    Loss = elapsed_fraction * (m3_multiplier - target_multiplier)^2
    """
    return elapsed_fraction * (m3_multiplier - target_multiplier) ** 2


def total_triangulation_loss(
    predicted_v: float,
    target_reward: float,
    predicted_action: str,
    expert_action: str,
    action_vocab: List[str],
    v_history: List[float],
    m3_multiplier: float,
    elapsed_fraction: float,
    weights: Optional[dict] = None,
) -> dict:
    """
    Composite triangulation loss.
    weights keys: v_pred, action, invariant, m3_budget
    Returns dict with individual + total loss.
    """
    w = weights or {
        "v_pred": 1.0,
        "action": 0.8,
        "invariant": 0.5,
        "m3_budget": 0.3,
    }
    l_v = v_prediction_loss(predicted_v, target_reward)
    l_a = action_alignment_loss(predicted_action, expert_action, action_vocab)
    l_i = invariant_loss(v_history, predicted_v)
    l_m = m3_budget_loss(m3_multiplier, elapsed_fraction)

    total = (
        w["v_pred"] * l_v
        + w["action"] * l_a
        + w["invariant"] * l_i
        + w["m3_budget"] * l_m
    )

    return {
        "total": round(total, 6),
        "v_prediction": round(l_v, 6),
        "action_alignment": round(l_a, 6),
        "invariant": round(l_i, 6),
        "m3_budget": round(l_m, 6),
    }
