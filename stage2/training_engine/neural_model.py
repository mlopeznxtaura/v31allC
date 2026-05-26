"""stage2/training_engine/neural_model.py — V31all PyTorch Foundation
Defines the neural network components for the v31 intelligence model:
- V31Model: concatenates 7 inputs (227 dims) → Linear(256) → MoELayer (4 experts) → next_frame (224 dims) & token_logits (5000 dims)
- MoELayer & MoERouter: tau-gated Mixture of Experts block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    """
    tau-gated MoE Router.
    Adjusts expert count or routing weight sharpness based on tau (coherence invariant).
    """
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # tau is the coherence invariant [0, 1]
        # In case tau is scalar or batch mismatches, match its dimensions to gate logits
        logits = self.gate(x)
        
        # When tau is high, sharpen the distribution (temp is low)
        # When tau is low, broaden the distribution (temp is high)
        # Handle division by zero via eps=1e-6
        temp = 1.0 / (tau + 1e-6)
        
        # Temperature scale the gating logits
        scaled_logits = logits * temp
        probs = F.softmax(scaled_logits, dim=-1)
        return probs


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer with weighted expert summation.
    """
    def __init__(self, input_dim: int, output_dim: int, num_experts: int):
        super().__init__()
        self.router = MoERouter(input_dim, num_experts)
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # Get routing weights: shape (batch, num_experts)
        probs = self.router(x, tau)
        
        # Compute expert outputs: shape list of (batch, output_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # (batch, num_experts, output_dim)
        
        # Weighted sum: multiply probabilities by expert outputs
        output = torch.sum(probs.unsqueeze(-1) * expert_outputs, dim=1)
        return output


class V31Model(nn.Module):
    """
    V31 Deep Learning Neural Network.
    Inputs:
        m1: (batch, 1) or scalar
        m2: (batch, 1) or scalar
        m3: (batch, 1) or scalar
        geo: (batch, 32)
        bin: (batch, 32)
        lng: (batch, 128)
        tri: (batch, 32) (or calculated triangulation tensor)
    Total concatenated input dimension: 1 + 1 + 1 + 32 + 32 + 128 + 32 = 227
    """
    def __init__(self, vocab_size: int = 5000, num_experts: int = 4):
        super().__init__()
        # Input: 227-dimensional concatenation
        self.input_proj = nn.Linear(227, 256)
        self.moe = MoELayer(256, 256, num_experts=num_experts)
        
        # Heads
        # next_frame_head: maps to 224 pixels (14x16 synthesized visual grid)
        self.next_frame_head = nn.Linear(256, 224)
        # token_head: maps to vocabulary word logits (5000 choices)
        self.token_head = nn.Linear(256, vocab_size)

    def forward(self, m1, m2, m3, geo, bin, lng, tri, tau=None):
        # Coerce inputs to torch float tensors & unsqueeze batches if single dimension
        def _ensure_tensor(t, expected_dim):
            t = torch.as_tensor(t, dtype=torch.float32)
            if t.dim() == 0:
                t = t.unsqueeze(0).unsqueeze(0) # (1, 1)
            elif t.dim() == 1:
                # If it's a batch sequence or a single vector
                if t.shape[0] == expected_dim:
                    t = t.unsqueeze(0) # (1, expected_dim)
                else:
                    t = t.unsqueeze(-1) # (batch, 1)
            return t

        m1 = _ensure_tensor(m1, 1)
        m2 = _ensure_tensor(m2, 1)
        m3 = _ensure_tensor(m3, 1)
        geo = _ensure_tensor(geo, 32)
        bin = _ensure_tensor(bin, 32)
        lng = _ensure_tensor(lng, 128)
        tri = _ensure_tensor(tri, 32)

        # Ensure correct hidden features dimensions
        batch_size = m1.shape[0]
        # Rescale shape of scalar inputs to (batch, 1)
        m1 = m1.view(batch_size, 1)
        m2 = m2.view(batch_size, 1)
        m3 = m3.view(batch_size, 1)
        geo = geo.view(batch_size, 32)
        bin = bin.view(batch_size, 32)
        lng = lng.view(batch_size, 128)
        tri = tri.view(batch_size, 32)

        # Concatenate inputs: (batch, 227)
        x = torch.cat([m1, m2, m3, geo, bin, lng, tri], dim=-1)
        h = F.relu(self.input_proj(x))
        
        # Robustly calculate tau if not supplied
        # In v31 schema, tau is the coherence metric: we map it as the mean of triangulation tensor, clamped to [0.0, 1.0]
        if tau is None:
            tau = torch.clamp(tri.mean(dim=-1, keepdim=True), 0.0, 1.0)
        else:
            tau = torch.as_tensor(tau, dtype=torch.float32).view(batch_size, 1)

        # Execute MoE gating with tau
        h_moe = self.moe(h, tau)
        
        # Output predictions
        next_frame = self.next_frame_head(h_moe)
        token_logits = self.token_head(h_moe)
        
        return next_frame, token_logits
