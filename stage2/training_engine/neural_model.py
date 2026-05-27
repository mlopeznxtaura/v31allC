"""stage2/training_engine/neural_model.py — V31all PyTorch Foundation v2

Architecture upgrade from v1:
  v1: 227 → Linear(256) → MoE(4 experts, 256) → [224, 5000]
  v2: 227 → InputProjection(512) → 3x ResidualMoEBlock(512, 8 experts)
         → ContextGRU(512) → LayerNorm → heads([224, 5000])

Key improvements:
  - Wider base dim: 256 → 512
  - 8 experts per MoE block (was 4)
  - 3 stacked ResidualMoEBlocks with LayerNorm + Dropout
  - GRU sequence mixer after MoE stack for temporal coherence
  - Deeper output heads (2-layer MLP each)
  - Proper weight init (Kaiming for relu paths)
  - tau gating sharpened: inverse-softplus instead of raw 1/tau
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM  = 227   # m1(1)+m2(1)+m3(1)+geo(32)+bin(32)+lng(128)+tri(32)
BASE_DIM   = 512
NUM_EXPERTS = 8
NUM_BLOCKS  = 4
DROPOUT     = 0.08
VOCAB_SIZE  = 5000
FRAME_SIZE  = 224


# ── Router ────────────────────────────────────────────────────────────────────

class MoERouter(nn.Module):
    """tau-gated sparse router. Top-2 expert selection with load balancing."""
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor, tau: torch.Tensor):
        logits = self.gate(x)                          # (B, E)
        # Sharpen with tau via softplus: higher tau → sharper selection
        temp = 1.0 / (F.softplus(tau) + 1e-4)         # (B, 1)
        scaled = logits * temp
        probs  = F.softmax(scaled, dim=-1)             # (B, E)

        # Top-2 sparse mask
        top2 = probs.topk(2, dim=-1).indices           # (B, 2)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, top2, 1.0)
        sparse = probs * mask
        sparse = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)  # renorm
        return sparse                                  # (B, E)


class MoELayer(nn.Module):
    """Sparse top-2 Mixture of Experts."""
    def __init__(self, dim: int, num_experts: int, dropout: float = DROPOUT):
        super().__init__()
        self.router  = MoERouter(dim, num_experts)
        # Each expert: 2-layer FFN with hidden=2*dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        probs = self.router(x, tau)                                    # (B, E)
        outs  = torch.stack([e(x) for e in self.experts], dim=1)      # (B, E, D)
        return torch.sum(probs.unsqueeze(-1) * outs, dim=1)            # (B, D)


# ── Residual Block ────────────────────────────────────────────────────────────

class ResidualMoEBlock(nn.Module):
    """LayerNorm → MoE → Dropout + residual."""
    def __init__(self, dim: int, num_experts: int, dropout: float = DROPOUT):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.moe   = MoELayer(dim, num_experts, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.moe(self.norm(x), tau))


# ── Main Model ────────────────────────────────────────────────────────────────

class V31Model(nn.Module):
    """
    V31 Deep Observer Network v2.

    Input dims:  m1(1) m2(1) m3(1) geo(32) bin(32) lng(128) tri(32) = 227
    Output dims: next_frame(224)  token_logits(5000)

    Pipeline:
      input → InputProj(512) → 3×ResidualMoEBlock(512, 8exp)
            → GRU(512→512) → LN → [FrameHead, TokenHead]
    """

    def __init__(
        self,
        vocab_size:  int = VOCAB_SIZE,
        num_experts: int = NUM_EXPERTS,
        num_blocks:  int = NUM_BLOCKS,
        base_dim:    int = BASE_DIM,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.base_dim = base_dim

        # Input projection with LN
        self.input_proj = nn.Sequential(
            nn.Linear(INPUT_DIM, base_dim),
            nn.LayerNorm(base_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # MoE residual stack
        self.blocks = nn.ModuleList([
            ResidualMoEBlock(base_dim, num_experts, dropout)
            for _ in range(num_blocks)
        ])

        # Temporal mixer: treats the block stack depth as a sequence
        self.gru = nn.GRU(base_dim, base_dim, batch_first=True, num_layers=2,
                          dropout=dropout if num_blocks > 1 else 0.0)
        self.post_gru_norm = nn.LayerNorm(base_dim)

        # Output heads (2-layer MLP each)
        self.frame_head = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_dim // 2, FRAME_SIZE),
        )
        self.token_head = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_dim, vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, m1, m2, m3, geo, bin, lng, tri, tau=None):
        def _t(t, d):
            t = torch.as_tensor(t, dtype=torch.float32)
            if t.dim() == 0: t = t.unsqueeze(0).unsqueeze(0)
            elif t.dim() == 1:
                t = t.unsqueeze(0) if t.shape[0] == d else t.unsqueeze(-1)
            return t

        m1  = _t(m1,  1);  m2  = _t(m2,  1);  m3  = _t(m3,  1)
        geo = _t(geo, 32); bin = _t(bin, 32); lng = _t(lng, 128); tri = _t(tri, 32)

        B = m1.shape[0]
        m1  = m1.view(B, 1);    m2  = m2.view(B, 1);   m3  = m3.view(B, 1)
        geo = geo.view(B, 32);  bin = bin.view(B, 32);  lng = lng.view(B, 128)
        tri = tri.view(B, 32)

        # Log-scale m1 (unbounded explorer)
        m1 = m1.sign() * torch.log1p(m1.abs()) / 10.0

        # Tau: coherence signal from triangulation mean
        if tau is None:
            tau = torch.clamp(tri.mean(dim=-1, keepdim=True), 0.0, 1.0)  # (B,1)
        else:
            tau = torch.as_tensor(tau, dtype=torch.float32).view(B, 1)

        x = torch.cat([m1, m2, m3, geo, bin, lng, tri], dim=-1)  # (B, 227)
        h = self.input_proj(x)                                     # (B, 512)

        # Collect each block's output as a sequence for GRU
        seq = []
        for block in self.blocks:
            h = block(h, tau)
            seq.append(h.unsqueeze(1))                             # (B,1,512)

        seq_tensor = torch.cat(seq, dim=1)                         # (B, num_blocks, 512)
        gru_out, _ = self.gru(seq_tensor)                          # (B, num_blocks, 512)
        h_final    = self.post_gru_norm(gru_out[:, -1, :])         # (B, 512) — last step

        return self.frame_head(h_final), self.token_head(h_final)


# ── Parameter count utility ───────────────────────────────────────────────────

def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  Trainable: {trainable:,}"


if __name__ == "__main__":
    m = V31Model()
    print(f"V31Model v2  {count_params(m)}")
    B = 4
    out_f, out_t = m(
        torch.randn(B),  torch.randn(B),  torch.randn(B),
        torch.randn(B, 32), torch.randn(B, 32), torch.randn(B, 128), torch.randn(B, 32)
    )
    print(f"frame={out_f.shape}  token_logits={out_t.shape}")
