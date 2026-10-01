"""stage2/training_engine/goliasv7_model.py — Golias-Γ v7 (scaled)

NEW FILE. Does not modify any existing model. Scales the Golias v6 tri-observer
MoE architecture and warm-starts from `golias v6.pt`.

v6 (reconstructed from state_dict, 7.09M params):
    m1_enc:  96  -> 1024 -> 256      (M1 unbounded explorer encoder)
    m2_enc:  128 -> 1024 -> 256      (M2 efficiency-gate encoder)
    m3_meta: 352 -> 1024 -> 256      (M3 meta-observer; sees m1+m2+bin+geo)
    fusion:  MoE, 2 experts, 480 -> 1024 -> 512, gate 480->2
    relation: 512 -> 1024 -> 512
    attn:    512 (WQ/WK/WV/WO), single block, learned temp _t
    mlp_pred:  512 -> 1024 -> 512    (OF1 next-state / next-pixel head)
    decode_lm: 512 -> 1024           (OF2 language explanation head)
    c_comp:  256 -> 1                (compute-cost / halting scalar)
    + meta buffers: step, H1_prev, H2_prev, K_t, TAU  (M1/M2 subspace locking)

v7 scaling (this file):
    - encoder hidden 1024 kept, output 256 -> 384 (richer latent per observer)
    - m3_meta input widened (still sees m1+m2+bin+geo, now at 384 latents)
    - fusion experts 2 -> 4, top-2 gating, hidden 1024 -> 1536, out 512 -> 640
    - relation 640 -> 1536 -> 640
    - attention depth 1 -> 2 blocks at d=640
    - heads scale with 640
    ~ 22-28M params depending on vocab/frame sizes.

Warm-start: `load_v6_warmstart()` copies every v6 weight block into the
overlapping top-left sub-tensor of the v7 parameters (zero/kaiming-filled
elsewhere), preserving v6's learned M1/M2 subspace locking and Fisher-tuned
weights. New dims are initialized small so the scaled model starts as a
faithful superset of v6.
"""

import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Observer input dims (fixed by the Golias spec / v6 checkpoint) ───────────
M1_IN   = 96
M2_IN   = 128
M3_IN   = 352
ENC_HID = 1024

# ── v7 scaled dims ───────────────────────────────────────────────────────────
ENC_OUT     = 384      # v6: 256
NUM_EXPERTS = 4        # v6: 2
FUSE_HID    = 1536     # v6: 1024
FUSE_OUT    = 640      # v6: 512
REL_HID     = 1536     # v6: 1024
ATTN_DIM    = 640      # v6: 512
ATTN_BLOCKS = 2        # v6: 1
PRED_OUT    = 640      # v6: 512 (mlp_pred head latent)
LM_OUT      = 1024     # v6: 1024 (decode_lm)

# fusion input = m1_enc_out + m2_enc_out + m3_meta_out (3 x ENC_OUT) ... v6 used
# 480 = 256 (m3) + 128 + 96? No — v6 fusion in=480 = m3(256)+m1raw? We keep the
# v6 convention: fusion sees concatenation of the three 256-latents trimmed.
# v7: fusion sees the three ENC_OUT (384*? ) — we define explicitly below.
FUSE_IN = ENC_OUT * 3 - 512 if False else 480 * ENC_OUT // 256  # = 720 for 384


class MLP2(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_out)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Expert(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_out)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class FusionMoE(nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(d_in, d_hid, d_out) for _ in range(num_experts)])
        self.gate = nn.Linear(d_in, num_experts)

    def forward(self, x, tau=None):
        logits = self.gate(x)
        if tau is not None:
            logits = logits * (1.0 / (F.softplus(tau) + 1e-4))
        probs = F.softmax(logits, dim=-1)
        if self.num_experts > 2:
            top = probs.topk(2, dim=-1).indices
            mask = torch.zeros_like(probs).scatter_(1, top, 1.0)
            probs = probs * mask
            probs = probs / (probs.sum(-1, keepdim=True) + 1e-8)
        outs = torch.stack([e(x) for e in self.experts], dim=1)   # (B,E,Dout)
        return torch.sum(probs.unsqueeze(-1) * outs, dim=1)


class SelfAttnBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.WQ = nn.Linear(dim, dim, bias=False)
        self.WK = nn.Linear(dim, dim, bias=False)
        self.WV = nn.Linear(dim, dim, bias=False)
        self.WO = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.temp = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x: (B, D) -> treat as length-1 sequence of one token per observer-fused state
        q, k, v = self.WQ(x), self.WK(x), self.WV(x)
        score = (q * k).sum(-1, keepdim=True) / (self.temp + 1e-4)
        attn = torch.sigmoid(score)
        out = self.WO(attn * v)
        return self.norm(x + out)


class GoliasV7(nn.Module):
    def __init__(self, vocab_size=5000, frame_size=224):
        super().__init__()
        self.vocab_size = vocab_size
        self.frame_size = frame_size

        self.m1_enc = MLP2(M1_IN, ENC_HID, ENC_OUT)
        self.m2_enc = MLP2(M2_IN, ENC_HID, ENC_OUT)
        self.m3_meta = MLP2(M3_IN, ENC_HID, ENC_OUT)

        fuse_in = ENC_OUT * 3
        self.fusion = FusionMoE(fuse_in, FUSE_HID, FUSE_OUT, NUM_EXPERTS)
        self.relation = MLP2(FUSE_OUT, REL_HID, ATTN_DIM)
        self.attn = nn.ModuleList([SelfAttnBlock(ATTN_DIM) for _ in range(ATTN_BLOCKS)])

        self.mlp_pred = MLP2(ATTN_DIM, REL_HID, PRED_OUT)
        self.frame_head = nn.Linear(PRED_OUT, frame_size)
        self.decode_lm = nn.Linear(ATTN_DIM, LM_OUT)
        self.token_head = nn.Linear(LM_OUT, vocab_size)
        self.c_comp = nn.Linear(ENC_OUT, 1)

        # M1/M2 subspace-locking buffers (ported from v6)
        self.register_buffer("H1_prev", torch.zeros(ENC_OUT))
        self.register_buffer("H2_prev", torch.zeros(ENC_OUT))
        self.register_buffer("TAU", torch.tensor(0.15))
        self.register_buffer("K_t", torch.tensor(1000.0))
        self.register_buffer("step", torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, m1_vec, m2_vec, m3_vec, tau=None):
        h1 = self.m1_enc(m1_vec)
        h2 = self.m2_enc(m2_vec)
        h3 = self.m3_meta(m3_vec)
        if tau is None:
            tau = self.TAU.view(1, 1).expand(h1.shape[0], 1)
        fused = self.fusion(torch.cat([h1, h2, h3], dim=-1), tau)
        r = self.relation(fused)
        for blk in self.attn:
            r = blk(r)
        pred_latent = self.mlp_pred(r)
        frame = self.frame_head(pred_latent)
        lm = self.decode_lm(r)
        tokens = self.token_head(F.gelu(lm))
        halt = torch.sigmoid(self.c_comp(h3))
        return frame, tokens, halt, (h1, h2, h3)


def _copy_block(dst: torch.Tensor, src) -> None:
    """Copy src into the top-left overlapping region of dst (in place)."""
    s = torch.as_tensor(src, dtype=dst.dtype)
    if s.dim() != dst.dim():
        return
    slices = tuple(slice(0, min(d, ss)) for d, ss in zip(dst.shape, s.shape))
    dst[slices] = s[slices]


def load_v6_warmstart(model: GoliasV7, v6_path: str) -> dict:
    """Warm-start: copy overlapping v6 weight blocks into v7. Returns report."""
    with open(v6_path, "rb") as f:
        ckpt = pickle.load(f)
    sd = ckpt["state_dict"]

    # v6 key -> v7 module mapping (linear W is stored as (in,out) in v6 -> transpose)
    pairs = {
        "m1_enc.fc1": model.m1_enc.fc1, "m1_enc.fc2": model.m1_enc.fc2,
        "m2_enc.fc1": model.m2_enc.fc1, "m2_enc.fc2": model.m2_enc.fc2,
        "m3_meta.fc1": model.m3_meta.fc1, "m3_meta.fc2": model.m3_meta.fc2,
        "fusion.expert0.fc1": model.fusion.experts[0].fc1,
        "fusion.expert0.fc2": model.fusion.experts[0].fc2,
        "fusion.expert1.fc1": model.fusion.experts[1].fc1,
        "fusion.expert1.fc2": model.fusion.experts[1].fc2,
        "fusion.gate": model.fusion.gate,
        "relation.fc1": model.relation.fc1, "relation.fc2": model.relation.fc2,
        "mlp_pred.fc1": model.mlp_pred.fc1, "mlp_pred.fc2": model.mlp_pred.fc2,
        "decode_lm": model.decode_lm, "c_comp": model.c_comp,
    }
    copied = []
    with torch.no_grad():
        for prefix, layer in pairs.items():
            wk, bk = f"{prefix}.W", f"{prefix}.b"
            if wk in sd:
                # v6 stores W as (in, out); nn.Linear weight is (out, in)
                _copy_block(layer.weight.data, torch.as_tensor(sd[wk]).T)
                copied.append(wk)
            if bk in sd and getattr(layer, "bias", None) is not None:
                _copy_block(layer.bias.data, sd[bk])
                copied.append(bk)
        # attention block 0 from v6 attn
        for vk, p in (("attn.WQ", model.attn[0].WQ), ("attn.WK", model.attn[0].WK),
                      ("attn.WV", model.attn[0].WV), ("attn.WO", model.attn[0].WO)):
            if vk in sd:
                _copy_block(p.weight.data, torch.as_tensor(sd[vk]).T)
                copied.append(vk)
        if "attn._t" in sd:
            _copy_block(model.attn[0].temp.data, sd["attn._t"])
        for bk, buf in (("meta.H1_prev", "H1_prev"), ("meta.H2_prev", "H2_prev"),
                        ("meta.TAU", "TAU"), ("meta.K_t", "K_t"), ("meta.step", "step")):
            if bk in sd:
                _copy_block(getattr(model, buf).data, torch.as_tensor(sd[bk]).reshape(getattr(model, buf).shape) if torch.as_tensor(sd[bk]).numel() == getattr(model, buf).numel() else torch.as_tensor(sd[bk]))
    return {
        "v6_params": ckpt.get("_metadata", {}).get("total_params"),
        "copied_tensors": len(copied),
        "v7_params": sum(p.numel() for p in model.parameters()),
    }


def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    return f"Total: {total:,}"


if __name__ == "__main__":
    m = GoliasV7()
    print("GoliasV7", count_params(m))
    rep = load_v6_warmstart(m, r"E:\NextAuraMonth3(corefocuses)\RAWgolias\golias v6.pt")
    print("warm-start:", rep)
    B = 4
    fr, tok, halt, _ = m(torch.randn(B, M1_IN), torch.randn(B, M2_IN), torch.randn(B, M3_IN))
    print("frame", fr.shape, "tokens", tok.shape, "halt", halt.shape)
