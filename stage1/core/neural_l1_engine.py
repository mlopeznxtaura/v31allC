"""stage1/core/neural_l1_engine.py — NextAura Neural Layer-1 (L1) Engine.
Integrates the genuine PyTorch Layer-1 model (M1, M2, M3) from weights checkpoints,
allowing the neural network to "live within the architecture" as requested.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Root directory helper
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_L1_CHECKPOINT = ROOT_DIR.parent / "Pt's" / "layer1_combined.pt"

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) matching the original L1/L2 training architecture.
    """
    def __init__(self, dim: int, max_seq_len: int = 64):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(2)
        if seq_len > self.cos_cached.size(2):
            self._update_cache(seq_len, q.device)
            
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot

    def _update_cache(self, seq_len: int, device: torch.device):
        dim = self.cos_cached.size(-1)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class GADCBlock(nn.Module):
    """
    GADC Attention Block matching the L1 training configuration.
    Performs multi-head attention with RoPE and a gated expert routing MLP.
    """
    def __init__(self, d_model: int = 32, n_heads: int = 2, d_ff: int = 64, chunk_size: int = 64, max_seq_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        device = x.device

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)

        x = self.ln1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)

        if seq_len <= self.chunk_size:
            chunk_mean = x.mean(dim=1, keepdim=True)
        else:
            pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
            if pad_len > 0:
                x_padded = F.pad(x, (0, 0, 0, pad_len))
            else:
                x_padded = x
            num_chunks = x_padded.size(1) // self.chunk_size
            chunks = x_padded.view(batch, num_chunks, self.chunk_size, self.d_model)
            chunk_mean = chunks.mean(dim=2)

        gate = self.gate_mlp(chunk_mean)

        if seq_len <= self.chunk_size:
            gate = gate.squeeze(1)

        return x, gate


class TinyTriadicTransformer(nn.Module):
    """
    Complete L1 Model representing the Seed Crystal network.
    Maps 4D inputs (mu_0) to predictions and structural/coherence gate metrics.
    """
    def __init__(self, syn_vocab_size: int = 3, d_model: int = 32, n_heads: int = 2, d_ff: int = 64, chunk_size: int = 64, max_seq_len: int = 64):
        super().__init__()
        self.fuse_proj = nn.Linear(4, d_model)
        self.gadc_block = GADCBlock(d_model, n_heads, d_ff, chunk_size, max_seq_len)
        self.syn_head = nn.Linear(d_model, syn_vocab_size)
        self.geo_head = nn.Linear(d_model, 1)
        self.boundary_head = nn.Linear(3, 2)

    def forward(self, mu_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.fuse_proj(mu_0)
        hidden, gate = self.gadc_block(x)
        
        # Output predictions (aligned with shift in training)
        seq_len = hidden.size(1)
        if seq_len > 1:
            syn_logits = self.syn_head(hidden[:, :-1, :])
            geo_pred = self.geo_head(hidden[:, :-1, :]).squeeze(-1)
        else:
            syn_logits = self.syn_head(hidden)
            geo_pred = self.geo_head(hidden).squeeze(-1)
            
        boundary_logits = self.boundary_head(gate)
        return syn_logits, geo_pred, gate, boundary_logits


class NeuralL1State:
    """
    A stateful wrapper that hosts the genuine PyTorch Layer-1 model.
    Maintains a local sequence history of mu_0 vectors and feeds them to GADC
    to determine live, deep-learning based M1, M2, and M3 values.
    """
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.model = TinyTriadicTransformer()
        self.is_loaded = False
        self._history: List[List[float]] = []
        self.checkpoint_path = ""
        
        # Load weights
        ckpt = checkpoint_path or str(DEFAULT_L1_CHECKPOINT)
        self.load_weights(ckpt)

    def load_weights(self, checkpoint_path: str) -> bool:
        """
        Loads the model's state dictionary from the PyTorch .pt checkpoint.
        Checks both Windows and WSL relative paths.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            # Try to resolve relative to root or parent
            wsl_path = Path(str(checkpoint_path).replace("D:", "/mnt/d"))
            if wsl_path.exists():
                path = wsl_path
            else:
                # Try fallback in project root parent
                alt_path = ROOT_DIR.parent / "Pt's" / Path(checkpoint_path).name
                if alt_path.exists():
                    path = alt_path

        if not path.exists():
            print(f"[NeuralL1] Weights file not found: {checkpoint_path}")
            self.is_loaded = False
            return False

        try:
            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
            # Extract l1 namespace if loading from an L2 model checkpoint (l2_baby_grounded_v3.pt)
            l1_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("l1."):
                    l1_state_dict[k[3:]] = v
                else:
                    l1_state_dict[k] = v

            # Filter state dict keys to only what the TinyTriadicTransformer expects
            filtered_state_dict = {}
            expected_keys = self.model.state_dict().keys()
            for k, v in l1_state_dict.items():
                if k in expected_keys:
                    filtered_state_dict[k] = v

            self.model.load_state_dict(filtered_state_dict, strict=False)
            self.model.eval()
            self.checkpoint_path = str(path)
            self.is_loaded = True
            print(f"[NeuralL1] Successfully loaded Layer-1 weights from {path.name}")
            return True
        except Exception as e:
            print(f"[NeuralL1] Error loading weights from {path.name}: {e}")
            self.is_loaded = False
            return False

    def reset(self):
        """Clears the sequence history of interaction steps."""
        self._history.clear()

    def step(
        self,
        progress: float,
        language_scalar: float,
        binary_scalar: float,
        stimulus: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Appends the current step inputs as a 4D mu_0 vector,
        evaluates the L1 PyTorch network, and extracts [M1, M2, M3] from the model gate.
        
        Returns:
            (M1, M2, M3) as float scalars.
        """
        # Construct mu_0 = [progress, language, binary, stimulus/noise]
        # Match dimensions used in L1 prechunked data format
        noise_term = (random.random() - 0.5) * 0.05
        mu_t = [
            float(progress),
            float(language_scalar),
            float(binary_scalar),
            float(stimulus + noise_term)
        ]
        
        self._history.append(mu_t)
        
        # Enforce sliding window context to prevent infinite attention sequence growth
        # and match the training max_seq_len (64)
        context_window = self._history[-64:]
        
        if not self.is_loaded:
            # Fallback to simulated scalars if weights are not yet loaded
            # Use deterministic continuous functions based on inputs to simulate neural response
            m1 = progress * 1.5 + stimulus
            m2 = max(0.0, 1.0 - (progress * 0.8))
            m3 = 0.5 + min(1.5, language_scalar * 1.2 + binary_scalar * 0.3)
            return m1, m2, m3

        try:
            with torch.no_grad():
                # Shape: (1, seq_len, 4)
                mu_tensor = torch.tensor([context_window], dtype=torch.float32)
                # Forward pass
                _, _, gate, _ = self.model(mu_tensor)
                
                # Gate shape is (1, 3) because seq_len <= 64 (chunk_size)
                gate_vector = gate.squeeze(0).cpu().numpy()
                
                # Extract meta-scalars
                # M1: Unbounded thinker exploration vector
                m1 = float(gate_vector[0])
                # M2: Efficiency score [0.0, 1.0] (clamp for safety)
                m2 = min(max(float(gate_vector[1]), 0.0), 1.0)
                # M3: Meta multiplier, typical range [0.5, 2.0]
                m3 = min(max(float(gate_vector[2]), 0.5), 2.0)
                
                return m1, m2, m3
        except Exception as e:
            print(f"[NeuralL1] Inference error: {e}. Falling back to default.")
            # Safety fallback
            return 1.0, 0.5, 1.0
