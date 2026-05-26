"""stage1/core/layer2_scalars.py - V31all
Layer-2 scalars: Binary, Geometry, Language.
GPU-accelerated batch path for large ingests (torch CUDA).
CPU fallback for single records / no-GPU environments.
"""

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Optional, List

# GPU setup - soft import, CPU fallback if unavailable
try:
    import torch
    _CUDA = torch.cuda.is_available()
    _DEVICE = torch.device("cuda" if _CUDA else "cpu")
    if _CUDA:
        print(f"[layer2] GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    torch = None
    _CUDA = False
    _DEVICE = None


@dataclass
class BinaryScalar:
    codebase_hash: str = ""
    mission: str = ""
    purpose: str = ""
    is_logical: bool = True

    def compute(self) -> float:
        h = hashlib.sha256((self.codebase_hash + self.mission).encode()).hexdigest()
        entropy = len(set(h)) / 16.0
        mission_score = min(len(self.mission) / 200.0, 1.0)
        logic_bonus = 1.0 if self.is_logical else 0.3
        raw = (entropy * 0.4 + mission_score * 0.3 + logic_bonus * 0.3)
        return min(max(raw, 0.0), 1.0)

    def to_field(self) -> str:
        score = self.compute()
        status = "logical" if self.is_logical else "incoherent"
        return (f"Phase gate {status}: hash={self.codebase_hash[:8]}, "
                f"mission='{self.mission[:60]}', score={score:.3f}")


@dataclass
class GeometryScalar:
    entry_index: int = 0
    total_entries: int = 50
    sandbox: bool = True
    system_id: str = "CASEBELIZE"
    telemetry_stable: bool = True
    _init_time: float = field(default_factory=time.time)

    def compute(self) -> float:
        progress = self.entry_index / max(self.total_entries, 1)
        stability = 1.0 if self.telemetry_stable else 0.5
        live_bonus = 0.0 if self.sandbox else 0.15
        return min(progress * stability + live_bonus, 1.0)

    def to_field(self) -> str:
        env = "Sandbox" if self.sandbox else "Live"
        elapsed = round(time.time() - self._init_time, 1)
        return (f"Entry {self.entry_index}/{self.total_entries}. "
                f"Env={env}. SystemID={self.system_id}. "
                f"TelemetryStable={self.telemetry_stable}. Elapsed={elapsed}s.")


@dataclass
class LanguageScalar:
    text: str = ""
    _vocab_size: int = 32768

    def compute(self) -> float:
        if not self.text:
            return 0.0
        words = self.text.split()
        unique = len(set(w.lower() for w in words))
        length_factor = math.log1p(len(words)) / math.log1p(500)
        density = unique / max(len(words), 1)
        return min(density * length_factor, 1.0)

    def to_field(self) -> str:
        return self.text


# ------------------------------------------------------------------ #
#  GPU batch path - vectorised language scalar for large ingests      #
# ------------------------------------------------------------------ #

def batch_language_scalars(texts: List[str]) -> List[float]:
    """
    Compute language_scalar for a batch of texts.
    Uses GPU tensor ops when CUDA available (>= 20 records).
    Falls back to CPU loop for small batches or no GPU.
    """
    if not texts:
        return []

    # CPU path for small batches - not worth GPU overhead
    if not _CUDA or torch is None or len(texts) < 20:
        return [LanguageScalar(text=t).compute() for t in texts]

    # GPU path: vectorise word count + unique count approximation
    # Encode each text as word-count and unique-word-count tensors
    word_counts = []
    unique_counts = []
    for t in texts:
        if not t:
            word_counts.append(0)
            unique_counts.append(0)
        else:
            words = t.lower().split()
            word_counts.append(len(words))
            unique_counts.append(len(set(words)))

    wc = torch.tensor(word_counts, dtype=torch.float32, device=_DEVICE)
    uc = torch.tensor(unique_counts, dtype=torch.float32, device=_DEVICE)

    # Vectorised formula matching LanguageScalar.compute()
    log500 = math.log1p(500)
    length_factor = torch.log1p(wc) / log500
    density = uc / torch.clamp(wc, min=1.0)
    scores = torch.clamp(density * length_factor, 0.0, 1.0)

    return scores.cpu().tolist()


def build_layer2(
    binary_hash: str,
    mission: str,
    purpose: str,
    entry_index: int,
    total_entries: int,
    language_text: str,
    sandbox: bool = True,
    system_id: str = "CASEBELIZE",
    is_logical: bool = True,
    telemetry_stable: bool = True,
) -> dict:
    b = BinaryScalar(
        codebase_hash=binary_hash,
        mission=mission,
        purpose=purpose,
        is_logical=is_logical,
    )
    g = GeometryScalar(
        entry_index=entry_index,
        total_entries=total_entries,
        sandbox=sandbox,
        system_id=system_id,
        telemetry_stable=telemetry_stable,
    )
    l = LanguageScalar(text=language_text)

    return {
        "binary_scalar": b.compute(),
        "geometry_scalar": g.compute(),
        "language_scalar": l.compute(),
        "binary_field": b.to_field(),
        "geometry_field": g.to_field(),
        "language_field": l.to_field(),
        "_objects": {"binary": b, "geometry": g, "language": l},
    }