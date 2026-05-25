"""stage1/core/layer2_scalars.py — V31all
Layer-2 scalars: Binary, Geometry, Language.
These are the contextual inputs that feed into Triangulation alongside M1/M2/M3.

Binary  — Raw machine initialization. Codebase hash, mission, purpose.
          Is it logical? (bool-weighted float)

Geometry — Where am I in space/time? Telemetry stable?
           Sandbox vs live? System ID? (positional/state float)

Language — Rich contextualization of Binary + Geometry with narrative.
           Full semantic string → compressed scalar + raw text field.
"""

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BinaryScalar:
    """
    Raw machine initialization scalar.
    Encodes: codebase_hash, mission_clarity, logical_coherence.
    Output: float [0, 1] where 1.0 = fully initialized, coherent, logical.
    """
    codebase_hash: str = ""
    mission: str = ""
    purpose: str = ""
    is_logical: bool = True

    def compute(self) -> float:
        """
        Hash entropy + mission presence + logical flag → scalar.
        High score = stable initialization.
        """
        # hash entropy: more unique hash = more initialized
        h = hashlib.sha256((self.codebase_hash + self.mission).encode()).hexdigest()
        entropy = len(set(h)) / 16.0  # [0,1]
        mission_score = min(len(self.mission) / 200.0, 1.0)
        logic_bonus = 1.0 if self.is_logical else 0.3
        raw = (entropy * 0.4 + mission_score * 0.3 + logic_bonus * 0.3)
        return min(max(raw, 0.0), 1.0)

    def to_field(self) -> str:
        """Human-readable field for JSONL record."""
        score = self.compute()
        status = "logical" if self.is_logical else "incoherent"
        return (f"Phase gate {status}: hash={self.codebase_hash[:8]}, "
                f"mission='{self.mission[:60]}', score={score:.3f}")


@dataclass
class GeometryScalar:
    """
    Positional/state scalar.
    Encodes: entry index, timestamp, sandbox_vs_live, system_id, telemetry_stable.
    Output: float — position in training space/time.
    """
    entry_index: int = 0
    total_entries: int = 50
    sandbox: bool = True
    system_id: str = "CASEBELIZE"
    telemetry_stable: bool = True
    _init_time: float = field(default_factory=time.time)

    def compute(self) -> float:
        """
        Progress fraction + stability bonus + sandbox penalty.
        """
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
    """
    Rich contextual narrative scalar.
    Encodes: semantic richness of binary + geometry combination.
    Output: float (narrative density) + raw text.
    """
    text: str = ""
    _vocab_size: int = 32768

    def compute(self) -> float:
        """
        Unique token density × length factor → scalar.
        """
        if not self.text:
            return 0.0
        words = self.text.split()
        unique = len(set(w.lower() for w in words))
        length_factor = math.log1p(len(words)) / math.log1p(500)
        density = unique / max(len(words), 1)
        return min(density * length_factor, 1.0)

    def to_field(self) -> str:
        return self.text


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
    """
    Construct all three Layer-2 scalars in one call.
    Returns dict with scalar values + human-readable fields for JSONL.
    """
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
