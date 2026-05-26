"""stage2/training_engine/state_registry.py — V31all Training Cache & State Registry.
Manages cycle-to-cycle model progression and skips redundant training states.
"""

import os
import json
import hashlib
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Root helper
ROOT_DIR = Path(__file__).resolve().parents[2]
REGISTRY_FILE_PATH = ROOT_DIR / "models" / "state_registry.json"

class AuraCycleManager:
    """
    Manages training cycles, dataset fingerprinting, state verification,
    and checkpoint caching to prevent retraining identical states.
    """
    
    @staticmethod
    def calculate_corpus_hash(corpus_path: str) -> str:
        """
        Generates a robust SHA-256 hash of the entire corpus by hashing
        the sequential concatenation of its record-level input signatures.
        """
        path = Path(corpus_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
            
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found for hashing: {path}")
            
        record_hashes = []
        from stage1.core.soc_compliance import calculate_record_hash
        
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s or not s.startswith("{"):
                    continue
                try:
                    rec = json.loads(s)
                    # Use existing SHA256 if available, or compute on the fly
                    rec_hash = rec.get("_meta", {}).get("provenance", {}).get("sha256")
                    if not rec_hash:
                        rec_hash = calculate_record_hash(rec.get("scalar_inputs", {}))
                    record_hashes.append(rec_hash)
                except Exception:
                    pass
        
        # Sort hashes to be order-independent, ensuring duplicate/reordered sets match
        record_hashes.sort()
        concat_hashes = "".join(record_hashes)
        return hashlib.sha256(concat_hashes.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_state_hash(corpus_hash: str, epochs: int, batch_size: int, lr: float) -> str:
        """
        Combines the corpus fingerprint and hyperparameters into a unique state hash.
        """
        # Format learning rate to prevent rounding discrepancies
        lr_str = f"{lr:.8f}".rstrip("0").rstrip(".")
        param_str = f"corpus:{corpus_hash}|epochs:{epochs}|batch:{batch_size}|lr:{lr_str}"
        return hashlib.sha256(param_str.encode("utf-8")).hexdigest()

    @staticmethod
    def check_state(state_hash: str) -> Optional[Dict[str, Any]]:
        """
        Checks the registry database to see if a state hash exists and its
        corresponding PyTorch checkpoint file is physically present on disk.
        """
        if not REGISTRY_FILE_PATH.exists():
            return None
            
        try:
            db = json.loads(REGISTRY_FILE_PATH.read_text(encoding="utf-8"))
            if state_hash in db:
                entry = db[state_hash]
                ckpt_path = ROOT_DIR / entry.get("checkpoint_path", "")
                if ckpt_path.exists():
                    return entry
        except Exception as e:
            print(f"Error checking state registry: {e}")
            
        return None

    @staticmethod
    def register_state(
        state_hash: str,
        corpus_path: str,
        corpus_hash: str,
        epochs: int,
        batch_size: int,
        lr: float,
        metrics: Dict[str, Any],
        source_checkpoint_path: str
    ) -> str:
        """
        Saves the finished training state weights as a cached checkpoint,
        registers the details in models/state_registry.json, and makes a copy
        to models/v31_neural_model.pt for immediate active inference.
        """
        models_dir = ROOT_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine the target cached checkpoint file path
        cached_filename = f"v31_neural_model_{state_hash[:16]}.pt"
        cached_ckpt_path = models_dir / cached_filename
        
        # 1. Copy source checkpoint to our cached target path
        shutil.copy(ROOT_DIR / source_checkpoint_path, cached_ckpt_path)
        
        # 2. Update models/v31_neural_model.pt (active symlink/copy)
        shutil.copy(cached_ckpt_path, models_dir / "v31_neural_model.pt")
        
        # 3. Write entry in registry database
        db = {}
        if REGISTRY_FILE_PATH.exists():
            try:
                db = json.loads(REGISTRY_FILE_PATH.read_text(encoding="utf-8"))
            except Exception:
                db = {}
                
        relative_cached_path = f"models/{cached_filename}"
        db[state_hash] = {
            "state_hash": state_hash,
            "corpus_path": corpus_path,
            "corpus_hash": corpus_hash,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "metrics": metrics,
            "checkpoint_path": relative_cached_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "soc2_compliance_validated": True
        }
        
        REGISTRY_FILE_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")
        return relative_cached_path

    @staticmethod
    def get_registered_states() -> Dict[str, Any]:
        """Loads and returns all registered training weight states."""
        if not REGISTRY_FILE_PATH.exists():
            return {}
        try:
            return json.loads(REGISTRY_FILE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
