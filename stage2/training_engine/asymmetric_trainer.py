"""stage2/training_engine/asymmetric_trainer.py — V31all Asymmetric Trainer
Implements deep learning model training for the V31Model:
- Decodes and down-samples 64x64 base64 corpus visual frames to 14x16 (224-dim) targets.
- Deterministically hashes output language tokens into a [0, 4999] vocabulary range.
- Implements the train epoch and single train step coordinate loops using Adam.
- Handles checkpoint saving and loading.
"""

import json
import base64
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Callable

from stage2.training_engine.neural_model import V31Model


def decode_and_pool_frame(b64_str: str, shape: Tuple[int, int, int] = (64, 64, 3)) -> torch.Tensor:
    """
    Decodes the 64x64x3 base64 image from the corpus and average-pools it into a
    14x16 grayscale visual frame (exactly 224 pixels).
    """
    if not b64_str:
        return torch.zeros(224)
        
    try:
        # Decode base64 bytes
        img_bytes = base64.b64decode(b64_str)
        flat_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        
        # Reshape to expected (64, 64, 3)
        expected_size = shape[0] * shape[1] * shape[2]
        if len(flat_arr) < expected_size:
            # Pad with zeros if size mismatch
            padded = np.zeros(expected_size, dtype=np.uint8)
            padded[:len(flat_arr)] = flat_arr
            flat_arr = padded
        else:
            flat_arr = flat_arr[:expected_size]
            
        img = flat_arr.reshape(shape).astype(np.float32)
        
        # Convert to Grayscale: Y = 0.2989*R + 0.5870*G + 0.1140*B
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        
        # Pool to 14x16
        # Height blocks: 14 regions. Width blocks: 16 regions.
        h, w = gray.shape
        pooled = np.zeros((14, 16), dtype=np.float32)
        
        for r in range(14):
            r_start = int(r * h / 14)
            r_end = int((r + 1) * h / 14)
            for c in range(16):
                c_start = int(c * w / 16)
                c_end = int((c + 1) * w / 16)
                
                block = gray[r_start:r_end, c_start:c_end]
                pooled[r, c] = block.mean() if block.size > 0 else 0.0
                
        # Flatten to 224 floats
        return torch.from_numpy(pooled.flatten())
    except Exception:
        # Graceful fallback to zeros on any decode failure
        return torch.zeros(224)


def get_deterministic_token_id(lto_str: str, vocab_size: int = 5000) -> int:
    """
    Deterministically maps a string action token to an integer ID in [0, vocab_size-1].
    """
    if not lto_str:
        return 0
    h_hex = hashlib.md5(lto_str.strip().encode("utf-8")).hexdigest()
    return int(h_hex, 16) % vocab_size


class AsymmetricTrainer:
    """
    Coordinates V31Model training steps, loss computation, backpropagation, and state saving.
    """
    def __init__(
        self,
        model: V31Model,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
    ):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion_mse = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.step_count = 0

    def load_corpus_data(self, corpus_path: str) -> List[Dict]:
        """
        Reads a .jsonl corpus, extracts vectors and targets, and returns a list of dictionaries.
        """
        frames = []
        path = Path(corpus_path)
        if not path.exists():
            return []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    meta = rec.get("_meta", {})
                    si = rec.get("scalar_inputs", {})
                    ot = rec.get("output_tokens", {})
                    
                    # Extract scalar inputs
                    m1 = float(meta.get("M1_raw", 1.0))
                    m2 = float(meta.get("M2_raw", 1.0))
                    m3 = float(meta.get("M3_raw", 1.0))
                    geo_s = float(meta.get("geometry_scalar", 0.0))
                    bin_s = float(meta.get("binary_scalar", 0.0))
                    lng_s = float(meta.get("language_scalar", 0.0))
                    tri_s = float(meta.get("V", 0.5))
                    
                    # Vectorize layer 2 scalars
                    geo = torch.ones(32) * geo_s
                    bin_v = torch.ones(32) * bin_s
                    lng = torch.ones(128) * lng_s
                    tri = torch.ones(32) * tri_s
                    
                    # Extract target next frame prediction
                    b64_frame = meta.get("pixel_frame_base64", "")
                    next_frame_target = decode_and_pool_frame(b64_frame)
                    
                    # Extract language token output
                    lto_str = ot.get("language_token_output", "")
                    next_token_id = get_deterministic_token_id(lto_str)
                    
                    frames.append({
                        "m1": m1, "m2": m2, "m3": m3,
                        "geo": geo, "bin": bin_v, "lng": lng, "tri": tri,
                        "next_frame_target": next_frame_target,
                        "next_token_id": next_token_id
                    })
                except Exception:
                    continue
        return frames

    def train_epoch(self, corpus_path: str, batch_size: int = 16, on_batch_complete: Callable = None) -> Dict[str, float]:
        """
        Trains the model for one epoch over the corpus data.
        """
        frames = self.load_corpus_data(corpus_path)
        if not frames:
            return {"loss": 0.0, "mse": 0.0, "ce": 0.0, "records": 0}

        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_ce = 0.0
        num_batches = 0

        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            
            # Collate batch tensors
            device = next(self.model.parameters()).device
            m1_b = torch.tensor([f["m1"] for f in batch], dtype=torch.float32).to(device)
            m2_b = torch.tensor([f["m2"] for f in batch], dtype=torch.float32).to(device)
            m3_b = torch.tensor([f["m3"] for f in batch], dtype=torch.float32).to(device)
            geo_b = torch.stack([f["geo"] for f in batch]).to(device)
            bin_b = torch.stack([f["bin"] for f in batch]).to(device)
            lng_b = torch.stack([f["lng"] for f in batch]).to(device)
            tri_b = torch.stack([f["tri"] for f in batch]).to(device)
            
            target_frame_b = torch.stack([f["next_frame_target"] for f in batch]).to(device)
            target_token_b = torch.tensor([f["next_token_id"] for f in batch], dtype=torch.long).to(device)

            self.optimizer.zero_grad()
            
            # Model forward
            pred_frame, pred_token_logits = self.model(m1_b, m2_b, m3_b, geo_b, bin_b, lng_b, tri_b)
            
            # Loss computations
            loss_mse = self.criterion_mse(pred_frame, target_frame_b)
            loss_ce = self.criterion_ce(pred_token_logits, target_token_b)
            
            # Combine losses: weighted average
            loss = loss_mse + 0.1 * loss_ce
            
            if torch.isnan(loss):
                self.optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            self.step_count += 1
            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_ce += loss_ce.item()
            num_batches += 1
            
            if on_batch_complete:
                on_batch_complete(num_batches, loss.item(), loss_mse.item(), loss_ce.item())

        return {
            "loss": total_loss / max(num_batches, 1),
            "mse": total_mse / max(num_batches, 1),
            "ce": total_ce / max(num_batches, 1),
            "records": len(frames)
        }

    def save_checkpoint(self, save_path: str):
        """
        Saves weights and progress checkpoint to disk.
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step_count
        }, str(path))

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Loads weights checkpoint from disk. Returns True if successful.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            return False
        try:
            checkpoint = torch.load(str(path))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.step_count = checkpoint.get("step", 0)
            return True
        except Exception:
            return False
