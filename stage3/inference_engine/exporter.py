"""stage3/inference_engine/exporter.py — V31all Model Exporter

Serializes the stateful V31InferenceEngine weights, metadata, and current states.
Outputs:
  1. A GGUF v3 compliant binary file (export_v31.gguf) containing model metadata, 
     vocabularies, and tensor weights.
  2. A human-readable JSON file (export_v31.json) for easy verification.
"""

import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from stage3.inference_engine.main import V31InferenceEngine
from stage1.core.triangulation import ACTION_VOCAB, STATE_VOCAB


class GGUFWriter:
    """
    Minimal compliant GGUF v3 writer.
    Allows packing metadata key-values and raw tensor weights.
    """
    def __init__(self):
        self.metadata: List[bytes] = []
        self.metadata_count = 0
        self.tensors: List[Dict[str, Any]] = []
        self.alignment = 32

    def _pack_string(self, s: str) -> bytes:
        b = s.encode('utf-8')
        return struct.pack('<Q', len(b)) + b

    def add_metadata(self, key: str, val_type: int, value: Any):
        """
        GGUF Metadata types:
          4: UINT32
          6: FLOAT32
          7: BOOL
          8: STRING
          9: ARRAY (nested)
          10: UINT64
        """
        # Pack key
        k_bytes = self._pack_string(key)
        # Pack type (uint32)
        t_bytes = struct.pack('<I', val_type)
        # Pack value based on type
        if val_type == 4:  # UINT32
            v_bytes = struct.pack('<I', int(value))
        elif val_type == 6:  # FLOAT32
            v_bytes = struct.pack('<f', float(value))
        elif val_type == 7:  # BOOL
            v_bytes = struct.pack('?', bool(value))
        elif val_type == 8:  # STRING
            v_bytes = self._pack_string(str(value))
        elif val_type == 10:  # UINT64
            v_bytes = struct.pack('<Q', int(value))
        elif val_type == 9:  # ARRAY
            # array is: type of array items (uint32), item count (uint64), then items
            elem_type, items = value
            elem_t_bytes = struct.pack('<I', elem_type)
            len_bytes = struct.pack('<Q', len(items))
            arr_bytes = []
            for item in items:
                if elem_type == 8:  # array of strings
                    arr_bytes.append(self._pack_string(str(item)))
                elif elem_type == 6:  # array of floats
                    arr_bytes.append(struct.pack('<f', float(item)))
                else:
                    raise ValueError(f"Array element type {elem_type} not implemented.")
            v_bytes = elem_t_bytes + len_bytes + b"".join(arr_bytes)
        else:
            raise ValueError(f"GGUF type {val_type} not implemented in exporter.")

        self.metadata.append(k_bytes + t_bytes + v_bytes)
        self.metadata_count += 1

    def add_tensor(self, name: str, data: np.ndarray):
        """
        Add a tensor to be written to GGUF.
        We support float32 (GGML_TYPE_F32 = 0) or int32 (GGML_TYPE_I32 = 23).
        """
        if data.dtype == np.float32:
            ggml_type = 0  # F32
        elif data.dtype == np.int32:
            ggml_type = 23  # I32
        else:
            data = data.astype(np.float32)
            ggml_type = 0

        self.tensors.append({
            "name": name,
            "data": data,
            "shape": list(data.shape),
            "ggml_type": ggml_type,
        })

    def write(self, path: Path) -> bytes:
        # GGUF Magic Header
        magic = b"GGUF"
        version = struct.pack('<I', 3)  # GGUF v3
        tensor_count = struct.pack('<Q', len(self.tensors))
        metadata_count = struct.pack('<Q', self.metadata_count)

        header = magic + version + tensor_count + metadata_count
        metadata_block = b"".join(self.metadata)

        # Tensor list header serialization
        tensor_infos = []
        current_offset = 0

        for t in self.tensors:
            name_bytes = self._pack_string(t["name"])
            n_dims = struct.pack('<I', len(t["shape"]))
            dims_bytes = b"".join(struct.pack('<Q', dim) for dim in t["shape"])
            type_bytes = struct.pack('<I', t["ggml_type"])
            offset_bytes = struct.pack('<Q', current_offset)

            info = name_bytes + n_dims + dims_bytes + type_bytes + offset_bytes
            tensor_infos.append(info)

            # Accumulate size of data for offsets, padded to alignment
            size_bytes = t["data"].nbytes
            current_offset += size_bytes
            # Tensors are padded to 32 bytes alignment
            padding = (self.alignment - (size_bytes % self.alignment)) % self.alignment
            current_offset += padding

        tensor_info_block = b"".join(tensor_infos)

        # Build final binary payload
        full_meta = header + metadata_block + tensor_info_block

        # Padding between metadata/info section and tensor data block
        meta_size = len(full_meta)
        pad_size = (self.alignment - (meta_size % self.alignment)) % self.alignment
        full_meta += b"\0" * pad_size

        # Serialize tensor data block
        tensor_data_blocks = []
        for t in self.tensors:
            data_bytes = t["data"].tobytes()
            padding = (self.alignment - (len(data_bytes) % self.alignment)) % self.alignment
            tensor_data_blocks.append(data_bytes + b"\0" * padding)

        full_payload = full_meta + b"".join(tensor_data_blocks)
        path.write_bytes(full_payload)
        return full_payload


def export_model(engine: V31InferenceEngine, output_dir: Path, name_prefix: str = "model_export") -> Dict[str, Path]:
    """
    Exports stateful engine weights and current context state.
    Creates:
      - output_dir / <name_prefix>_v31.gguf (binary)
      - output_dir / <name_prefix>_v31.json (JSON validation)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = output_dir / f"{name_prefix}_v31.gguf"
    json_path = output_dir / f"{name_prefix}_v31.json"

    # 1. Gather Tensors (weights & states)
    tri_weights = np.array(engine._tri.weights, dtype=np.float32)
    m1_exploration_depth = np.array([engine._m1.exploration_depth], dtype=np.float32)
    m1_branch_count = np.array([engine._m1.branch_count], dtype=np.int32)
    m2_cost_budget = np.array([engine._m2.cost_budget], dtype=np.float32)
    m2_step_count = np.array([engine._m2._step_count], dtype=np.int32)
    m3_logic_multiplier = np.array([engine._m3.logic_multiplier], dtype=np.float32)
    m3_efficiency_multiplier = np.array([engine._m3.efficiency_multiplier], dtype=np.float32)
    m3_creative_multiplier = np.array([engine._m3.creative_multiplier], dtype=np.float32)
    v_prior = np.array([engine._tri._prior_v], dtype=np.float32)
    v_history = np.array(engine._v_history, dtype=np.float32) if engine._v_history else np.array([0.5], dtype=np.float32)

    # 2. Gather Human-Readable Dictionary
    json_data = {
        "model_architecture": "observer_v31",
        "description": "NextAura Stateful Observer model weights and current parameters",
        "system_id": engine.system_id,
        "codebase_hash": engine.codebase_hash,
        "mission": engine.mission,
        "sandbox": engine.sandbox,
        "state": {
            "turn": engine._turn,
            "m1_state": {
                "branch_count": int(engine._m1.branch_count),
                "exploration_depth": float(engine._m1.exploration_depth)
            },
            "m2_state": {
                "cost_budget": float(engine._m2.cost_budget),
                "steps_allowed": float(engine._m2.steps_allowed),
                "step_count": int(engine._m2._step_count)
            },
            "m3_state": {
                "time_budget_seconds": float(engine._m3.time_budget_seconds),
                "logic_multiplier": float(engine._m3.logic_multiplier),
                "efficiency_multiplier": float(engine._m3.efficiency_multiplier),
                "creative_multiplier": float(engine._m3.creative_multiplier)
            },
            "triangulation_state": {
                "weights": [float(w) for w in engine._tri.weights],
                "prior_v": float(engine._tri._prior_v),
                "step": int(engine._tri._step)
            },
            "phase_gate_state": {
                "consecutive_opens": sum(1 for x in reversed(engine._gate._history) if x) if engine._gate._history else 0
            }
        },
        "vocabularies": {
            "action_vocab": ACTION_VOCAB,
            "state_vocab": STATE_VOCAB
        }
    }

    # Save JSON Export
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    # 3. Compile GGUF Binary
    writer = GGUFWriter()
    
    # Metadata Key-Values
    writer.add_metadata("general.architecture", 8, "observer")
    writer.add_metadata("general.name", 8, "NextAura Observer v31")
    writer.add_metadata("general.description", 8, "Triangulation stateful RL value function")
    writer.add_metadata("general.version", 8, "3.1")
    
    writer.add_metadata("observer.sandbox", 7, engine.sandbox)
    writer.add_metadata("observer.system_id", 8, engine.system_id)
    writer.add_metadata("observer.codebase_hash", 8, engine.codebase_hash)
    writer.add_metadata("observer.mission", 8, engine.mission)
    writer.add_metadata("observer.turn", 4, engine._turn)
    
    # Vocabs as metadata arrays of strings (GGUF Type 9, subarray string type 8)
    writer.add_metadata("observer.vocabulary.actions", 9, (8, ACTION_VOCAB))
    writer.add_metadata("observer.vocabulary.states", 9, (8, STATE_VOCAB))

    # Add Tensors
    writer.add_tensor("triangulation_weights", tri_weights)
    writer.add_tensor("m1_exploration_depth", m1_exploration_depth)
    writer.add_tensor("m1_branch_count", m1_branch_count)
    writer.add_tensor("m2_cost_budget", m2_cost_budget)
    writer.add_tensor("m2_step_count", m2_step_count)
    writer.add_tensor("m3_logic_multiplier", m3_logic_multiplier)
    writer.add_tensor("m3_efficiency_multiplier", m3_efficiency_multiplier)
    writer.add_tensor("m3_creative_multiplier", m3_creative_multiplier)
    writer.add_tensor("v_prior", v_prior)
    writer.add_tensor("v_history", v_history)

    # Write GGUF File
    writer.write(gguf_path)

    return {
        "gguf": gguf_path,
        "json": json_path,
    }


if __name__ == "__main__":
    engine = V31InferenceEngine()
    engine.infer("Test binary", "Test geometry", "Test language")
    paths = export_model(engine, Path("../../output"))
    print(f"[OK] GGUF export: {paths['gguf'].name} ({paths['gguf'].stat().st_size} bytes)")
    print(f"[OK] JSON export: {paths['json'].name} ({paths['json'].stat().st_size} bytes)")
