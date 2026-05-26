"""stage3/master_corpus/manager.py — NextAura Master Corpus Submodule.
Manages a growing master corpus bucket at output/master_corpus.jsonl.
Handles drag-and-drop file merging (JSON, JSONL, CSV), SHA-256 deduplication,
compliance provenance block injection, human-interaction logging, and corpus regeneration.
"""

import csv
import json
import io
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from stage1.core.soc_compliance import IngestionComplianceEngine, calculate_record_hash
from stage1.core.record_builder import RecordBuilder
from stage1.core.triangulation import build_mathematical_relationship

# Root directory helper
ROOT_DIR = Path(__file__).resolve().parents[2]
MASTER_CORPUS_PATH = ROOT_DIR / "output" / "master_corpus.jsonl"


class MasterCorpusManager:
    """
    Manages the growing master corpus. Reuses the IngestionComplianceEngine
    for SHA-256 record hashing, deduplication, and SOC 2/3 provenance verification.
    """
    
    @staticmethod
    def get_existing_hashes() -> set:
        """Reads master_corpus.jsonl and returns a set of all unique record SHA-256 hashes."""
        hashes = set()
        if not MASTER_CORPUS_PATH.exists():
            return hashes
        try:
            with MASTER_CORPUS_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or not s.startswith("{"):
                        continue
                    try:
                        rec = json.loads(s)
                        h = rec.get("_meta", {}).get("provenance", {}).get("sha256")
                        if not h:
                            h = calculate_record_hash(rec.get("scalar_inputs", {}))
                        hashes.add(h)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[MasterCorpus] Error loading existing hashes: {e}")
        return hashes

    @staticmethod
    def add_interaction(
        question: str,
        answer: str,
        m_vals: Dict[str, float],
        l2_scalars: Dict[str, float],
        v_classical: float,
        action: str = "converge",
        next_frame_prediction: str = ""
    ) -> bool:
        """
        Formats a human chat interaction into a standard v31 compliant record,
        deduplicates it, and appends it to output/master_corpus.jsonl.
        """
        # 1. Format descriptions
        m1_desc = f"Unbounded thinker exploration_vector={m_vals.get('M1', 1.0):.4f}; neural feedback state."
        m2_desc = f"Efficiency score={m_vals.get('M2', 0.5):.4f}; neural threshold."
        m3_desc = f"Meta-multiplier={m_vals.get('M3', 1.0):.4f}; feedback gain."
        
        math_rel = build_mathematical_relationship(
            M1=m_vals.get('M1', 1.0),
            M2=m_vals.get('M2', 0.5),
            M3=m_vals.get('M3', 1.0),
            binary=l2_scalars.get('binary_scalar', 0.5),
            geometry=l2_scalars.get('geometry_scalar', 0.5),
            language=l2_scalars.get('language_scalar', 0.5),
            context_tags=f"interaction, V={v_classical:.4f}"
        )
        
        # 2. Build record
        record = {
            "M1": m1_desc,
            "M2": m2_desc,
            "M3": m3_desc,
            "scalar_inputs": {
                "binary": f"Chat session active. Score={l2_scalars.get('binary_scalar', 0.5):.3f}",
                "geometry": f"Interactive prompt step. Progress={l2_scalars.get('geometry_scalar', 0.5)*100:.1f}%",
                "language": question,
                "mathematical_relationship": math_rel
            },
            "output_tokens": {
                "next_frame_prediction": next_frame_prediction or f"Response: {answer[:60]}...",
                "language_token_output": action
            }
        }
        
        # 3. Calculate SHA-256 context hash
        rec_hash = calculate_record_hash(record["scalar_inputs"])
        
        # 4. Check for duplicates
        existing = MasterCorpusManager.get_existing_hashes()
        if rec_hash in existing:
            print(f"[MasterCorpus] Skipping duplicate interaction: {rec_hash[:12]}")
            return False
            
        # 5. Inject Compliance Provenance Block
        record["_meta"] = {
            "provenance": {
                "uuid": str(uuid.uuid4()),
                "sha256": rec_hash,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "lineage_source": "human_interaction_chat",
                "soc2_integrity_checked": True
            },
            "V": round(v_classical, 6),
            "M1_raw": round(m_vals.get('M1', 1.0), 6),
            "M2_raw": round(m_vals.get('M2', 0.5), 6),
            "M3_raw": round(m_vals.get('M3', 1.0), 6),
            "binary_scalar": round(l2_scalars.get('binary_scalar', 0.5), 6),
            "geometry_scalar": round(l2_scalars.get('geometry_scalar', 0.5), 6),
            "language_scalar": round(l2_scalars.get('language_scalar', 0.5), 6)
        }
        
        # 6. Append to file
        try:
            MASTER_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with MASTER_CORPUS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(f"[MasterCorpus] Successfully logged interaction: {rec_hash[:12]} to master_corpus.jsonl")
            return True
        except Exception as e:
            print(f"[MasterCorpus] Failed writing to master_corpus.jsonl: {e}")
            return False

    @staticmethod
    def add_file_to_master_corpus(file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Parses uploaded file bytes (JSON, JSONL, or CSV), deduplicates records,
        injects SOC compliance provenance blocks, and appends them to master_corpus.jsonl.
        """
        records: List[Dict[str, Any]] = []
        name_lower = filename.lower()
        
        try:
            # 1. Parse JSONL / JSON / CSV
            if name_lower.endswith(".jsonl"):
                content_str = file_content.decode("utf-8", errors="replace")
                for line in content_str.splitlines():
                    s = line.strip()
                    if not s or not s.startswith("{"):
                        continue
                    try:
                        records.append(json.loads(s))
                    except Exception:
                        pass
            elif name_lower.endswith(".json"):
                content_str = file_content.decode("utf-8", errors="replace")
                data = json.loads(content_str)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    records = [data]
            elif name_lower.endswith(".csv"):
                content_str = file_content.decode("utf-8", errors="replace")
                reader = csv.DictReader(io.StringIO(content_str))
                for row in reader:
                    # Map CSV fields to standard v31 record
                    # If columns are raw, package them into scalar_inputs
                    records.append({
                        "M1": row.get("M1", "M1 scale description"),
                        "M2": row.get("M2", "M2 scale description"),
                        "M3": row.get("M3", "M3 scale description"),
                        "scalar_inputs": {
                            "binary": row.get("binary", row.get("binary_text", "unspecified")),
                            "geometry": row.get("geometry", row.get("geometry_text", "unspecified")),
                            "language": row.get("language", row.get("language_text", "unspecified")),
                            "mathematical_relationship": row.get("mathematical_relationship", "")
                        },
                        "output_tokens": {
                            "next_frame_prediction": row.get("next_frame_prediction", ""),
                            "language_token_output": row.get("language_token_output", "converge")
                        }
                    })
            else:
                return {"error": "Unsupported file format. Please upload .json, .jsonl, or .csv"}
                
        except Exception as e:
            return {"error": f"Failed parsing file: {e}"}

        if not records:
            return {"error": "No valid records found in the uploaded file."}

        # 2. Get existing hashes in master corpus to prevent cross-file duplicates
        existing_hashes = MasterCorpusManager.get_existing_hashes()
        
        # 3. Clean and process the new records using the compliance engine
        clean_new_records: List[Dict[str, Any]] = []
        skipped_duplicates = 0
        
        for rec in records:
            # Enforce minimal skeleton structure
            if "scalar_inputs" not in rec or not isinstance(rec["scalar_inputs"], dict):
                rec["scalar_inputs"] = {
                    "binary": str(rec.get("binary", "unspecified")),
                    "geometry": str(rec.get("geometry", "unspecified")),
                    "language": str(rec.get("language", rec.get("text", "unspecified")))
                }
            if "output_tokens" not in rec or not isinstance(rec["output_tokens"], dict):
                rec["output_tokens"] = {
                    "next_frame_prediction": str(rec.get("next_frame_prediction", "")),
                    "language_token_output": str(rec.get("language_token_output", "converge"))
                }
                
            rec_hash = calculate_record_hash(rec["scalar_inputs"])
            if rec_hash in existing_hashes:
                skipped_duplicates += 1
                continue
                
            existing_hashes.add(rec_hash)
            clean_new_records.append(rec)
            
        if not clean_new_records:
            return {
                "success": True,
                "filename": filename,
                "total_read": len(records),
                "added_count": 0,
                "duplicates_skipped": skipped_duplicates,
                "message": "All records were duplicates of existing entries."
            }

        # 4. Use compliance engine to build provenance metadata for new clean entries
        processed, metrics = IngestionComplianceEngine.process_records(
            clean_new_records,
            source_name=filename
        )
        
        # 5. Append clean records to the master_corpus.jsonl file
        try:
            MASTER_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with MASTER_CORPUS_PATH.open("a", encoding="utf-8") as f:
                for rec in processed:
                    # Strip raw _meta properties except provenance to keep it clean, or keep all
                    f.write(json.dumps(rec) + "\n")
            
            return {
                "success": True,
                "filename": filename,
                "total_read": len(records),
                "added_count": len(processed),
                "duplicates_skipped": skipped_duplicates + metrics["duplicates_skipped"],
                "corpus_sha256_signature": metrics["corpus_sha256_signature"],
                "message": f"Successfully ingested {len(processed)} records into master_corpus.jsonl!"
            }
        except Exception as e:
            return {"error": f"Failed writing to master_corpus.jsonl: {e}"}

    @staticmethod
    def regenerate_corpus(engine: Any) -> Dict[str, Any]:
        """
        Iterates through output/master_corpus.jsonl and feeds inputs into the active
        V31NeuralInferenceEngine to refresh output_tokens and _meta values.
        Prevents stale/outdated predictions across model iterations.
        """
        if not MASTER_CORPUS_PATH.exists():
            return {"error": "Master corpus file does not exist. Nothing to regenerate."}
            
        t0 = time.time()
        records: List[Dict[str, Any]] = []
        
        # 1. Read existing corpus
        try:
            with MASTER_CORPUS_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or not s.startswith("{"):
                        continue
                    try:
                        records.append(json.loads(s))
                    except Exception:
                        pass
        except Exception as e:
            return {"error": f"Failed reading master corpus: {e}"}
            
        if not records:
            return {"success": True, "regenerated_count": 0, "message": "Corpus is empty."}

        # 2. Reset engine to clear stateful sequence tracking
        if hasattr(engine, "reset"):
            engine.reset()
            
        regenerated_records = []
        refreshed_count = 0
        
        # 3. Step through sequentially
        for i, rec in enumerate(records):
            scalar_in = rec.get("scalar_inputs", {})
            binary_text = scalar_in.get("binary", "")
            geometry_text = scalar_in.get("geometry", "")
            language_text = scalar_in.get("language", "")
            
            # Infer outputs using active model
            out = engine.infer(
                binary_text=binary_text,
                geometry_text=geometry_text,
                language_text=language_text,
                entry_index=i + 1,
                total_entries=len(records)
            )
            
            # Form descriptions
            m1_desc = f"Unbounded thinker exploration_vector={out['M1']:.4f}; neural feedback state."
            m2_desc = f"Efficiency score={out['M2']:.4f}; neural threshold."
            m3_desc = f"Meta-multiplier={out['M3']:.4f}; feedback gain."
            
            math_rel = build_mathematical_relationship(
                M1=out["M1"],
                M2=out["M2"],
                M3=out["M3"],
                binary=out["binary_scalar"],
                geometry=out["geometry_scalar"],
                language=out["language_scalar"],
                context_tags=f"entry={i+1}, V={out['V']:.4f}"
            )
            
            # Copy over unchanged fields but refresh outputs & descriptions
            rec["M1"] = m1_desc
            rec["M2"] = m2_desc
            rec["M3"] = m3_desc
            rec["scalar_inputs"]["mathematical_relationship"] = math_rel
            rec["output_tokens"] = {
                "next_frame_prediction": out["next_frame_prediction"],
                "language_token_output": out["language_token_output"]
            }
            
            # Preserve existing provenance but update raw scalars
            prov = rec.get("_meta", {}).get("provenance", {})
            if not prov:
                rec_hash = calculate_record_hash(rec["scalar_inputs"])
                prov = {
                    "uuid": str(uuid.uuid4()),
                    "sha256": rec_hash,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "lineage_source": "restored_during_regeneration",
                    "soc2_integrity_checked": True
                }
                
            rec["_meta"] = {
                "provenance": prov,
                "V": out["V"],
                "M1_raw": out["M1"],
                "M2_raw": out["M2"],
                "M3_raw": out["M3"],
                "binary_scalar": out["binary_scalar"],
                "geometry_scalar": out["geometry_scalar"],
                "language_scalar": out["language_scalar"],
                "timestamp": round(time.time(), 3),
                "is_regenerated": True
            }
            
            regenerated_records.append(rec)
            refreshed_count += 1
            
        # 4. Overwrite master_corpus.jsonl with regenerated records
        try:
            with MASTER_CORPUS_PATH.open("w", encoding="utf-8") as f:
                for rec in regenerated_records:
                    f.write(json.dumps(rec) + "\n")
            
            return {
                "success": True,
                "regenerated_count": refreshed_count,
                "elapsed_seconds": round(time.time() - t0, 3),
                "message": f"Successfully regenerated {refreshed_count} master corpus records!"
            }
        except Exception as e:
            return {"error": f"Failed saving regenerated corpus: {e}"}
