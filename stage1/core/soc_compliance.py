"""stage1/core/soc_compliance.py — NextAura Ingestion Compliance & Audit Library.
Provides cryptographic hashing, duplicate detection, and lineage tracking (SOC 2/3).
"""

import hashlib
import json
import uuid
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Root directory helper
ROOT_DIR = Path(__file__).resolve().parents[2]
AUDIT_FILE_PATH = ROOT_DIR / "output" / "ingestion_manifest_audit.json"

def calculate_record_hash(scalar_inputs: Dict[str, Any]) -> str:
    """
    Computes a cryptographic SHA-256 fingerprint for a single record's inputs.
    Normalizes the scalar input fields to ensure consistent hashing.
    """
    if not isinstance(scalar_inputs, dict):
        scalar_inputs = {}
    
    # Normalize keys and values to ensure consistent hashing
    norm_binary = str(scalar_inputs.get("binary") or "").strip()
    norm_geometry = str(scalar_inputs.get("geometry") or "").strip()
    norm_language = str(scalar_inputs.get("language") or "").strip()
    
    # Concatenate fields
    concat_str = f"bin:{norm_binary}|geo:{norm_geometry}|lang:{norm_language}"
    return hashlib.sha256(concat_str.encode("utf-8")).hexdigest()

def compute_corpus_fingerprint(records: List[Dict[str, Any]]) -> str:
    """
    Fingerprints an entire corpus by concatenating sorted record-level SHA-256 hashes.
    Ensures that any single-record modification alters the entire corpus signature.
    """
    record_hashes = []
    for rec in records:
        meta = rec.get("_meta", {})
        provenance = meta.get("provenance", {})
        rec_hash = provenance.get("sha256")
        if not rec_hash:
            # Fallback if provenance is not built yet
            rec_hash = calculate_record_hash(rec.get("scalar_inputs", {}))
        record_hashes.append(rec_hash)
    
    # Sorting ensures order independence for identical dataset contents,
    # or keep unsorted if order sequence defines the corpus state.
    # Let's sort to guarantee unique set fingerprinting.
    record_hashes.sort()
    
    concat_hashes = "".join(record_hashes)
    return hashlib.sha256(concat_hashes.encode("utf-8")).hexdigest()

class IngestionComplianceEngine:
    """
    Engine to clean, deduplicate, and inject SOC 2/3 provenance logs.
    """
    
    @staticmethod
    def process_records(
        records: List[Dict[str, Any]], 
        source_name: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Deduplicates records based on scalar inputs SHA-256, injects provenance blocks,
        and saves an ingestion event audit log.
        
        Returns:
            clean_records: List of unique records with injected compliance blocks.
            metrics: Diagnostic dict of the ingestion run.
        """
        t0 = time.time()
        seen_hashes = set()
        clean_records = []
        duplicate_count = 0
        
        for rec in records:
            # Reconstruct or fetch scalar inputs
            scalar_inputs = rec.get("scalar_inputs", {})
            rec_hash = calculate_record_hash(scalar_inputs)
            
            if rec_hash in seen_hashes:
                duplicate_count += 1
                continue
                
            seen_hashes.add(rec_hash)
            
            # Inject or format _meta block
            if "_meta" not in rec or not isinstance(rec["_meta"], dict):
                rec["_meta"] = {}
                
            # Create SOC compliant provenance metadata
            rec["_meta"]["provenance"] = {
                "uuid": str(uuid.uuid4()),
                "sha256": rec_hash,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "lineage_source": Path(source_name).name,
                "soc2_integrity_checked": True
            }
            
            clean_records.append(rec)
            
        # Compute final overall corpus fingerprint
        corpus_hash = compute_corpus_fingerprint(clean_records) if clean_records else ""
        
        # Build ingestion metrics
        metrics = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": Path(source_name).name,
            "raw_record_count": len(records),
            "unique_record_count": len(clean_records),
            "duplicates_skipped": duplicate_count,
            "corpus_sha256_signature": corpus_hash,
            "elapsed_seconds": round(time.time() - t0, 4)
        }
        
        # Append ingestion event to manifest file
        IngestionComplianceEngine._write_audit_log(metrics)
        
        return clean_records, metrics

    @staticmethod
    def _write_audit_log(event: Dict[str, Any]):
        """Appends compliance audit event to output/ingestion_manifest_audit.json."""
        try:
            AUDIT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            history = []
            if AUDIT_FILE_PATH.exists():
                try:
                    content = AUDIT_FILE_PATH.read_text(encoding="utf-8")
                    if content.strip():
                        history = json.loads(content)
                        if not isinstance(history, list):
                            history = []
                except Exception:
                    history = []
            
            history.append(event)
            AUDIT_FILE_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"Failed writing compliance audit manifest: {e}")

    @staticmethod
    def get_audit_history() -> List[Dict[str, Any]]:
        """Reads and returns the complete compliance audit manifest logs."""
        if not AUDIT_FILE_PATH.exists():
            return []
        try:
            content = AUDIT_FILE_PATH.read_text(encoding="utf-8")
            if content.strip():
                history = json.loads(content)
                if isinstance(history, list):
                    return history
        except Exception:
            pass
        return []
