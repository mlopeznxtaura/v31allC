"""stage1/tests/test_compliance_pipeline.py — Comprehensive unit tests for the NextAura AGI compliance, cache, and corpus pipeline.
Tests SOC 2/3 cryptographic hashing, training cache skips, and master corpus grow/dedup logic.
"""

import os
import json
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from stage1.core.soc_compliance import (
    calculate_record_hash,
    compute_corpus_fingerprint,
    IngestionComplianceEngine
)
from stage2.training_engine.state_registry import AuraCycleManager
from stage3.master_corpus.manager import MasterCorpusManager


class TestCompliancePipeline(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test artifacts
        self.test_dir = Path(tempfile.mkdtemp())
        self.audit_path = self.test_dir / "test_audit_manifest.json"
        self.registry_path = self.test_dir / "test_state_registry.json"
        self.master_corpus_path = self.test_dir / "test_master_corpus.jsonl"
        self.models_dir = self.test_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Mock physical checkpoint directories/files inside the test sandbox
        self.mock_src_checkpoint = self.test_dir / "test_model_weights.pt"
        self.mock_src_checkpoint.write_bytes(b"MOCK_PT_WEIGHTS_CONTENT")

    def tearDown(self):
        # Recursively remove the sandbox directory after testing
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_record_cryptographic_hashing(self):
        """Verify identical scalar inputs yield identical SHA-256 hashes, and differences yield different hashes."""
        inputs_1 = {
            "binary": "Logical gate open",
            "geometry": "Box coordinates set",
            "language": "Begin evaluation cycle"
        }
        inputs_2 = {
            "binary": "Logical gate open",
            "geometry": "Box coordinates set",
            "language": "Begin evaluation cycle"
        }
        inputs_different = {
            "binary": "Logical gate CLOSED",
            "geometry": "Box coordinates set",
            "language": "Begin evaluation cycle"
        }

        hash_1 = calculate_record_hash(inputs_1)
        hash_2 = calculate_record_hash(inputs_2)
        hash_different = calculate_record_hash(inputs_different)

        self.assertEqual(hash_1, hash_2, "Identical records must compute identical SHA-256 hashes.")
        self.assertNotEqual(hash_1, hash_different, "Differing record contents must compute distinct hashes.")
        self.assertEqual(len(hash_1), 64, "SHA-256 hash must be exactly 64 hex characters.")

    def test_corpus_fingerprinting(self):
        """Verify that corpus fingerprints are unique to content and change when individual records change."""
        records = [
            {"scalar_inputs": {"binary": "1", "geometry": "box1", "language": "eval"}},
            {"scalar_inputs": {"binary": "0", "geometry": "box2", "language": "train"}}
        ]
        
        # Calculate fingerprint on clean records
        fingerprint_1 = compute_corpus_fingerprint(records)
        
        # Modify a single record field
        records_modified = [
            {"scalar_inputs": {"binary": "1", "geometry": "box1", "language": "eval"}},
            {"scalar_inputs": {"binary": "0", "geometry": "box2", "language": "train_modified"}}
        ]
        fingerprint_modified = compute_corpus_fingerprint(records_modified)

        self.assertNotEqual(fingerprint_1, fingerprint_modified, "Modifying one field must alter the corpus-level hash.")
        self.assertEqual(len(fingerprint_1), 64)

    def test_ingestion_compliance_engine_deduplication(self):
        """Verify duplicate records are filtered and unique records are stamped with compliant SOC 2/3 provenance blocks."""
        raw_records = [
            {"scalar_inputs": {"binary": "gate_1", "geometry": "geom_1", "language": "lang_1"}},
            {"scalar_inputs": {"binary": "gate_1", "geometry": "geom_1", "language": "lang_1"}},  # Duplicate
            {"scalar_inputs": {"binary": "gate_2", "geometry": "geom_2", "language": "lang_2"}},  # Unique
        ]

        # Patch the audit manifest output path to our test sandbox path
        with patch("stage1.core.soc_compliance.AUDIT_FILE_PATH", self.audit_path):
            clean_records, metrics = IngestionComplianceEngine.process_records(
                raw_records, 
                source_name="test_ingestion_feed.jsonl"
            )

            # Check duplication removal
            self.assertEqual(len(clean_records), 2, "Duplicate records must be removed.")
            self.assertEqual(metrics["duplicates_skipped"], 1, "Metrics must record precisely 1 duplicate skipped.")
            self.assertEqual(metrics["unique_record_count"], 2)

            # Check SOC 2/3 compliance stamping
            for rec in clean_records:
                self.assertIn("_meta", rec)
                self.assertIn("provenance", rec["_meta"])
                prov = rec["_meta"]["provenance"]
                self.assertIn("uuid", prov)
                self.assertIn("sha256", prov)
                self.assertEqual(prov["lineage_source"], "test_ingestion_feed.jsonl")
                self.assertTrue(prov["soc2_integrity_checked"])

            # Verify the audit log file was written
            self.assertTrue(self.audit_path.exists(), "Compliance audit file must be written to disk.")
            history = IngestionComplianceEngine.get_audit_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["source"], "test_ingestion_feed.jsonl")

    def test_training_cache_fingerprinting_and_progression(self):
        """Verify corpus dataset fingerprint is order-independent and AuraCycleManager computes robust state hashes."""
        # 1. Create a mock temporary JSONL corpus
        corpus_file_1 = self.test_dir / "test_corpus_a.jsonl"
        rec_a = {"scalar_inputs": {"binary": "0.1", "geometry": "circle", "language": "hello"}}
        rec_b = {"scalar_inputs": {"binary": "0.9", "geometry": "square", "language": "world"}}
        
        with corpus_file_1.open("w", encoding="utf-8") as f:
            f.write(json.dumps(rec_a) + "\n")
            f.write(json.dumps(rec_b) + "\n")

        # 2. Create the same corpus but with records reordered
        corpus_file_2 = self.test_dir / "test_corpus_b.jsonl"
        with corpus_file_2.open("w", encoding="utf-8") as f:
            f.write(json.dumps(rec_b) + "\n")
            f.write(json.dumps(rec_a) + "\n")

        # 3. Check order-independent fingerprinting
        hash_a = AuraCycleManager.calculate_corpus_hash(str(corpus_file_1))
        hash_b = AuraCycleManager.calculate_corpus_hash(str(corpus_file_2))
        self.assertEqual(hash_a, hash_b, "Corpus hashing must be order-independent to support robust deduplication.")

        # 4. Check state hashes with hyperparameter variations
        sh_1 = AuraCycleManager.compute_state_hash(hash_a, epochs=10, batch_size=16, lr=0.001)
        sh_different_lr = AuraCycleManager.compute_state_hash(hash_a, epochs=10, batch_size=16, lr=0.002)
        sh_different_batch = AuraCycleManager.compute_state_hash(hash_a, epochs=10, batch_size=32, lr=0.001)
        
        self.assertNotEqual(sh_1, sh_different_lr, "Changing learning rate must alter the final training state signature.")
        self.assertNotEqual(sh_1, sh_different_batch, "Changing batch size must alter the final training state signature.")

    def test_state_registry_cache_skips(self):
        """Verify registering completed states properly saves checkpoints and training checks trigger skips on cache hits."""
        state_hash = "abc123statehashffffff"
        corpus_hash = "xyz987corpuhashffffff"
        metrics = {"total_loss": 0.0123, "mse_loss": 0.0095, "ce_loss": 0.0028}

        # Mock the AuraCycleManager's registry path and root dir
        with patch("stage2.training_engine.state_registry.REGISTRY_FILE_PATH", self.registry_path), \
             patch("stage2.training_engine.state_registry.ROOT_DIR", self.test_dir):

            # Initial check must yield None
            cached_entry = AuraCycleManager.check_state(state_hash)
            self.assertIsNone(cached_entry)

            # Register completed training run
            relative_ckpt = AuraCycleManager.register_state(
                state_hash=state_hash,
                corpus_path="output/test_corpus_a.jsonl",
                corpus_hash=corpus_hash,
                epochs=15,
                batch_size=16,
                lr=0.001,
                metrics=metrics,
                source_checkpoint_path=str(self.mock_src_checkpoint.relative_to(self.test_dir))
            )

            # Verify registry DB entry and file creation
            self.assertTrue(self.registry_path.exists())
            self.assertEqual(relative_ckpt, f"models/v31_neural_model_{state_hash[:16]}.pt")
            self.assertTrue((self.models_dir / f"v31_neural_model_{state_hash[:16]}.pt").exists())
            self.assertTrue((self.models_dir / "v31_neural_model.pt").exists())

            # Subsequent check must HIT the cache
            cached_hit = AuraCycleManager.check_state(state_hash)
            self.assertIsNotNone(cached_hit)
            self.assertEqual(cached_hit["state_hash"], state_hash)
            self.assertEqual(cached_hit["metrics"]["total_loss"], 0.0123)

    def test_master_corpus_appending_and_merging(self):
        """Verify MasterCorpusManager successfully logs, deduplicates, and parses uploaded JSON, JSONL, and CSV lines."""
        # 1. Patch the master corpus file path
        with patch("stage3.master_corpus.manager.MASTER_CORPUS_PATH", self.master_corpus_path), \
             patch("stage3.master_corpus.manager.ROOT_DIR", self.test_dir):

            # Check adding interaction
            m_vals = {"M1": 0.85, "M2": 0.40, "M3": 1.15}
            l2_scalars = {"binary_scalar": 0.55, "geometry_scalar": 0.70, "language_scalar": 0.80}
            
            success = MasterCorpusManager.add_interaction(
                question="What is the phase gate status?",
                answer="Gate is currently BUILDING with V=0.72",
                m_vals=m_vals,
                l2_scalars=l2_scalars,
                v_classical=0.72,
                action="monitor"
            )
            self.assertTrue(success, "Interaction logging must succeed on a fresh corpus.")
            self.assertTrue(self.master_corpus_path.exists())

            # Test duplicate protection
            success_duplicate = MasterCorpusManager.add_interaction(
                question="What is the phase gate status?",  # Same question
                answer="Gate is currently BUILDING with V=0.72",
                m_vals=m_vals,
                l2_scalars=l2_scalars,
                v_classical=0.72,
                action="monitor"
            )
            self.assertFalse(success_duplicate, "Identical question context should trigger deduplication skip.")

            # Test merging JSONL file
            jsonl_content = (
                json.dumps({"scalar_inputs": {"binary": "0.1", "geometry": "g1", "language": "distinct prompt 1"}}) + "\n" +
                json.dumps({"scalar_inputs": {"binary": "0.2", "geometry": "g2", "language": "distinct prompt 2"}}) + "\n"
            ).encode("utf-8")
            
            res = MasterCorpusManager.add_file_to_master_corpus(jsonl_content, "batch_import.jsonl")
            self.assertTrue(res.get("success"), f"JSONL merge failed: {res.get('error')}")
            self.assertEqual(res["added_count"], 2, "Both records should be appended as they are unique.")

            # Verify the growing count
            hashes = MasterCorpusManager.get_existing_hashes()
            self.assertEqual(len(hashes), 3, "Master corpus should now contain exactly 3 unique records.")


if __name__ == "__main__":
    unittest.main()
