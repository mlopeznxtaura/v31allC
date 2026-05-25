#!/usr/bin/env python3
"""Ingest JSONL into v31 schema (7-input -> 2-output), with validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

REQ_TOP = ("M1", "M2", "M3", "scalar_inputs", "output_tokens")
REQ_SCALAR = ("binary", "geometry", "language", "mathematical_relationship")
REQ_OUT = ("next_frame_prediction", "language_token_output")


def _coerce_record(rec: dict, idx: int) -> dict | None:
    legacy_binary = str(rec.get("Binary") or rec.get("binary") or "")
    legacy_geometry = str(rec.get("Geometry") or rec.get("geometry") or "")
    legacy_language = str(rec.get("LanguageContext") or rec.get("language") or "")
    legacy_math = str(rec.get("MathematicalRelationshipTriangulation") or rec.get("mathematical_relationship") or "")
    legacy_nfp = str(rec.get("NextFramePrediction") or rec.get("next_frame_prediction") or "")
    legacy_lto = str(rec.get("LanguageTokenOutput") or rec.get("language_token_output") or "")

    # Already v31-like
    if all(k in rec for k in REQ_TOP):
        si = rec.get("scalar_inputs", {}) if isinstance(rec.get("scalar_inputs"), dict) else {}
        ot = rec.get("output_tokens", {}) if isinstance(rec.get("output_tokens"), dict) else {}
        return {
            "M1": str(rec.get("M1", "")),
            "M2": str(rec.get("M2", "")),
            "M3": str(rec.get("M3", "")),
            "scalar_inputs": {
                "binary": str(si.get("binary") or legacy_binary),
                "geometry": str(si.get("geometry") or legacy_geometry),
                "language": str(si.get("language") or legacy_language),
                "mathematical_relationship": str(si.get("mathematical_relationship") or legacy_math),
            },
            "output_tokens": {
                "next_frame_prediction": str(ot.get("next_frame_prediction") or legacy_nfp),
                "language_token_output": str(ot.get("language_token_output") or legacy_lto),
            },
        }

    # Fallback mapping for legacy rows
    mapped = {
        "M1": str(rec.get("M1", f"Ingested row {idx}: M1 unspecified.")),
        "M2": str(rec.get("M2", f"Ingested row {idx}: M2 unspecified.")),
        "M3": str(rec.get("M3", f"Ingested row {idx}: M3 unspecified.")),
        "scalar_inputs": {
            "binary": legacy_binary,
            "geometry": legacy_geometry,
            "language": legacy_language,
            "mathematical_relationship": legacy_math,
        },
        "output_tokens": {
            "next_frame_prediction": legacy_nfp,
            "language_token_output": legacy_lto,
        },
    }
    # If a row has neither v31 nor legacy scalar/output fields, treat as metadata.
    si = mapped["scalar_inputs"]
    ot = mapped["output_tokens"]
    if not any(si.values()) and not any(ot.values()):
        return None
    if not ot["language_token_output"]:
        ot["language_token_output"] = "continue_monitor"
    if not ot["next_frame_prediction"]:
        lang = si["language"] or "ingested row"
        ot["next_frame_prediction"] = f"Ingested next frame from: {lang[:96]}"
    return mapped


def _iter_records(paths: Iterable[Path]):
    bad_json = 0
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                if not s.startswith("{"):
                    continue
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError as e:
                    bad_json += 1
                    continue
                if not isinstance(rec, dict):
                    raise ValueError(f"{p}:{line_no} record is not object")
                yield p, line_no, rec
    if bad_json:
        print(json.dumps({"warning": "skipped_malformed_json_lines", "count": bad_json}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--strict", action="store_true", help="Fail on missing core output fields")
    args = ap.parse_args()

    inputs = [Path(x) for x in args.inputs]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"input not found: {p}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    with out.open("w", encoding="utf-8") as wf:
        for p, line_no, rec in _iter_records(inputs):
            norm = _coerce_record(rec, count)
            if norm is None:
                skipped += 1
                continue
            count += 1
            if args.strict:
                if not norm["scalar_inputs"]["language"]:
                    raise ValueError(f"{p}:{line_no} strict mode: missing scalar_inputs.language")
                if not norm["output_tokens"]["language_token_output"]:
                    raise ValueError(f"{p}:{line_no} strict mode: missing output_tokens.language_token_output")
            wf.write(json.dumps(norm, ensure_ascii=True) + "\n")

    print(json.dumps({"status": "ok", "records": count, "skipped": skipped, "output": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
