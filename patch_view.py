import re
path = "/mnt/d/NextAura/v31all_1/v31allC/gui/stage1/view.py"
src = open(path, encoding="utf-8").read()

old_handlers = '''def handle_upload(e, ingest_log, ingest_stats):
    try:
        content = e.content.read().decode()
        fmt = Stage1Ingest.detect_format(content)
        parsed = Stage1Ingest.parse_format(content, fmt)
        
        _records.extend(parsed)
        ingest_log.push(f"\u2713 Uploaded ({fmt}) \u2192 {len(parsed)} records")
        ingest_stats.set_text(f"Ingested {len(_records)} total records")
        
        _recorder.log_event(
            EventType.USER_INPUT,
            message=f"File uploaded: {e.name}",
            data={"filename": e.name, "format": fmt, "records": len(parsed)}
        )
    except Exception as ex:
        ingest_log.push(f"\u2717 Error: {str(ex)}")
        _recorder.log_error(f"Upload failed for {e.name}", ex)

def handle_paste(content, ingest_log, ingest_stats):
    try:
        fmt = Stage1Ingest.detect_format(content)
        parsed = Stage1Ingest.parse_format(content, fmt)
        
        _records.extend(parsed)
        ingest_log.push(f"\u2713 Pasted ({fmt}) \u2192 {len(parsed)} records")
        ingest_stats.set_text(f"Ingested {len(_records)} total records")
        
        _recorder.log_event(
            EventType.USER_INPUT,
            message=f"Data pasted: {fmt} format",
            data={"format": fmt, "records": len(parsed)}
        )
        
        # Auto-compute scalars after ingest
        _auto_compute_after_ingest()
    except Exception as ex:
        ingest_log.push(f"\u2717 Error: {str(ex)}")
        _recorder.log_error(f"Paste failed", ex)'''

new_handlers = '''def _run_batch(parsed, source_name, ingest_log, ingest_stats):
    """Route parsed records through RecordBuilder.build_batch (GPU language scalars)."""
    rows = []
    for rec in parsed:
        si = rec.get("scalar_inputs", {}) if isinstance(rec.get("scalar_inputs"), dict) else {}
        lang = si.get("language") or rec.get("description") or rec.get("language") or rec.get("text") or ""
        geo  = si.get("geometry") or rec.get("analysis") or rec.get("geometry") or rec.get("phase") or ""
        bin_ = si.get("binary") or str(rec.get("step") or rec.get("entry_id") or rec.get("binary") or "")
        if not lang and not geo:
            continue
        rows.append({"binary_text": bin_[:300], "geometry_text": geo[:400], "language_text": lang[:600]})
    if not rows:
        ingest_log.push("No usable records found.")
        return
    built = _rb.build_batch(rows)
    _records.extend(built)
    vs = [r["_meta"]["V"] for r in built]
    v_mean = sum(vs)/len(vs) if vs else 0
    v_max  = max(vs) if vs else 0
    ingest_log.push(f"Built {len(built)} records | V_mean={v_mean:.4f} V_max={v_max:.4f}")
    ingest_stats.set_text(f"{len(_records)} total | last batch V_mean={v_mean:.4f} V_max={v_max:.4f}")
    _recorder.log_event(EventType.USER_INPUT, message=f"Batch: {source_name}",
                        data={"records": len(built), "V_mean": round(v_mean,4)})

def handle_upload(e, ingest_log, ingest_stats):
    try:
        content = e.content.read().decode(errors="replace")
        fmt = Stage1Ingest.detect_format(content)
        parsed = Stage1Ingest.parse_format(content, fmt)
        ingest_log.push(f"Parsed {len(parsed)} records ({fmt}) - GPU batch build...")
        _run_batch(parsed, e.name, ingest_log, ingest_stats)
    except Exception as ex:
        ingest_log.push(f"Error: {str(ex)}")
        _recorder.log_error(f"Upload failed for {e.name}", ex)

def handle_paste(content, ingest_log, ingest_stats):
    try:
        fmt = Stage1Ingest.detect_format(content)
        parsed = Stage1Ingest.parse_format(content, fmt)
        ingest_log.push(f"Parsed {len(parsed)} records ({fmt}) - GPU batch build...")
        _run_batch(parsed, "paste", ingest_log, ingest_stats)
        _auto_compute_after_ingest()
    except Exception as ex:
        ingest_log.push(f"Error: {str(ex)}")
        _recorder.log_error("Paste failed", ex)'''

if old_handlers in src:
    src = src.replace(old_handlers, new_handlers)
    open(path, "w", encoding="utf-8").write(src)
    print("patched view.py")
else:
    print("MISMATCH - dumping search target:")
    print(repr(src[src.find("def handle_upload"):src.find("def handle_upload")+200]))