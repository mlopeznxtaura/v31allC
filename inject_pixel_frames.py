"""inject_pixel_frames.py — NextAura v31allC
Captures real pixel frames via Windows screenshot, resizes to 64x64x3,
encodes as base64, and patches existing master_corpus.jsonl records that
are missing pixel_frame_base64 in their _meta block.

Strategy:
  - Take N desktop screenshots (different crop regions to get visual variety)
  - Encode each as 64x64x3 base64 PNG bytes
  - Round-robin assign them across all master corpus records missing frames
  - Rewrite master_corpus.jsonl in-place with patched _meta blocks
"""

import sys, json, base64, time, io
from pathlib import Path
from PIL import ImageGrab, Image

ROOT   = Path(__file__).resolve().parent
MASTER = ROOT / "output" / "master_corpus.jsonl"

# How many distinct frame templates to generate
N_FRAMES = 32


def capture_frame_variants(n: int) -> list:
    """
    Capture N screenshot crops from the desktop, resize each to 64x64x3,
    return list of base64-encoded raw RGB bytes (not PNG — raw pixel array).
    The asymmetric_trainer expects raw uint8 bytes for a 64x64x3 image.
    """
    print(f"  Capturing {n} frame variants from desktop …")
    frames_b64 = []

    try:
        full = ImageGrab.grab(all_screens=False)  # primary screen
    except Exception as ex:
        print(f"  [WARN] ImageGrab failed ({ex}), using synthetic frames.")
        full = None

    for i in range(n):
        if full is not None:
            w, h = full.size
            # Slide a crop window across the screen for variety
            cx = int((i / n) * (w - w // 4))
            cy = int(((i * 7) % n / n) * (h - h // 4))
            crop = full.crop((cx, cy, cx + w // 4, cy + h // 4))
        else:
            # Synthetic: gradient pattern
            import numpy as np
            arr = np.zeros((64, 64, 3), dtype="uint8")
            arr[:, :, 0] = (i * 8) % 256          # R channel sweeps
            arr[:, :, 1] = ((i * 13) + 64) % 256  # G channel
            arr[:, :, 2] = 128                     # B fixed
            b64 = base64.b64encode(arr.tobytes()).decode("ascii")
            frames_b64.append(b64)
            continue

        # Resize to 64x64 RGB
        small = crop.convert("RGB").resize((64, 64), Image.LANCZOS)
        import numpy as np
        arr = np.array(small, dtype="uint8")  # (64, 64, 3)
        b64 = base64.b64encode(arr.tobytes()).decode("ascii")
        frames_b64.append(b64)

    print(f"  Generated {len(frames_b64)} frames  ({len(frames_b64[0])} chars each)")
    return frames_b64


def patch_master_corpus(frames_b64: list):
    """Read master_corpus.jsonl, inject pixel_frame_base64 into each _meta, rewrite."""
    records = []
    with MASTER.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    records.append(json.loads(s))
                except Exception:
                    pass

    n_frames = len(frames_b64)
    patched = 0
    for idx, rec in enumerate(records):
        meta = rec.setdefault("_meta", {})
        if not meta.get("pixel_frame_base64"):
            meta["pixel_frame_base64"] = frames_b64[idx % n_frames]
            patched += 1

    with MASTER.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"  Patched {patched}/{len(records)} records with pixel frames.")
    return len(records), patched


if __name__ == "__main__":
    print("=" * 60)
    print("PIXEL FRAME INJECTOR")
    print("=" * 60)
    frames = capture_frame_variants(N_FRAMES)
    total, patched = patch_master_corpus(frames)
    size_kb = MASTER.stat().st_size / 1024
    print(f"  master_corpus.jsonl: {total} records, {size_kb:.1f} KB")
    print("  Done.")
