"""extract_mp4_frames.py — extract real pixel frames from screen recording, inject into master corpus"""
import sys, json, base64, time, subprocess
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
MASTER = ROOT / "output" / "master_corpus.jsonl"
VIDEO  = Path(r"D:\Users\archi\Downloads\screen_rec_1779777342.mp4")
FRAMES_DIR = ROOT / "output" / "_mp4_frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

N_FRAMES = 64  # extract 64 evenly-spaced frames

def extract_frames():
    # Use ffmpeg to extract N evenly-spaced frames, resize to 64x64
    cmd = [
        "ffmpeg", "-y", "-i", str(VIDEO),
        "-vf", f"fps=1/3,scale=64:64",
        "-frames:v", str(N_FRAMES),
        str(FRAMES_DIR / "frame_%04d.png")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    frames = sorted(FRAMES_DIR.glob("frame_*.png"))
    print(f"  Extracted {len(frames)} frames from {VIDEO.name}")
    return frames

def frames_to_b64(frame_paths):
    from PIL import Image
    import numpy as np
    out = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB").resize((64, 64))
        arr = __import__("numpy").array(img, dtype="uint8")
        out.append(base64.b64encode(arr.tobytes()).decode("ascii"))
    return out

def patch_corpus(frames_b64):
    records = []
    with MASTER.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try: records.append(json.loads(s))
                except: pass
    n = len(frames_b64)
    patched = 0
    for idx, rec in enumerate(records):
        rec.setdefault("_meta", {})["pixel_frame_base64"] = frames_b64[idx % n]
        patched += 1
    with MASTER.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Patched {patched} records with REAL mp4 frames")
    return patched

if __name__ == "__main__":
    print("=" * 56)
    print("MP4 FRAME INJECTOR — real pixel training signal")
    print("=" * 56)
    frames = extract_frames()
    if not frames:
        print("  ERROR: no frames extracted"); sys.exit(1)
    b64s = frames_to_b64(frames)
    print(f"  Encoded {len(b64s)} frames ({len(b64s[0])} chars each)")
    patch_corpus(b64s)
    kb = MASTER.stat().st_size / 1024
    print(f"  Corpus: {kb:.1f} KB — done.")
