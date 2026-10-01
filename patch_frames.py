import sys, json, base64
from pathlib import Path
from PIL import Image
import numpy as np

MASTER = Path("output/master_corpus.jsonl")
FRAMES_DIR = Path("output/_mp4_frames")

frames = sorted(FRAMES_DIR.glob("frame_*.png"))
print(f"Found {len(frames)} frames")

b64s = []
for p in frames:
    img = Image.open(p).convert("RGB").resize((64, 64))
    arr = np.array(img, dtype="uint8")
    b64s.append(base64.b64encode(arr.tobytes()).decode("ascii"))

print(f"Encoded {len(b64s)} real frames ({len(b64s[0])} chars each)")

records = []
with open(MASTER, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if s:
            try:
                records.append(json.loads(s))
            except Exception:
                pass

n = len(b64s)
for idx, rec in enumerate(records):
    rec.setdefault("_meta", {})["pixel_frame_base64"] = b64s[idx % n]

with open(MASTER, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

kb = MASTER.stat().st_size / 1024
print(f"Patched {len(records)} records with REAL mp4 frames — {kb:.1f} KB")
