"""
Compare legacy v0.19 voicepacks against current v1.0 voicepacks.

Why this matters:
  - legacy public evidence already exposed exact voicepack arithmetic
  - if the same table format predates v1.0, that supports the idea that
    voicepacks are first-class learned artifacts, not a late export hack
"""

from pathlib import Path
import requests
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = ROOT / "kokoro/weights/voices"
LEGACY_DIR = Path("/tmp/klegacy_voices")
LEGACY_BASE = "https://huggingface.co/hexgrad/kLegacy/resolve/main/v0.19/voices"
LEGACY_FILES = ["af.pt", "af_bella.pt", "af_sarah.pt", "am_adam.pt"]


def sep(title="", width=88, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)


def ensure_legacy():
    LEGACY_DIR.mkdir(exist_ok=True)
    for name in LEGACY_FILES:
        path = LEGACY_DIR / name
        if path.exists():
            continue
        r = requests.get(f"{LEGACY_BASE}/{name}", timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)


def const_prefix_len(x, tol=1e-7):
    d = (x[1:] - x[:-1]).abs().amax(dim=-1)
    n = 0
    for val in d:
        if float(val) <= tol:
            n += 1
        else:
            break
    return n + 1


ensure_legacy()

print("=" * 88)
print("  LEGACY VS CURRENT VOICEPACKS")
print("=" * 88)

sep("LEGACY SHAPES")
for name in LEGACY_FILES:
    x = torch.load(LEGACY_DIR / name, map_location="cpu", weights_only=True).squeeze(1).float()
    print(f"  {name:<12} shape={list(x.shape)}  const-prefix={const_prefix_len(x)}")

sep("LEGACY EXACT ARITHMETIC")
af = torch.load(LEGACY_DIR / "af.pt", map_location="cpu", weights_only=True).float()
af_bella = torch.load(LEGACY_DIR / "af_bella.pt", map_location="cpu", weights_only=True).float()
af_sarah = torch.load(LEGACY_DIR / "af_sarah.pt", map_location="cpu", weights_only=True).float()
avg = torch.mean(torch.stack([af_bella, af_sarah]), dim=0)
print(f"  af.pt == mean(af_bella.pt, af_sarah.pt): {torch.equal(af, avg)}")

sep("CURRENT SHAPES")
for name in ["af_bella", "af_sarah", "am_adam", "af_heart"]:
    x = torch.load(CURRENT_DIR / f"{name}.pt", map_location="cpu", weights_only=True).squeeze(1).float()
    print(f"  {name:<10} shape={list(x.shape)}  const-prefix={const_prefix_len(x)}")

sep("SHORT-LENGTH BEHAVIOR")
legacy = torch.load(LEGACY_DIR / "af_bella.pt", map_location="cpu", weights_only=True).squeeze(1).float()
current = torch.load(CURRENT_DIR / "af_bella.pt", map_location="cpu", weights_only=True).squeeze(1).float()
print(f"  legacy adjacent cosine:  {float(F.cosine_similarity(legacy[:-1], legacy[1:], dim=-1).mean()):.6f}")
print(f"  current adjacent cosine: {float(F.cosine_similarity(current[:-1], current[1:], dim=-1).mean()):.6f}")
print(f"  legacy row0==row1:       {torch.equal(legacy[0], legacy[1])}")
print(f"  current row1==row2:      {torch.equal(current[0], current[1])}")

sep("INTERPRETATION")
print("  v0.19 already shipped full voice-length style tables.")
print("  Those tables supported exact arithmetic at the artifact level.")
print("  v1.0 keeps the same general artifact class, but replaces the old flat")
print("  short-length prefix with a smoother decay from the first slot onward.")

sep("WHY THIS MATTERS")
print("  This makes a training-time or jointly optimized voice-table hypothesis")
print("  more plausible than a one-off exporter from single reference embeddings.")

sep()
print("Done.")
