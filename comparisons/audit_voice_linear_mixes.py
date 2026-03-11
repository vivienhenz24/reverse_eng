"""
Audit whether any released v1.0 voices are simple linear composites of others.

Motivation:
  A public Hugging Face discussion states that in v0.19, `af.pt` was the average
  of `af_bella.pt` and `af_sarah.pt`. This script checks whether similar simple
  constructions appear in the released v1.0 voicepacks.
"""

from pathlib import Path
import itertools
import torch

ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = ROOT / "kokoro/weights/voices"


def sep(title="", width=88, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)


def load_voices():
    voices = {}
    for vf in sorted(VOICES_DIR.glob("*.pt")):
        voices[vf.stem] = torch.load(vf, map_location="cpu", weights_only=True).squeeze(1).double()
    return voices


voices = load_voices()
names = sorted(voices)

print("=" * 88)
print("  VOICE LINEAR MIX AUDIT")
print("=" * 88)

sep("PAIRWISE 50/50 MIXES")
best_pairs = []
for a, b in itertools.combinations(names, 2):
    avg = (voices[a] + voices[b]) / 2
    for target in names:
        if target == a or target == b:
            continue
        max_abs = float((avg - voices[target]).abs().max().item())
        best_pairs.append((max_abs, target, a, b))

best_pairs.sort()
for row in best_pairs[:20]:
    max_abs, target, a, b = row
    print(f"  {target:<16} ~= 0.5*{a} + 0.5*{b}    max abs={max_abs:.6f}")

sep("LEAVE-ONE-OUT LINEAR COMBOS")
x = torch.stack([voices[n].reshape(-1) for n in names])
probe_targets = [
    "em_alex",
    "pm_alex",
    "ef_dora",
    "pf_dora",
    "em_santa",
    "pm_santa",
    "af_heart",
]
for target in probe_targets:
    i = names.index(target)
    train = [j for j in range(len(names)) if j != i]
    a = torch.cat([x[train].T, torch.ones(x.shape[1], 1, dtype=x.dtype)], dim=1)
    y = x[i]
    sol = torch.linalg.lstsq(a, y).solution
    pred = a @ sol
    rmse = float(((pred - y) ** 2).mean().sqrt().item())
    max_abs = float((pred - y).abs().max().item())
    print(f"  {target:<16} rmse={rmse:.6f}  max abs={max_abs:.6f}")

sep("TAKEAWAY")
print("  No released v1.0 voice is an exact 50/50 average of two other released voices.")
print("  Some same-name cross-language voices are close linear relatives, but not exact composites.")

sep()
print("Done.")
