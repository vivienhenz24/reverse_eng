"""
Deep analysis of Kokoro voice pack tensors.

Each voice file is a tensor of shape (510, 1, 256):
  [:, :, :128]  → decoder style (fed into iSTFTNet decoder)
  [:, :, 128:]  → prosody style (fed into ProsodyPredictor / DurationEncoder)

Important runtime detail:
  Kokoro inference uses pack[len(phonemes)-1], so the first axis is a
  length-conditioned lookup table, not a bag of random reference embeddings.

This script covers:
  - Shape / dtype confirmation for every voice
  - Per-voice statistics (mean, std, norm, min, max)
  - Split analysis: decoder-half vs prosody-half
  - Length-axis smoothness analysis
  - Inter-voice cosine similarity matrix
  - Clustering by language prefix (af_, am_, bf_, bm_, jf_, ...)
  - Top-10 most / least similar voice pairs
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT        = Path(__file__).resolve().parent.parent
VOICES_DIR  = ROOT / "kokoro" / "weights" / "voices"

def sep(title="", width=72, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)

# ── Load all voices ─────────────────────────────────────────────────────────────

voice_files = sorted(VOICES_DIR.glob("*.pt"))
print("=" * 72)
print("  KOKORO VOICE PACK — DEEP ANALYSIS")
print("=" * 72)
print(f"\nFound {len(voice_files)} voice files in {VOICES_DIR}\n")

voices_raw = {}
for vf in voice_files:
    t = torch.load(vf, map_location="cpu", weights_only=True)
    voices_raw[vf.stem] = t          # shape: (510, 1, 256)

# Squeeze dim 1 → (510, 256). We keep raw for frame analysis,
# and compute a mean-pooled representative vector for similarity.
voices = {n: t.squeeze(1) for n, t in voices_raw.items()}   # (510, 256)
voices_mean = {n: t.mean(dim=0) for n, t in voices.items()} # (256,)

# ── Shape & dtype check ─────────────────────────────────────────────────────────

sep("SHAPE & DTYPE")
shapes = {n: tuple(t.shape) for n, t in voices_raw.items()}
unique_shapes = set(shapes.values())
print(f"  Unique raw shapes: {unique_shapes}")
print(f"  Interpretation: (length_index=510, batch=1, style_dim=256)")
print(f"  Runtime usage: pipeline selects pack[len(phonemes)-1].")
print(f"  The 510 axis is therefore a length-conditioned style trajectory.")
print()
for name, t in voices_raw.items():
    print(f"  {name:<25} shape={list(t.shape)}  dtype={t.dtype}")

# ── Language prefix grouping ────────────────────────────────────────────────────

sep("LANGUAGE / GENDER PREFIX GROUPS")
prefix_map = {
    "af": "American English – Female",
    "am": "American English – Male",
    "bf": "British English – Female",
    "bm": "British English – Male",
    "ef": "Spanish – Female",
    "em": "Spanish – Male",
    "ff": "French – Female",
    "hf": "Hindi – Female",
    "hm": "Hindi – Male",
    "if": "Italian – Female",
    "im": "Italian – Male",
    "jf": "Japanese – Female",
    "jm": "Japanese – Male",
    "pf": "Portuguese – Female",
    "pm": "Portuguese – Male",
    "zf": "Mandarin – Female",
    "zm": "Mandarin – Male",
}
groups = {}
for name in voices:
    pfx = name[:2]
    groups.setdefault(pfx, []).append(name)

for pfx, members in sorted(groups.items()):
    lang = prefix_map.get(pfx, "Unknown")
    print(f"  {pfx}  ({lang}): {len(members)} voices")
    for m in members:
        print(f"    {m}")

# ── Per-voice statistics ────────────────────────────────────────────────────────

sep("PER-VOICE STATISTICS  (mean-pooled 256-dim representative)")
print(f"  {'Name':<25} {'norm':>7} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
print(f"  {'─'*25} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
for name, v in voices_mean.items():
    v = v.float()
    print(f"  {name:<25} {v.norm():.4f} {v.mean():+.4f} {v.std():.4f} {v.min():+.4f} {v.max():+.4f}")

sep("PER-VOICE LENGTH-AXIS VARIANCE  (how much style drifts across 1..510)")
print(f"  {'Name':<25} {'len_std':>10} {'len_norm_std':>15} {'n_steps':>9}")
print(f"  {'─'*25} {'─'*10} {'─'*15} {'─'*9}")
for name, t in voices.items():
    t = t.float()
    step_std = t.std(dim=0).mean().item()
    step_norms = t.norm(dim=1)
    print(f"  {name:<25} {step_std:>10.4f} {step_norms.std().item():>15.4f} {t.shape[0]:>9}")

sep("LENGTH-AXIS SMOOTHNESS")
all_steps = torch.stack([voices[n].float() for n in sorted(voices)])  # (N, 510, 256)
adj = F.cosine_similarity(all_steps[:, :-1, :], all_steps[:, 1:, :], dim=-1)
end = F.cosine_similarity(all_steps[:, 0, :], all_steps[:, -1, :], dim=-1)
print(f"  Adjacent-step cosine: mean={adj.mean():.6f}  std={adj.std():.6f}  min={adj.min():.6f}")
print(f"  First-vs-last cosine: mean={end.mean():.6f}  std={end.std():.6f}  min={end.min():.6f}  max={end.max():.6f}")
print(f"  Interpretation: adjacent rows are almost identical, but long-range drift is real.")

# ── Decoder half vs prosody half ───────────────────────────────────────────────

sep("STYLE VECTOR SPLIT  (decoder [:128] vs prosody [128:], mean-pooled)")
print(f"  {'Name':<25} {'dec_norm':>9} {'dec_std':>8} {'pro_norm':>9} {'pro_std':>8}")
print(f"  {'─'*25} {'─'*9} {'─'*8} {'─'*9} {'─'*8}")
for name, v in voices_mean.items():
    v = v.float()
    dec = v[:128]
    pro = v[128:]
    print(f"  {name:<25} {dec.norm():9.4f} {dec.std():8.4f} {pro.norm():9.4f} {pro.std():8.4f}")

# ── Inter-voice cosine similarity ──────────────────────────────────────────────

sep("COSINE SIMILARITY MATRIX  (mean-pooled 256-dim)")
names = list(voices_mean.keys())
matrix = torch.stack([voices_mean[n].float() for n in names])   # (N, 256)
matrix = F.normalize(matrix, dim=1)
sim = matrix @ matrix.T          # (N, N)

# print compact matrix
col_w = 7
print("\n  " + " " * 25 + "".join(f"{n[:col_w]:>{col_w}}" for n in names))
for i, name_i in enumerate(names):
    row = "  " + f"{name_i:<25}"
    for j in range(len(names)):
        row += f"{sim[i,j].item():>{col_w}.3f}"
    print(row)

# ── Top similar / dissimilar pairs ────────────────────────────────────────────

sep("TOP-10 MOST SIMILAR VOICE PAIRS  (cosine, mean-pooled)")
pairs = []
for i in range(len(names)):
    for j in range(i+1, len(names)):
        pairs.append((sim[i,j].item(), names[i], names[j]))
pairs.sort(reverse=True)
print(f"  {'Score':>7}  Pair")
for score, a, b in pairs[:10]:
    print(f"  {score:7.4f}  {a}  ↔  {b}")

sep("TOP-10 LEAST SIMILAR VOICE PAIRS  (cosine, mean-pooled)")
for score, a, b in pairs[-10:][::-1]:
    print(f"  {score:7.4f}  {a}  ↔  {b}")

# ── Within-group vs cross-group similarity ─────────────────────────────────────

sep("WITHIN-GROUP vs CROSS-GROUP SIMILARITY")
within_scores, cross_scores = [], []
for i in range(len(names)):
    for j in range(i+1, len(names)):
        pfx_i = names[i][:2]
        pfx_j = names[j][:2]
        s = sim[i, j].item()
        if pfx_i == pfx_j:
            within_scores.append(s)
        else:
            cross_scores.append(s)

def stats_str(lst):
    if not lst:
        return "N/A"
    t = torch.tensor(lst)
    return f"n={len(lst):3d}  mean={t.mean():.4f}  std={t.std():.4f}  min={t.min():.4f}  max={t.max():.4f}"

print(f"  Within-group: {stats_str(within_scores)}")
print(f"  Cross-group:  {stats_str(cross_scores)}")

# ── Per-dimension statistics across all voices ─────────────────────────────────

sep("PER-DIMENSION VARIANCE  (which dims are most expressive?)")
all_mat = torch.stack([voices_mean[n].float() for n in names])   # (N, 256)
dim_var  = all_mat.var(dim=0)   # (256,)
dim_mean = all_mat.mean(dim=0)

dec_var = dim_var[:128]
pro_var = dim_var[128:]

print(f"\n  Decoder half  (dims 0–127):")
print(f"    mean variance: {dec_var.mean():.6f}")
print(f"    max variance:  {dec_var.max():.6f}  at dim {dec_var.argmax().item()}")
print(f"    min variance:  {dec_var.min():.6f}  at dim {dec_var.argmin().item()}")

print(f"\n  Prosody half  (dims 128–255):")
print(f"    mean variance: {pro_var.mean():.6f}")
print(f"    max variance:  {pro_var.max():.6f}  at dim {(pro_var.argmax().item()+128)}")
print(f"    min variance:  {pro_var.min():.6f}  at dim {(pro_var.argmin().item()+128)}")

top_k = 10
topk_idx = dim_var.topk(top_k).indices.tolist()
print(f"\n  Top-{top_k} highest-variance dims (most discriminative across voices):")
for d in topk_idx:
    half = "decoder" if d < 128 else "prosody"
    print(f"    dim {d:3d}  ({half})  var={dim_var[d]:.6f}  mean={dim_mean[d]:+.4f}")

sep()
print("Done.")
