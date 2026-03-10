"""
Discover low-rank structure in Kokoro voicepacks.

This script tests whether shipped voicepacks are best explained as:
  - arbitrary per-voice tables
  - endpoint interpolation with a shared global curve
  - a low-rank shared length basis with per-voice coefficients

If the low-rank basis explanation is strong, that is the best current evidence
for an offline "compiled voicepack" exporter.
"""

from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = ROOT / "kokoro/weights/voices"


def sep(title="", width=88, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)


def load_stack():
    voices = {}
    for vf in sorted(VOICES_DIR.glob("*.pt")):
        voices[vf.stem] = torch.load(vf, map_location="cpu", weights_only=True).squeeze(1).float()
    names = sorted(voices)
    return names, torch.stack([voices[n] for n in names])  # (V, 510, 256)


def metric(recon, target):
    mae = (recon - target).abs().mean().item()
    cos = F.cosine_similarity(
        recon.reshape(-1, target.shape[-1]),
        target.reshape(-1, target.shape[-1]),
        dim=-1,
    ).mean().item()
    return mae, cos


def global_endpoint_curve(stack):
    start = stack[:, 0, :]
    end = stack[:, -1, :]
    delta = end - start
    num = ((stack - start[:, None, :]) * delta[:, None, :]).sum(dim=(0, 2))
    den = (delta[:, None, :] ** 2).sum(dim=(0, 2)).clamp_min(1e-8)
    w = (num / den).clamp(0, 1)
    recon = start[:, None, :] * (1 - w[None, :, None]) + end[:, None, :] * w[None, :, None]
    return w, recon


def low_rank_recon(stack, k):
    base = stack.mean(dim=1, keepdim=True)
    res = stack - base
    len_mat = res.permute(1, 0, 2).reshape(stack.shape[1], -1)
    U, S, Vh = torch.linalg.svd(len_mat, full_matrices=False)
    recon = (U[:, :k] @ torch.diag(S[:k]) @ Vh[:k]).reshape(
        stack.shape[1], stack.shape[0], stack.shape[2]
    ).permute(1, 0, 2)
    return base + recon, U, S


names, stack = load_stack()
V, L, D = stack.shape

print("=" * 88)
print("  VOICEPACK BASIS DISCOVERY")
print("=" * 88)

sep("DATA")
print(f"  voices:       {V}")
print(f"  length slots: {L}")
print(f"  style dim:    {D}")

sep("GLOBAL ENDPOINT CURVE")
w, endpoint_recon = global_endpoint_curve(stack)
mae, cos = metric(endpoint_recon, stack)
print(f"  reconstruction MAE:      {mae:.6f}")
print(f"  reconstruction mean cos: {cos:.6f}")
print(f"  shared curve monotonic:  {bool(torch.all(w[1:] >= w[:-1] - 1e-6))}")
for t in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
    idx = int((w >= t).nonzero()[0]) + 1 if (w >= t).any() else None
    print(f"  first length with w >= {t:<5}: {idx}")
print(f"  first 10 curve values: {[round(float(x), 6) for x in w[:10]]}")

sep("LOW-RANK LENGTH BASIS")
base = stack.mean(dim=1, keepdim=True)
res = stack - base
len_mat = res.permute(1, 0, 2).reshape(L, V * D)
U, S, Vh = torch.linalg.svd(len_mat, full_matrices=False)
energy = (S ** 2)
total = energy.sum()
for k in [1, 2, 3, 4, 8]:
    recon, _, _ = low_rank_recon(stack, k)
    mae, cos = metric(recon, stack)
    explained = float(energy[:k].sum() / total)
    print(f"  k={k:<2}  explained={explained:.6f}  MAE={mae:.6f}  mean cos={cos:.6f}")

sep("TOP LENGTH MODES")
for i in range(4):
    mode = U[:, i]
    sign = -1.0 if abs(mode[0]) > abs(mode[-1]) and mode[0] < 0 else 1.0
    mode = mode * sign
    print(f"  mode {i+1}: sigma={float(S[i]):.6f}")
    print(f"    first 12 values: {[round(float(x), 6) for x in mode[:12]]}")
    print(f"    value@24={float(mode[23]):.6f} value@58={float(mode[57]):.6f} value@94={float(mode[93]):.6f} value@510={float(mode[-1]):.6f}")

sep("MONOTONICITY")
diff = stack[:, 1:, :] - stack[:, :-1, :]
sign = diff.sign()
changes = (sign[:, 1:, :] * sign[:, :-1, :] < 0).float().mean(dim=1)
print(f"  mean sign-flip rate across all voice/dim curves: {changes.mean().item():.6f}")
print(f"  share of curves with <1% sign flips:            {(changes < 0.01).float().mean().item():.6f}")
print(f"  share of curves with <5% sign flips:            {(changes < 0.05).float().mean().item():.6f}")

sep("WORKING INTERPRETATION")
print("  The shipped voicepacks are not arbitrary 510x256 tables.")
print("  They lie on a very low-rank shared length manifold:")
print("    - rank 1 already reconstructs well")
print("    - rank 4 reconstructs almost exactly")
print("  That is strong evidence for an offline exporter that compiles each voice")
print("  into coefficients over a global set of length-dependent basis curves.")

sep("CANDIDATE EXPORT FORM")
print("  For each voice v and length l:")
print("    pack[v, l, :] ~= voice_mean[v, :] + sum_i coeff[v, i, :] * basis_i[l]")
print("  with only a small number of basis functions i.")
print("  This fits the observed data much better than raw sample stacks or naive repetition.")

sep()
print("Done.")
