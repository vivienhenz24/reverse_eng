"""
Reconstruct a minimal low-rank voicepack exporter from shipped Kokoro packs.

Model:
  pack[v, l, d] ~= mean[v, d] + sum_k coeff[v, k, d] * basis[k, l]

This script learns the shared length bases from the released voicepacks,
solves for per-voice coefficients, and reports reconstruction quality.
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
    return names, torch.stack([voices[n] for n in names])  # (V, L, D)


def metric(recon, target):
    mae = (recon - target).abs().mean().item()
    rmse = ((recon - target) ** 2).mean().sqrt().item()
    cos = F.cosine_similarity(
        recon.reshape(-1, target.shape[-1]),
        target.reshape(-1, target.shape[-1]),
        dim=-1,
    ).mean().item()
    return mae, rmse, cos


def learn_basis(stack, k):
    voice_mean = stack.mean(dim=1)  # (V, D)
    residual = stack - voice_mean.unsqueeze(1)
    length_matrix = residual.permute(1, 0, 2).reshape(stack.shape[1], -1)  # (L, V*D)
    U, S, _ = torch.linalg.svd(length_matrix, full_matrices=False)
    basis = U[:, :k]  # (L, K)
    coeff = torch.einsum("vld,lk->vkd", residual, basis)  # (V, K, D)
    recon = voice_mean.unsqueeze(1) + torch.einsum("vkd,lk->vld", coeff, basis)
    return basis, voice_mean, coeff, recon, S


names, stack = load_stack()
V, L, D = stack.shape

print("=" * 88)
print("  LOW-RANK VOICEPACK EXPORTER RECONSTRUCTION")
print("=" * 88)

sep("DATA")
print(f"  voices:       {V}")
print(f"  length slots: {L}")
print(f"  style dim:    {D}")

sep("RECONSTRUCTION QUALITY")
results = {}
for k in [1, 2, 3, 4]:
    basis, voice_mean, coeff, recon, S = learn_basis(stack, k)
    mae, rmse, cos = metric(recon, stack)
    results[k] = (basis, voice_mean, coeff, recon, S, mae, rmse, cos)
    print(f"  k={k}: MAE={mae:.6f}  RMSE={rmse:.6f}  mean cos={cos:.6f}")

basis, voice_mean, coeff, recon, S, mae, rmse, cos = results[4]

sep("BASIS SUMMARY (k=4)")
energy = (S ** 2)
total = energy.sum()
for i in range(4):
    explained = float(energy[i] / total)
    col = basis[:, i]
    sign = -1.0 if col[0] < 0 else 1.0
    col = col * sign
    print(f"  basis {i+1}: explained={explained:.6f}")
    print(f"    first 10 values: {[round(float(x), 6) for x in col[:10]]}")
    print(f"    value@24={float(col[23]):.6f} value@58={float(col[57]):.6f} value@94={float(col[93]):.6f} value@510={float(col[-1]):.6f}")

sep("EXPORTER FORM")
print("  Candidate exporter:")
print("    1. Store one voice mean vector per voice: mean[v, d]")
print("    2. Store a small shared basis over phoneme length: basis[k, l]")
print("    3. Store per-voice basis coefficients: coeff[v, k, d]")
print("    4. Reconstruct:")
print("         pack[v, l, d] = mean[v, d] + sum_k coeff[v, k, d] * basis[k, l]")
print("  This is equivalent to a compact compiled representation of the shipped packs.")

sep("ONE VOICE EXAMPLE")
idx = 0
name = names[idx]
voice_mae, voice_rmse, voice_cos = metric(recon[idx:idx+1], stack[idx:idx+1])
print(f"  voice: {name}")
print(f"  reconstruction MAE={voice_mae:.6f}  RMSE={voice_rmse:.6f}  mean cos={voice_cos:.6f}")
for i in range(4):
    print(f"  coeff norm basis{i+1}: {coeff[idx, i].norm().item():.6f}")

sep("TAKEAWAY")
print("  A 4-basis shared length manifold reconstructs the released voicepacks almost exactly.")
print("  The released .pt files therefore look like compiled inference tables, not raw samples.")

sep()
print("Done.")
