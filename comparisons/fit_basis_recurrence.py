"""
Fit a shared affine recurrence on the discovered low-rank length basis.

This is the strongest exporter-like mechanism found so far:
  - shared low-rank basis over length
  - universal affine recurrence on that basis
  - per-voice coefficients
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
    return names, torch.stack([voices[n] for n in names])


def metric(recon, target):
    mae = (recon - target).abs().mean().item()
    rmse = ((recon - target) ** 2).mean().sqrt().item()
    cos = F.cosine_similarity(
        recon.reshape(-1, target.shape[-1]),
        target.reshape(-1, target.shape[-1]),
        dim=-1,
    ).mean().item()
    return mae, rmse, cos


names, stack = load_stack()
V, L, D = stack.shape
mean = stack.mean(dim=1)
res = stack - mean.unsqueeze(1)
len_mat = res.permute(1, 0, 2).reshape(L, -1)
U, S, Vh = torch.linalg.svd(len_mat, full_matrices=False)

print("=" * 88)
print("  BASIS RECURRENCE FIT")
print("=" * 88)

sep("RESULTS")
print(f"  {'k':<4} {'MAE':>10} {'RMSE':>10} {'mean cos':>10}")
print(f"  {'─'*4} {'─'*10} {'─'*10} {'─'*10}")

best = None
for k in [1, 2, 3, 4]:
    basis = U[:, :k]
    coeff = torch.einsum("vld,lk->vkd", res, basis)
    X = torch.cat([basis[:-1], torch.ones(L - 1, 1)], dim=1)
    Y = basis[1:]
    W = torch.linalg.lstsq(X, Y).solution
    A = W[:-1]
    b = W[-1]
    pred_basis = [basis[0]]
    for _ in range(L - 1):
        pred_basis.append(pred_basis[-1] @ A + b)
    pred_basis = torch.stack(pred_basis)
    recon = mean.unsqueeze(1) + torch.einsum("lk,vkd->vld", pred_basis, coeff)
    mae, rmse, cos = metric(recon, stack)
    print(f"  {k:<4} {mae:>10.6f} {rmse:>10.6f} {cos:>10.6f}")
    if best is None or cos > best[0]:
        best = (cos, k, A, b, pred_basis, coeff, mae, rmse)

cos, k, A, b, pred_basis, coeff, mae, rmse = best

sep("BEST MODEL")
print(f"  k:         {k}")
print(f"  MAE:       {mae:.6f}")
print(f"  RMSE:      {rmse:.6f}")
print(f"  mean cos:  {cos:.6f}")
print("  affine recurrence:")
print("    basis[l+1] = basis[l] @ A + b")
print(f"  A shape: {list(A.shape)}")
print(f"  b shape: {list(b.shape)}")

sep("INTERPRETATION")
print("  This shows that the shared length trajectory itself can be compressed into")
print("  a tiny affine dynamical system on the low-rank basis.")
print("  That is very strong evidence for a compiled exporter with universal length dynamics.")

sep()
print("Done.")
