"""
Test how much of a voicepack is determined by its first few slots.

If a small prefix of the table predicts the full table nearly exactly, then the
unknown exporter is likely generating a smooth deterministic trajectory from an
early-length anchor state, not independently sampling all 510 slots.
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


def learn_basis(stack, k):
    mean = stack.mean(dim=1)
    residual = stack - mean.unsqueeze(1)
    length_matrix = residual.permute(1, 0, 2).reshape(stack.shape[1], -1)
    U, _, _ = torch.linalg.svd(length_matrix, full_matrices=False)
    basis = U[:, :k]
    coeff = torch.einsum("vld,lk->vkd", residual, basis)
    return mean, basis, coeff


def ridge_predict(train_x, train_y, test_x, lam=1e-2):
    xtx = train_x.T @ train_x
    w = torch.linalg.solve(xtx + lam * torch.eye(xtx.shape[0]), train_x.T @ train_y)
    return test_x @ w


def reconstruct(mean, basis, coeff_pred):
    return mean.unsqueeze(0) + torch.einsum("kd,lk->ld", coeff_pred, basis)


names, stack = load_stack()
print("=" * 88)
print("  PREFIX-SLOT FIT TEST")
print("=" * 88)

sep("RESULTS")
print(f"  {'prefix slots':<12} {'k':<4} {'mean MAE':>10} {'mean RMSE':>11} {'mean cos':>10}")
print(f"  {'─'*12} {'─'*4} {'─'*10} {'─'*11} {'─'*10}")

best = None
for prefix in [1, 2, 4, 8]:
    x = stack[:, :prefix, :].reshape(stack.shape[0], -1)
    for k in [1, 2, 4]:
        mean, basis, coeff = learn_basis(stack, k)
        y = coeff.reshape(coeff.shape[0], -1)
        rows = []
        for i in range(len(names)):
            mask = torch.ones(len(names), dtype=torch.bool)
            mask[i] = False
            pred = ridge_predict(x[mask], y[mask], x[i])
            coeff_pred = pred.reshape(k, stack.shape[-1])
            recon = reconstruct(mean[i], basis, coeff_pred)
            mae, rmse, cos = metric(recon.unsqueeze(0), stack[i:i+1])
            rows.append((mae, rmse, cos, names[i]))
        mae = sum(r[0] for r in rows) / len(rows)
        rmse = sum(r[1] for r in rows) / len(rows)
        cos = sum(r[2] for r in rows) / len(rows)
        print(f"  {prefix:<12} {k:<4} {mae:>10.6f} {rmse:>11.6f} {cos:>10.6f}")
        cand = (cos, mae, rmse, prefix, k, rows)
        if best is None or cand[0] > best[0]:
            best = cand

cos, mae, rmse, prefix, k, rows = best
rows = sorted(rows, key=lambda r: r[2])
sep("BEST SETTING")
print(f"  prefix slots: {prefix}")
print(f"  k:            {k}")
print(f"  mean MAE:     {mae:.6f}")
print(f"  mean RMSE:    {rmse:.6f}")
print(f"  mean cos:     {cos:.6f}")

sep("HARDEST VOICES")
for r in rows[:10]:
    print(f"  {r[3]:<20} cos={r[2]:.6f}  mae={r[0]:.6f}")

sep("INTERPRETATION")
print("  If a tiny prefix predicts the whole pack well, the exporter likely evolves")
print("  a compact state over length rather than independently generating far-length slots.")

sep()
print("Done.")
