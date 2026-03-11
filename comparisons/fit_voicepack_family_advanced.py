"""
Advanced voicepack family fitting.

Compare multiple deterministic coefficient-prediction paths:
  - global ridge on first-slot style
  - separate decoder/prosody ridge
  - prefix-conditioned ridge
  - nearest-neighbor coefficient copy
  - RBF kernel regression

All tests are leave-one-out over voices.
"""

from pathlib import Path
import math
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
    return names, torch.stack([voices[n] for n in names])  # (V,L,D)


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


def rbf_predict(train_x, train_y, test_x, gamma):
    d2 = ((train_x - test_x.unsqueeze(0)) ** 2).sum(dim=1)
    w = torch.softmax(-gamma * d2, dim=0)
    return w @ train_y


def reconstruct(mean, basis, coeff_pred):
    return mean.unsqueeze(0) + torch.einsum("kd,lk->ld", coeff_pred, basis)


def evaluate(names, stack, k, family):
    mean, basis, coeff = learn_basis(stack, k)
    x = stack[:, 0, :]  # ref_s proxy
    y = coeff.reshape(coeff.shape[0], -1)
    prefixes = [n[:2] for n in names]
    rows = []
    for i, name in enumerate(names):
        mask = torch.ones(len(names), dtype=torch.bool)
        mask[i] = False
        train_x = x[mask]
        train_y = y[mask]

        if family == "global_ridge":
            pred = ridge_predict(train_x, train_y, x[i])
        elif family == "split_ridge":
            yk = coeff[mask]  # (N,K,D)
            pred_dec = ridge_predict(train_x[:, :128], yk[:, :, :128].reshape(yk.shape[0], -1), x[i, :128])
            pred_pro = ridge_predict(train_x[:, 128:], yk[:, :, 128:].reshape(yk.shape[0], -1), x[i, 128:])
            pred = torch.cat([pred_dec, pred_pro], dim=0)
        elif family == "prefix_ridge":
            prefix = prefixes[i]
            idx = torch.tensor([p == prefix for p in prefixes], dtype=torch.bool)
            idx[i] = False
            if idx.sum() >= 2:
                pred = ridge_predict(x[idx], y[idx], x[i])
            else:
                pred = ridge_predict(train_x, train_y, x[i])
        elif family == "nearest_neighbor":
            sims = F.cosine_similarity(train_x, x[i].unsqueeze(0), dim=1)
            pred = train_y[sims.argmax()]
        elif family == "rbf":
            pred = rbf_predict(train_x, train_y, x[i], gamma=0.5 / x.shape[1])
        else:
            raise ValueError(family)

        coeff_pred = pred.reshape(k, stack.shape[-1])
        recon = reconstruct(mean[i], basis, coeff_pred)
        mae, rmse, cos = metric(recon.unsqueeze(0), stack[i:i+1])
        rows.append({"name": name, "mae": mae, "rmse": rmse, "cos": cos})

    mae = sum(r["mae"] for r in rows) / len(rows)
    rmse = sum(r["rmse"] for r in rows) / len(rows)
    cos = sum(r["cos"] for r in rows) / len(rows)
    return rows, mae, rmse, cos


names, stack = load_stack()
families = ["global_ridge", "split_ridge", "prefix_ridge", "nearest_neighbor", "rbf"]

print("=" * 88)
print("  ADVANCED VOICEPACK FAMILY FIT")
print("=" * 88)

sep("RESULTS")
print(f"  {'family':<18} {'k':<4} {'mean MAE':>10} {'mean RMSE':>11} {'mean cos':>10}")
print(f"  {'─'*18} {'─'*4} {'─'*10} {'─'*11} {'─'*10}")

all_results = []
for family in families:
    for k in [1, 2, 4]:
        rows, mae, rmse, cos = evaluate(names, stack, k, family)
        all_results.append((family, k, rows, mae, rmse, cos))
        print(f"  {family:<18} {k:<4} {mae:>10.6f} {rmse:>11.6f} {cos:>10.6f}")

best = max(all_results, key=lambda x: x[5])
family, k, rows, mae, rmse, cos = best

sep("BEST MODEL")
print(f"  family:     {family}")
print(f"  k:          {k}")
print(f"  mean MAE:   {mae:.6f}")
print(f"  mean RMSE:  {rmse:.6f}")
print(f"  mean cos:   {cos:.6f}")

rows = sorted(rows, key=lambda r: r["cos"])
sep("HARDEST VOICES")
for r in rows[:10]:
    print(f"  {r['name']:<20} cos={r['cos']:.6f}  mae={r['mae']:.6f}")

sep("INTERPRETATION")
print("  If local/deterministic predictors beat nearest-neighbor and diffusion-free")
print("  families stay strong, that supports a structured exporter from ref_s-like")
print("  identity rather than per-voice memorization or noisy per-length sampling.")

sep()
print("Done.")
