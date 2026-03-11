"""
Fit a concrete voicepack family and test generalization across voices.

Main test:
  Can a shared low-rank length basis be combined with a deterministic map
  from a voice's first-slot style vector to the full per-voice coefficient set?

If yes, that is strong evidence for a deterministic exporter from ref_s-like
voice identity, rather than per-length stochastic sampling.
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
    length_matrix = residual.permute(1, 0, 2).reshape(stack.shape[1], -1)
    U, S, _ = torch.linalg.svd(length_matrix, full_matrices=False)
    basis = U[:, :k]  # (L, K)
    coeff = torch.einsum("vld,lk->vkd", residual, basis)  # (V, K, D)
    return voice_mean, basis, coeff, S


def ridge_predict(train_x, train_y, test_x, lam=1e-3):
    # train_x: (N, F), train_y: (N, O), test_x: (F,)
    x = train_x
    y = train_y
    xtx = x.T @ x
    eye = torch.eye(xtx.shape[0], dtype=x.dtype)
    w = torch.linalg.solve(xtx + lam * eye, x.T @ y)  # (F, O)
    return test_x @ w, w


def reconstruct_from_predicted_coeff(voice_mean, basis, coeff_pred):
    return voice_mean.unsqueeze(0) + torch.einsum("kd,lk->ld", coeff_pred, basis)


def run_leave_one_out(names, stack, k=4, lam=1e-3):
    voice_mean, basis, coeff, _ = learn_basis(stack, k)
    x_all = stack[:, 0, :]  # first slot as ref_s proxy, (V, 256)
    y_all = coeff.reshape(coeff.shape[0], -1)  # (V, K*D)

    rows = []
    for i, name in enumerate(names):
        train_mask = torch.ones(len(names), dtype=torch.bool)
        train_mask[i] = False
        pred_flat, _ = ridge_predict(x_all[train_mask], y_all[train_mask], x_all[i], lam=lam)
        coeff_pred = pred_flat.reshape(k, stack.shape[-1])
        recon = reconstruct_from_predicted_coeff(voice_mean[i], basis, coeff_pred)
        mae, rmse, cos = metric(recon.unsqueeze(0), stack[i:i+1])
        rows.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "cos": cos,
        })
    return rows, basis, voice_mean, coeff


names, stack = load_stack()

print("=" * 88)
print("  VOICEPACK FAMILY FIT HARNESS")
print("=" * 88)

sep("GOAL")
print("  Test whether the full voicepack can be predicted from the first-slot style")
print("  vector using a shared low-rank length basis and a deterministic map to")
print("  per-voice coefficients.")

results = {}
for k in [1, 2, 4, 8]:
    rows, basis, voice_mean, coeff = run_leave_one_out(names, stack, k=k, lam=1e-2)
    mae = sum(r["mae"] for r in rows) / len(rows)
    rmse = sum(r["rmse"] for r in rows) / len(rows)
    cos = sum(r["cos"] for r in rows) / len(rows)
    results[k] = (rows, mae, rmse, cos)

sep("LEAVE-ONE-OUT RESULTS")
print(f"  {'k':<4} {'mean MAE':>10} {'mean RMSE':>11} {'mean cos':>10}")
print(f"  {'─'*4} {'─'*10} {'─'*11} {'─'*10}")
for k in [1, 2, 4, 8]:
    _, mae, rmse, cos = results[k]
    print(f"  {k:<4} {mae:>10.6f} {rmse:>11.6f} {cos:>10.6f}")

best_k = max(results, key=lambda k: results[k][3])
rows, mae, rmse, cos = results[best_k]

sep("BEST SETTING")
print(f"  best k:        {best_k}")
print(f"  mean MAE:      {mae:.6f}")
print(f"  mean RMSE:     {rmse:.6f}")
print(f"  mean cosine:   {cos:.6f}")

rows = sorted(rows, key=lambda r: r["cos"], reverse=True)
sep("TOP-10 VOICES")
for r in rows[:10]:
    print(f"  {r['name']:<20} cos={r['cos']:.6f}  mae={r['mae']:.6f}")

sep("BOTTOM-10 VOICES")
for r in rows[-10:]:
    print(f"  {r['name']:<20} cos={r['cos']:.6f}  mae={r['mae']:.6f}")

sep("INTERPRETATION")
print("  This is a strict test: each held-out voice is reconstructed from its")
print("  first-slot vector only, using a global basis and a deterministic map")
print("  learned from the other voices.")
print("  If this works well, then most of the exporter is explainable as")
print("  ref_s-like identity -> coefficient prediction -> shared length basis.")

sep()
print("Done.")
