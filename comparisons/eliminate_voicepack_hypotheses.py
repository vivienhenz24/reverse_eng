"""
Eliminate voicepack generation hypotheses by exact reconstruction quality.

This is not a style-matching script. It scores concrete artifact families against
the released Kokoro voicepacks and keeps only families that can reproduce the
stored tensors closely enough to still be plausible as the real exporter.
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
        voices[vf.stem] = torch.load(vf, map_location="cpu", weights_only=True).squeeze(1).double()
    names = sorted(voices)
    return names, torch.stack([voices[n] for n in names])


def metric(recon, target):
    err = recon - target
    mae = err.abs().mean().item()
    rmse = (err.pow(2).mean().sqrt()).item()
    cos = F.cosine_similarity(
        recon.reshape(-1, target.shape[-1]),
        target.reshape(-1, target.shape[-1]),
        dim=-1,
    ).mean().item()
    max_abs = err.abs().max().item()
    return mae, rmse, cos, max_abs


def fit_shared_basis(residual, k):
    mat = residual.permute(1, 0, 2).reshape(residual.shape[1], -1)
    u, s, _ = torch.linalg.svd(mat, full_matrices=False)
    basis = u[:, :k]
    coeff = torch.einsum("vld,lk->vkd", residual, basis)
    recon = torch.einsum("lk,vkd->vld", basis, coeff)
    return basis, coeff, recon, s


def single_curve_first_to_last(stack):
    x0 = stack[:, 0, :]
    x1 = stack[:, -1, :]
    alpha = torch.linspace(0, 1, stack.shape[1], dtype=stack.dtype)
    return x0.unsqueeze(1) * (1 - alpha.view(1, -1, 1)) + x1.unsqueeze(1) * alpha.view(1, -1, 1)


def two_curve_split(stack):
    x0 = stack[:, 0, :]
    x1 = stack[:, -1, :]
    a_dec = torch.linspace(0, 1, stack.shape[1], dtype=stack.dtype).pow(0.35)
    a_pro = torch.linspace(0, 1, stack.shape[1], dtype=stack.dtype).pow(0.18)
    left = x0[:, None, :128] * (1 - a_dec.view(1, -1, 1)) + x1[:, None, :128] * a_dec.view(1, -1, 1)
    right = x0[:, None, 128:] * (1 - a_pro.view(1, -1, 1)) + x1[:, None, 128:] * a_pro.view(1, -1, 1)
    return torch.cat([left, right], dim=-1)


def mean_rank(stack, k):
    mean = stack.mean(dim=1, keepdim=True)
    basis, coeff, recon, _ = fit_shared_basis(stack - mean, k)
    return mean + recon


def terminal_rank(stack, k):
    terminal = stack[:, -1:, :]
    basis, coeff, recon, _ = fit_shared_basis(stack - terminal, k)
    return terminal + recon


def affine_dynamics_terminal(stack, k):
    terminal = stack[:, -1:, :]
    basis, coeff, _, _ = fit_shared_basis(stack - terminal, k)
    x = torch.cat([basis[:-1], torch.ones(basis.shape[0] - 1, 1, dtype=basis.dtype)], dim=1)
    y = basis[1:]
    w = torch.linalg.lstsq(x, y).solution
    a = w[:-1]
    b = w[-1]
    pred = [basis[0]]
    for _ in range(basis.shape[0] - 1):
        pred.append(pred[-1] @ a + b)
    pred = torch.stack(pred)
    return terminal + torch.einsum("lk,vkd->vld", pred, coeff)


def history_dynamics_terminal(stack, basis_k, order):
    terminal = stack[:, -1:, :]
    basis, coeff, _, _ = fit_shared_basis(stack - terminal, basis_k)
    x_rows = []
    y_rows = []
    for t in range(order, basis.shape[0]):
        x_rows.append(basis[t - order:t].reshape(-1))
        y_rows.append(basis[t])
    x = torch.stack(x_rows)
    y = torch.stack(y_rows)
    w = torch.linalg.lstsq(x, y).solution

    pred = [basis[t] for t in range(order)]
    for t in range(order, basis.shape[0]):
        window = torch.cat(pred[t - order:t]).unsqueeze(0)
        pred.append((window @ w).squeeze(0))
    pred = torch.stack(pred)
    return terminal + torch.einsum("lk,vkd->vld", pred, coeff)


names, stack = load_stack()

print("=" * 88)
print("  VOICEPACK HYPOTHESIS ELIMINATION")
print("=" * 88)

sep("DATA")
print(f"  voices:       {stack.shape[0]}")
print(f"  length slots: {stack.shape[1]}")
print(f"  style dim:    {stack.shape[2]}")

hypotheses = [
    ("linear first->last", single_curve_first_to_last(stack)),
    ("two-curve split", two_curve_split(stack)),
    ("mean-centered rank-4", mean_rank(stack, 4)),
    ("terminal-centered rank-4", terminal_rank(stack, 4)),
    ("terminal-centered rank-6", terminal_rank(stack, 6)),
    ("terminal-centered rank-11", terminal_rank(stack, 11)),
    ("affine dynamics rank-5", affine_dynamics_terminal(stack, 5)),
    ("affine dynamics rank-8", affine_dynamics_terminal(stack, 8)),
    ("history dynamics rank-11 order-12", history_dynamics_terminal(stack, 11, 12)),
]

sep("RESULTS")
print(f"  {'hypothesis':<34} {'MAE':>10} {'RMSE':>10} {'mean cos':>10} {'max abs':>10}")
print(f"  {'─'*34} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
rows = []
for name, recon in hypotheses:
    mae, rmse, cos, max_abs = metric(recon, stack)
    rows.append((name, mae, rmse, cos, max_abs))
    print(f"  {name:<34} {mae:>10.6f} {rmse:>10.6f} {cos:>10.6f} {max_abs:>10.6f}")

sep("ELIMINATION")
for name, mae, rmse, cos, max_abs in rows:
    if max_abs < 1e-4:
        verdict = "survives as exact artifact reproducer"
    elif rmse < 1e-3:
        verdict = "survives as near-exact compact reproducer"
    elif cos > 0.999:
        verdict = "matches geometry but not exact artifact"
    else:
        verdict = "eliminated"
    print(f"  {name:<34} {verdict}")

sep("CURRENT SURVIVORS")
print("  1. terminal-centered shared-rank factorization")
print("  2. shared affine dynamics on terminal-centered basis")
print("  Both are compiled-table families, not raw per-length sampling families.")

sep()
print("Done.")
