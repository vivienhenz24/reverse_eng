"""
Prototype Kokoro voicepack exporter simulator.

Simulator form:
  1. Build a low-rank proposal table over length.
  2. Anchor each voice to ref_s = pack[:, 0, :].
  3. Apply separate decoder/prosody alpha/beta-style mixing.
  4. Apply s_prev-like recurrence over length.

This does not prove the original exporter, but it tests whether a compact
StyleTTS2-inspired mechanism can numerically reproduce the shipped tables.
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


def learn_low_rank_proposal(stack, k=4):
    voice_mean = stack.mean(dim=1)
    residual = stack - voice_mean.unsqueeze(1)
    length_matrix = residual.permute(1, 0, 2).reshape(stack.shape[1], -1)
    U, S, Vh = torch.linalg.svd(length_matrix, full_matrices=False)
    basis = U[:, :k]
    coeff = torch.einsum("vld,lk->vkd", residual, basis)
    proposal = voice_mean.unsqueeze(1) + torch.einsum("vkd,lk->vld", coeff, basis)
    return proposal


def simulate(proposal, ref_s, alpha, beta, t_dec, t_pro):
    out = torch.empty_like(proposal)
    out[:, 0, :] = ref_s
    for i in range(1, proposal.shape[1]):
        mixed_dec = alpha * proposal[:, i, :128] + (1 - alpha) * ref_s[:, :128]
        mixed_pro = beta * proposal[:, i, 128:] + (1 - beta) * ref_s[:, 128:]
        out[:, i, :128] = t_dec * out[:, i - 1, :128] + (1 - t_dec) * mixed_dec
        out[:, i, 128:] = t_pro * out[:, i - 1, 128:] + (1 - t_pro) * mixed_pro
    return out


def search_params(target, proposal, ref_s):
    # The released tables are already extremely smooth and close to the low-rank
    # proposal, so concentrate the search in the high-reference / high-smoothing regime.
    alphas = [1.0]
    betas = [1.0]
    ts = [0.90, 0.93, 0.95, 0.97]

    best = None
    for alpha in alphas:
        for beta in betas:
            for t_dec in ts:
                for t_pro in ts:
                    recon = simulate(proposal, ref_s, alpha, beta, t_dec, t_pro)
                    mae, rmse, cos = metric(recon, target)
                    score = cos - mae - 0.25 * rmse
                    cand = {
                        "alpha": alpha,
                        "beta": beta,
                        "t_dec": t_dec,
                        "t_pro": t_pro,
                        "mae": mae,
                        "rmse": rmse,
                        "cos": cos,
                        "score": score,
                        "recon": recon,
                    }
                    if best is None or cand["score"] > best["score"]:
                        best = cand
    return best


names, stack = load_stack()
proposal = learn_low_rank_proposal(stack, k=4)
ref_s = stack[:, 0, :]

print("=" * 88)
print("  PROTOTYPE VOICEPACK EXPORTER")
print("=" * 88)

sep("BASELINES")
mae, rmse, cos = metric(proposal, stack)
print(f"  low-rank proposal only: MAE={mae:.6f}  RMSE={rmse:.6f}  mean cos={cos:.6f}")
const = ref_s.unsqueeze(1).repeat(1, stack.shape[1], 1)
mae, rmse, cos = metric(const, stack)
print(f"  ref_s repeated:        MAE={mae:.6f}  RMSE={rmse:.6f}  mean cos={cos:.6f}")

sep("GRID SEARCH")
best = search_params(stack, proposal, ref_s)
print(f"  best alpha: {best['alpha']}")
print(f"  best beta:  {best['beta']}")
print(f"  best t_dec: {best['t_dec']}")
print(f"  best t_pro: {best['t_pro']}")
print(f"  best MAE:   {best['mae']:.6f}")
print(f"  best RMSE:  {best['rmse']:.6f}")
print(f"  best cos:   {best['cos']:.6f}")

sep("ONE VOICE EXAMPLE")
idx = 0
voice = names[idx]
mae, rmse, cos = metric(best["recon"][idx:idx+1], stack[idx:idx+1])
print(f"  voice: {voice}")
print(f"  MAE={mae:.6f}  RMSE={rmse:.6f}  mean cos={cos:.6f}")
print(f"  first row preserved exactly: {bool(torch.allclose(best['recon'][idx, 0], stack[idx, 0]))}")

sep("INTERPRETATION")
print("  If the best fit uses alpha/beta near 1 with large t_dec/t_pro, then the")
print("  simulator behaves like a low-rank proposal compiled into a smooth table")
print("  while staying anchored to a reference style at short lengths.")
print("  That is the closest local code analogue to combining StyleTTS2 ref_s,")
print("  alpha/beta mixing, and s_prev smoothing into Kokoro's shipped voicepacks.")

sep()
print("Done.")
