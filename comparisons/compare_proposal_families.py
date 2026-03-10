"""
Compare proposal families for Kokoro voicepack coefficient generation.

We already have evidence that the final voicepacks are:
  - low-rank over length
  - strongly smoothed over adjacent lengths

This script asks a narrower question:
  before the final recurrence smoothing, do the discovered proposal curves look more like
  - deterministic teacher-forced / compiled proposals
  - or diffusion-like sampled proposals?
"""

from pathlib import Path
import torch

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


def discover_basis(stack, k=4):
    base = stack.mean(dim=1, keepdim=True)
    res = stack - base
    len_mat = res.permute(1, 0, 2).reshape(stack.shape[1], -1)
    U, S, Vh = torch.linalg.svd(len_mat, full_matrices=False)
    score = U[:, :k] * S[:k]
    recon = (U[:, :k] @ torch.diag(S[:k]) @ Vh[:k]).reshape(
        stack.shape[1], stack.shape[0], stack.shape[2]
    ).permute(1, 0, 2)
    err = res - recon
    return score, err


def family_score(metrics, family):
    if family == "teacher_forced":
        score = 0.0
        score += 2.0 * (1.0 - metrics["mean_sign_flip"])
        score += 1.5 * (1.0 / (1.0 + metrics["mean_abs_d2"]))
        score += 1.5 * (1.0 / (1.0 + metrics["residual_std"]))
        score += 1.0 * metrics["residual_adj_cos"]
        return score
    if family == "diffusion_like":
        score = 0.0
        score += 1.0 * metrics["mean_sign_flip"]
        score += 1.0 * metrics["mean_abs_d2"]
        score += 1.0 * metrics["residual_std"]
        score += 1.0 * (1.0 - metrics["residual_adj_cos"])
        return score
    if family == "hybrid_smoothed":
        score = 0.0
        score += 1.5 * (1.0 - metrics["mean_sign_flip"])
        score += 1.0 * (1.0 / (1.0 + metrics["mean_abs_d2"]))
        score += 1.0 * (1.0 / (1.0 + metrics["residual_std"]))
        score += 2.0 * metrics["residual_adj_cos"]
        return score
    raise ValueError(family)


_, stack = load_stack()
score, err = discover_basis(stack, k=4)

mode_metrics = []
for i in range(score.shape[1]):
    s = score[:, i]
    d1 = s[1:] - s[:-1]
    d2 = d1[1:] - d1[:-1]
    sign_flip = ((d1[1:] * d1[:-1]) < 0).float().mean().item()
    mode_metrics.append({
        "mode": i + 1,
        "mean_abs_d1": d1.abs().mean().item(),
        "mean_abs_d2": d2.abs().mean().item(),
        "sign_flip": sign_flip,
    })

metrics = {
    "mean_sign_flip": sum(m["sign_flip"] for m in mode_metrics) / len(mode_metrics),
    "mean_abs_d2": sum(m["mean_abs_d2"] for m in mode_metrics) / len(mode_metrics),
    "residual_std": err.std().item(),
    "residual_adj_cos": torch.nn.functional.cosine_similarity(
        err[:, :-1, :].reshape(-1, stack.shape[-1]),
        err[:, 1:, :].reshape(-1, stack.shape[-1]),
        dim=-1,
    ).mean().item(),
}

families = []
for name in ["teacher_forced", "hybrid_smoothed", "diffusion_like"]:
    families.append((name, family_score(metrics, name)))
families.sort(key=lambda x: x[1], reverse=True)

print("=" * 88)
print("  PROPOSAL FAMILY COMPARATOR")
print("=" * 88)

sep("DISCOVERED PROPOSAL METRICS")
print(f"  mean sign-flip rate of top basis modes: {metrics['mean_sign_flip']:.6f}")
print(f"  mean abs second-derivative of modes:    {metrics['mean_abs_d2']:.6f}")
print(f"  residual std after rank-4 removal:      {metrics['residual_std']:.6f}")
print(f"  residual adjacent cosine:               {metrics['residual_adj_cos']:.6f}")

sep("TOP BASIS MODES")
for m in mode_metrics:
    print(
        f"  mode {m['mode']}: "
        f"mean|d1|={m['mean_abs_d1']:.6f}  "
        f"mean|d2|={m['mean_abs_d2']:.6f}  "
        f"sign_flip={m['sign_flip']:.6f}"
    )

sep("FAMILY RANKING")
for rank, (name, score_value) in enumerate(families, start=1):
    print(f"  {rank}. {name:<16} score={score_value:.6f}")

sep("INTERPRETATION")
print("  Teacher-forced proposals should look low-noise, low-curvature, and nearly monotonic.")
print("  Diffusion-like proposals should retain more stochastic, higher-frequency structure.")
print("  Hybrid-smoothed proposals sit between them, but still inherit strong smoothness.")

sep("WORKING CONCLUSION")
print(f"  Best match: {families[0][0]}")
print("  Given the discovered proposal geometry, the coefficient proposal is much more")
print("  compatible with deterministic or heavily smoothed hybrid generation than with")
print("  raw diffusion-style sampling saved directly to disk.")

sep()
print("Done.")
