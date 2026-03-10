"""
Score concrete Kokoro voicepack export variants against observed pack statistics.

The goal is not to recover the exact exporter, but to identify which families of
offline export behavior best match the shipped voicepack tables.
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
    return names, torch.stack([voices[n] for n in names])  # (V, 510, 256)


def summarize(stack):
    adj = F.cosine_similarity(stack[:, :-1, :], stack[:, 1:, :], dim=-1)
    first_last = F.cosine_similarity(stack[:, 0, :], stack[:, -1, :], dim=-1)
    per_len_voice_var = stack.var(dim=0).mean(dim=1)
    delta = (stack[:, 1:, :] - stack[:, :-1, :]).norm(dim=-1).mean(dim=0)
    dec = stack[:, :, :128]
    pro = stack[:, :, 128:]
    return {
        "adj_mean": adj.mean().item(),
        "adj_min": adj.min().item(),
        "first_last_mean": first_last.mean().item(),
        "first_len_var": per_len_voice_var[0].item(),
        "len_10_var": per_len_voice_var[9].item(),
        "len_100_var": per_len_voice_var[99].item(),
        "last_len_var": per_len_voice_var[-1].item(),
        "first_delta": delta[0].item(),
        "delta_10": delta[9].item(),
        "delta_100": delta[99].item(),
        "decoder_len_var": dec.var(dim=1).mean().item(),
        "prosody_len_var": pro.var(dim=1).mean().item(),
    }


def make_anchor_pair(real_stack):
    base = real_stack.mean(dim=1)  # (V, 256)
    delta = real_stack[:, -1, :] - real_stack[:, 0, :]
    delta = F.normalize(delta, dim=-1)
    magnitude = real_stack.std(dim=1).mean(dim=1, keepdim=True)
    start = base + delta * magnitude * 4.0
    end = base - delta * magnitude * 4.0
    return start, end


def constant_variant(real_stack):
    base = real_stack.mean(dim=1, keepdim=True)
    return base.repeat(1, real_stack.shape[1], 1)


def linear_variant(real_stack):
    start, end = make_anchor_pair(real_stack)
    t = torch.linspace(0.0, 1.0, real_stack.shape[1], device=real_stack.device).view(1, -1, 1)
    return start.unsqueeze(1) * (1 - t) + end.unsqueeze(1) * t


def exp_decay_variant(real_stack, strength=6.0):
    start, end = make_anchor_pair(real_stack)
    t = torch.linspace(0.0, 1.0, real_stack.shape[1], device=real_stack.device).view(1, -1, 1)
    w = (1 - torch.exp(-strength * t)) / (1 - math.exp(-strength))
    return start.unsqueeze(1) * (1 - w) + end.unsqueeze(1) * w


def short_heavy_variant(real_stack, strength=4.0, prosody_boost=3.0):
    start, end = make_anchor_pair(real_stack)
    t = torch.linspace(0.0, 1.0, real_stack.shape[1], device=real_stack.device).view(1, -1, 1)
    w = torch.sqrt(t)
    out = start.unsqueeze(1) * (1 - w) + end.unsqueeze(1) * w
    # Make the prosody half more length-sensitive than the decoder half.
    pro_start = start[:, 128:].unsqueeze(1)
    pro_end = end[:, 128:].unsqueeze(1)
    w_pro = (1 - torch.exp(-strength * t)) / (1 - math.exp(-strength))
    out[:, :, 128:] = pro_start * (1 - w_pro) + pro_end * w_pro
    out[:, :, 128:] += (1 - t) * real_stack[:, :1, 128:] * (prosody_boost - 1) * 0.15
    return out


def two_stage_variant(real_stack):
    start, end = make_anchor_pair(real_stack)
    n = real_stack.shape[1]
    t = torch.linspace(0.0, 1.0, n, device=real_stack.device)
    w = torch.empty_like(t)
    split = 24
    w[:split] = torch.linspace(0.0, 0.7, split, device=real_stack.device)
    w[split:] = torch.linspace(0.7, 1.0, n - split, device=real_stack.device)
    w = w.view(1, -1, 1)
    out = start.unsqueeze(1) * (1 - w) + end.unsqueeze(1) * w
    # Prosody half moves faster early than decoder half.
    w_dec = w
    w_pro = torch.clamp(w * 1.35, max=1.0)
    out[:, :, :128] = start[:, :128].unsqueeze(1) * (1 - w_dec) + end[:, :128].unsqueeze(1) * w_dec
    out[:, :, 128:] = start[:, 128:].unsqueeze(1) * (1 - w_pro) + end[:, 128:].unsqueeze(1) * w_pro
    return out


def teacher_forced_like_variant(real_stack):
    # Use each voice's actual mean as identity and impose a steep early-length schedule.
    base = real_stack.mean(dim=1, keepdim=True)
    early = real_stack[:, :1, :] - base
    late = real_stack[:, -1:, :] - base
    n = real_stack.shape[1]
    t = torch.linspace(0.0, 1.0, n, device=real_stack.device).view(1, -1, 1)
    w = torch.log1p(40 * t) / math.log1p(40)
    out = base + early * (1 - w) + late * w
    out[:, :, 128:] = base[:, :, 128:] + (real_stack[:, :1, 128:] - base[:, :, 128:]) * (1 - torch.sqrt(t)) + (real_stack[:, -1:, 128:] - base[:, :, 128:]) * torch.sqrt(t)
    return out


def score_variant(name, synth_stats, real_stats):
    keys = [
        "adj_mean",
        "first_last_mean",
        "first_len_var",
        "len_10_var",
        "len_100_var",
        "first_delta",
        "delta_10",
        "delta_100",
        "decoder_len_var",
        "prosody_len_var",
    ]
    errors = {}
    total = 0.0
    for k in keys:
        denom = max(abs(real_stats[k]), 1e-6)
        err = abs(synth_stats[k] - real_stats[k]) / denom
        errors[k] = err
        total += err
    # Lower is better. Convert to a convenience score.
    fit = 1.0 / (1.0 + total)
    return {
        "name": name,
        "fit": fit,
        "total_error": total,
        "errors": errors,
        "stats": synth_stats,
    }


_, real_stack = load_stack()
real_stats = summarize(real_stack)

variants = {
    "constant_repeat": constant_variant(real_stack),
    "linear_interp": linear_variant(real_stack),
    "exp_decay_interp": exp_decay_variant(real_stack),
    "short_heavy_interp": short_heavy_variant(real_stack),
    "two_stage_interp": two_stage_variant(real_stack),
    "teacher_forced_like": teacher_forced_like_variant(real_stack),
}

results = []
for name, synth in variants.items():
    results.append(score_variant(name, summarize(synth), real_stats))
results.sort(key=lambda x: x["fit"], reverse=True)

print("=" * 88)
print("  EXPORT VARIANT SCORER")
print("=" * 88)

sep("REAL VOICEPACK TARGET STATS")
for k, v in real_stats.items():
    print(f"  {k:<18} {v:.6f}")

sep("RANKING")
print(f"  {'Variant':<22} {'Fit':>8} {'Total rel err':>16}")
print(f"  {'─'*22} {'─'*8} {'─'*16}")
for r in results:
    print(f"  {r['name']:<22} {r['fit']:>8.4f} {r['total_error']:>16.4f}")

sep("TOP VARIANT DETAILS")
best = results[0]
print(f"  best variant: {best['name']}")
print("  synthetic stats:")
for k, v in best["stats"].items():
    print(f"    {k:<18} {v:.6f}")
print("  relative errors:")
for k, v in best["errors"].items():
    print(f"    {k:<18} {v:.4f}")

sep("INTERPRETATION")
print("  These are shape-family fits, not training reconstructions.")
print("  If short-length-heavy or two-stage schedules win, that supports an exporter")
print("  that changes style most at short phoneme lengths and then saturates.")
print("  If teacher_forced_like wins, that supports a deterministic export process")
print("  built from utterance-level StyleTTS2 style targets rather than raw diffusion.")

sep()
print("Done.")
