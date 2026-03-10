"""
Deep analysis of StyleTTS2 pretrained weights.

Covers both checkpoints:
  - LJSpeech  (single-speaker, epoch_2nd_00100.pth, iSTFTNet decoder)
  - LibriTTS  (multi-speaker, epochs_2nd_00020.pth, HiFiGAN decoder)

For each model:
  - All checkpoint keys with shapes and param counts
  - Per-submodule parameter budget
  - Weight statistics (mean, std, min, max, sparsity)
  - Config diff between LJSpeech and LibriTTS
  - Key architectural differences
"""

import sys, math
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
LJS_CKPT   = ROOT / "StyleTTS2/Models/LJSpeech/Models/LJSpeech/epoch_2nd_00100.pth"
LJS_CFG    = ROOT / "StyleTTS2/Models/LJSpeech/Models/LJSpeech/config.yml"
LIBRITTS_CKPT = ROOT / "StyleTTS2/Models/LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth"
LIBRITTS_CFG  = ROOT / "StyleTTS2/Models/LibriTTS/Models/LibriTTS/config.yml"

def sep(title="", width=72, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)

def tensor_stats(t: torch.Tensor) -> str:
    t = t.float()
    sparsity = (t == 0).float().mean().item()
    return (f"mean={t.mean():.5f}  std={t.std():.5f}  "
            f"min={t.min():.5f}  max={t.max():.5f}  sparsity={sparsity:.3%}")

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)

def analyze_checkpoint(name, ckpt_path, cfg_path):
    sep(f"{name} — CHECKPOINT ANALYSIS")

    cfg = load_cfg(cfg_path)
    mp  = cfg["model_params"]
    pp  = cfg["preprocess_params"]

    print(f"\n  Config snapshot:")
    print(f"    sample_rate:       {pp['sr']}")
    print(f"    hop_length:        {pp['spect_params']['hop_length']}")
    print(f"    n_fft:             {pp['spect_params']['n_fft']}")
    print(f"    win_length:        {pp['spect_params']['win_length']}")
    print(f"    hidden_dim:        {mp['hidden_dim']}")
    print(f"    style_dim:         {mp['style_dim']}")
    print(f"    n_layer:           {mp['n_layer']}")
    print(f"    n_token:           {mp['n_token']}")
    print(f"    n_mels:            {mp['n_mels']}")
    print(f"    multispeaker:      {mp['multispeaker']}")
    print(f"    decoder type:      {mp['decoder']['type']}")
    dec = mp["decoder"]
    print(f"    upsample_rates:    {dec['upsample_rates']}")
    print(f"    upsample_init_ch:  {dec['upsample_initial_channel']}")
    if "gen_istft_n_fft" in dec:
        print(f"    gen_istft_n_fft:   {dec['gen_istft_n_fft']}")
        print(f"    gen_istft_hop:     {dec['gen_istft_hop_size']}")
    total_up = math.prod(dec["upsample_rates"])
    if "gen_istft_hop_size" in dec:
        total_up *= dec["gen_istft_hop_size"]
    print(f"    total upsample:    {total_up}×")
    diff = cfg.get("model_params", {}).get("diffusion", {})
    print(f"    diffusion layers:  {diff.get('transformer', {}).get('num_layers', 'N/A')}")
    print(f"    diffusion heads:   {diff.get('transformer', {}).get('num_heads', 'N/A')}")
    print(f"    SLM model:         {mp['slm']['model']}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"\n  Checkpoint type: {type(ckpt)}")

    if isinstance(ckpt, dict):
        top_keys = list(ckpt.keys())
        print(f"  Top-level keys: {top_keys[:20]}")
    else:
        print(f"  WARNING: unexpected checkpoint format")
        return

    # collect all tensors
    def flatten(obj, prefix=""):
        out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(obj, torch.Tensor):
            out[prefix] = obj
        return out

    flat = flatten(ckpt)
    total_params = sum(v.numel() for v in flat.values())
    print(f"  Total tensors: {len(flat)}")
    print(f"  Total params:  {total_params:,}  (~{total_params/1e6:.1f}M)")

    # ── Key inventory ──────────────────────────────────────────────────────────

    sep(f"{name} — ALL KEYS (shape, numel)", char="·")
    print(f"  {'Key':<60} {'Shape':>20}  {'Params':>10}")
    print(f"  {'─'*60} {'─'*20}  {'─'*10}")
    for k, v in sorted(flat.items()):
        shp = str(list(v.shape))
        print(f"  {k:<60} {shp:>20}  {v.numel():>10,}")

    # ── Submodule breakdown ────────────────────────────────────────────────────

    sep(f"{name} — PARAMETER BUDGET BY SUBMODULE", char="·")
    sub_totals = {}
    for k, v in flat.items():
        top = k.split(".")[0]
        sub_totals[top] = sub_totals.get(top, 0) + v.numel()

    print(f"  {'Submodule':<30} {'Params':>12}   {'Share':>7}")
    print(f"  {'─'*30} {'─'*12}   {'─'*7}")
    for sub, n in sorted(sub_totals.items(), key=lambda x: -x[1]):
        print(f"  {sub:<30} {n:>12,}   {n/total_params:>6.1%}")
    print(f"  {'─'*30} {'─'*12}")
    print(f"  {'TOTAL':<30} {total_params:>12,}")

    # ── Weight statistics per submodule ───────────────────────────────────────

    sep(f"{name} — WEIGHT STATISTICS PER SUBMODULE", char="·")
    for sub in sorted(sub_totals):
        tensors = [v.float() for k, v in flat.items() if k.startswith(sub + ".") or k == sub]
        if not tensors:
            continue
        cat = torch.cat([t.reshape(-1) for t in tensors])
        sparsity = (cat == 0).float().mean().item()
        norms = [t.norm().item() for t in tensors]
        print(f"\n  [{sub}]  {len(tensors)} tensors  {cat.numel():,} values")
        print(f"    mean={cat.mean():.6f}  std={cat.std():.6f}  "
              f"min={cat.min():.6f}  max={cat.max():.6f}  sparsity={sparsity:.3%}")
        print(f"    tensor norms — min={min(norms):.4f}  max={max(norms):.4f}  "
              f"mean={sum(norms)/len(norms):.4f}")

    return flat, total_params, sub_totals

# ── Run both ────────────────────────────────────────────────────────────────────

print("=" * 72)
print("  STYLETTS2 — DEEP WEIGHT & ARCHITECTURE ANALYSIS")
print("=" * 72)

ljs_flat, ljs_total, ljs_subs  = analyze_checkpoint("LJSpeech",  LJS_CKPT,      LJS_CFG)
lib_flat, lib_total, lib_subs  = analyze_checkpoint("LibriTTS",  LIBRITTS_CKPT, LIBRITTS_CFG)

# ── LJSpeech vs LibriTTS comparison ───────────────────────────────────────────

sep("LJSpeech vs LibriTTS — CONFIG DIFF")
ljs_cfg = load_cfg(LJS_CFG)
lib_cfg = load_cfg(LIBRITTS_CFG)

def flat_dict(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat_dict(v, f"{prefix}.{k}" if prefix else k))
    else:
        out[prefix] = d
    return out

ljs_flat_cfg = flat_dict(ljs_cfg)
lib_flat_cfg = flat_dict(lib_cfg)
all_keys = set(ljs_flat_cfg) | set(lib_flat_cfg)
diffs = []
for k in sorted(all_keys):
    lv = ljs_flat_cfg.get(k, "<missing>")
    rv = lib_flat_cfg.get(k, "<missing>")
    if lv != rv:
        diffs.append((k, lv, rv))

print(f"\n  {'Config key':<55} {'LJSpeech':>15}  {'LibriTTS':>15}")
print(f"  {'─'*55} {'─'*15}  {'─'*15}")
for k, lv, rv in diffs:
    lv_s = str(lv)[:15]
    rv_s = str(rv)[:15]
    print(f"  {k:<55} {lv_s:>15}  {rv_s:>15}")

sep("LJSpeech vs LibriTTS — PARAM COUNT COMPARISON")
all_subs = sorted(set(ljs_subs) | set(lib_subs))
print(f"  {'Submodule':<30} {'LJSpeech':>12}  {'LibriTTS':>12}  {'Δ':>12}")
print(f"  {'─'*30} {'─'*12}  {'─'*12}  {'─'*12}")
for sub in all_subs:
    lv = ljs_subs.get(sub, 0)
    rv = lib_subs.get(sub, 0)
    delta = rv - lv
    sign = "+" if delta > 0 else ""
    print(f"  {sub:<30} {lv:>12,}  {rv:>12,}  {sign}{delta:>11,}")
print(f"  {'─'*30} {'─'*12}  {'─'*12}  {'─'*12}")
lv_t = ljs_total
rv_t = lib_total
delta_t = rv_t - lv_t
sign = "+" if delta_t > 0 else ""
print(f"  {'TOTAL':<30} {lv_t:>12,}  {rv_t:>12,}  {sign}{delta_t:>11,}")

sep("KEY ARCHITECTURAL DIFFERENCES")
print("""
  LJSpeech (single-speaker):
    • multispeaker=False — no speaker embedding, style from reference audio only
    • Decoder: iSTFTNet (upsample_rates=[10,6], gen_istft_n_fft=20, hop=5)
    • Trained 200 epochs (1st) + 100 epochs (2nd)
    • max_len=400 frames (~5 sec audio at 24kHz/300hop)
    • Diffusion sigma_data≈0.457 (higher variance style prior)
    • Checkpoint is epoch 100 of 2nd stage

  LibriTTS (multi-speaker):
    • multispeaker=True — speaker-conditioned style diffusion
    • Decoder: HiFiGAN (upsample_rates=[10,5,3,2], 4 stages vs 2)
    • Total upsample: 10×5×3×2 = 300 (same 300 hop at 24kHz)
    • Trained 40 epochs (1st) + 25 epochs (2nd)
    • max_len=300 frames (~3.75 sec audio)
    • Diffusion sigma_data≈0.199 (tighter style distribution — many speakers)
    • Checkpoint is epoch 20 of 2nd stage
    • Denoiser architecture uses speaker embedding in diffusion transformer
""")

sep()
print("Done.")
