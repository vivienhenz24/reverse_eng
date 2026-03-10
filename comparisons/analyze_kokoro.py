"""
Deep analysis of Kokoro-82M weights and architecture.

Covers:
  - Full module tree with shapes and param counts
  - Per-submodule statistics (mean, std, min, max, sparsity)
  - PL-BERT (CustomAlbert) attention head / layer breakdown
  - TextEncoder CNN + LSTM breakdown
  - ProsodyPredictor (DurationEncoder, LSTM, F0/N heads)
  - iSTFTNet Decoder (Encoder, Decode stack, Generator)
  - Vocabulary analysis
"""

import sys, json, math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
KOKORO_WEIGHTS = ROOT / "kokoro" / "weights"
CONFIG_PATH    = KOKORO_WEIGHTS / "config.json"
MODEL_PATH     = KOKORO_WEIGHTS / "kokoro-v1_0.pth"

# ── helpers ────────────────────────────────────────────────────────────────────

def sep(title="", width=72, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char * 1} {title} {char * (pad - 1)}")
    else:
        print(char * width)

def tensor_stats(t: torch.Tensor) -> str:
    t = t.float()
    sparsity = (t == 0).float().mean().item()
    return (f"shape={list(t.shape)}  dtype={t.dtype}  "
            f"mean={t.mean():.4f}  std={t.std():.4f}  "
            f"min={t.min():.4f}  max={t.max():.4f}  "
            f"sparsity={sparsity:.2%}")

def count_params(state_dict, prefix=""):
    total = 0
    for k, v in state_dict.items():
        if k == prefix or k.startswith(prefix + "."):
            total += v.numel()
    return total

def module_param_table(state_dict, prefix, depth=2):
    """Print a table of tensors up to `depth` sub-levels under `prefix`."""
    seen = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix + ".") and k != prefix:
            continue
        rel = k[len(prefix):].lstrip(".")
        parts = rel.split(".")
        key = ".".join(parts[:depth]) if len(parts) >= depth else rel
        seen.setdefault(key, 0)
        seen[key] += v.numel()
    for k, n in sorted(seen.items()):
        print(f"    {prefix}.{k:<55} {n:>10,} params")

# ── load ───────────────────────────────────────────────────────────────────────

print("=" * 72)
print("  KOKORO-82M — DEEP WEIGHT & ARCHITECTURE ANALYSIS")
print("=" * 72)

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

raw = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

# raw is a dict of submodule -> state_dict
top_keys = list(raw.keys())
print(f"\nTop-level checkpoint keys: {top_keys}")

# flatten into a single state dict with prefixed keys
flat = {}
for mod_name, sd in raw.items():
    for k, v in sd.items():
        flat[f"{mod_name}.{k}"] = v

total_params = sum(v.numel() for v in flat.values())
print(f"Total parameters: {total_params:,}  (~{total_params/1e6:.1f}M)")

# ── CONFIG ─────────────────────────────────────────────────────────────────────

sep("CONFIG")
skip_keys = {"vocab"}
for k, v in cfg.items():
    if k in skip_keys:
        continue
    print(f"  {k}: {v}")

sep("VOCABULARY")
vocab = cfg["vocab"]
print(f"  vocab size (n_token): {cfg['n_token']}")
print(f"  assigned tokens:      {len(vocab)}")
print(f"  max token id:         {max(vocab.values())}")
# character classes
ipa_chars  = [c for c in vocab if ord(c) > 127]
ascii_chars = [c for c in vocab if ord(c) <= 127]
print(f"  ASCII symbols:        {len(ascii_chars)}  -> {ascii_chars}")
print(f"  IPA / Unicode chars:  {len(ipa_chars)}")

# ── PL-BERT (bert) ─────────────────────────────────────────────────────────────

sep("PL-BERT  (CustomAlbert backbone)")
pb = cfg["plbert"]
for k, v in pb.items():
    print(f"  {k}: {v}")

bert_params = count_params(flat, "bert")
print(f"\n  Total bert params: {bert_params:,}")
print(f"  Approx attention params per layer (if ALBERT blocks were unshared):")
hs = pb["hidden_size"]
heads = pb["num_attention_heads"]
inter = pb["intermediate_size"]
attn_per_layer   = 4 * hs * hs          # Q, K, V, O projections
ffn_per_layer    = 2 * hs * inter        # two linear layers
total_per_layer  = attn_per_layer + ffn_per_layer
print(f"    attention (Q/K/V/O): {attn_per_layer:,}")
print(f"    FFN (up+down):        {ffn_per_layer:,}")
print(f"    total per layer:      {total_per_layer:,}")
print(f"    × {pb['num_hidden_layers']} layers:           {total_per_layer * pb['num_hidden_layers']:,}")

sep("PL-BERT layer-by-layer param counts", char="·")
module_param_table(flat, "bert", depth=3)

# ── bert_encoder ───────────────────────────────────────────────────────────────

sep("BERT_ENCODER  (Linear: hidden_size → hidden_dim)")
be_w = flat.get("bert_encoder.weight")
be_b = flat.get("bert_encoder.bias")
if be_w is not None:
    print(f"  weight: {tensor_stats(be_w)}")
if be_b is not None:
    print(f"  bias:   {tensor_stats(be_b)}")
print(f"  Projects {hs} → {cfg['hidden_dim']}")

# ── TextEncoder ────────────────────────────────────────────────────────────────

sep("TEXT_ENCODER")
hd = cfg["hidden_dim"]
n_tok = cfg["n_token"]
n_layer = cfg["n_layer"]
k_size = cfg["text_encoder_kernel_size"]
print(f"  Embedding:  n_token={n_tok} → channels={hd}")
print(f"  CNN blocks: {n_layer} × Conv1d({hd},{hd}, k={k_size}) + LayerNorm + LeakyReLU + Dropout(0.2)")
print(f"  LSTM:       input={hd}, hidden={hd//2}, bidirectional → output={hd}")
te_params = count_params(flat, "text_encoder")
print(f"  Total params: {te_params:,}")
print()
module_param_table(flat, "text_encoder", depth=2)

# ── ProsodyPredictor ────────────────────────────────────────────────────────────

sep("PROSODY_PREDICTOR")
sty = cfg["style_dim"]
max_dur = cfg["max_dur"]
print(f"  style_dim={sty}  hidden_dim={hd}  n_layer={n_layer}  max_dur={max_dur}")
print()
print(f"  DurationEncoder (text_encoder):")
print(f"    {n_layer} × [ LSTM({hd+sty} → {hd//2}*2 bidir) + AdaLayerNorm({sty},{hd}) ]")
print(f"  Duration LSTM:  ({hd+sty} → {hd//2}*2 bidir)")
print(f"  Duration proj:  LinearNorm({hd}, {max_dur})")
print(f"  Shared LSTM:    ({hd+sty} → {hd//2}*2 bidir)")
print(f"  F0 head:  3 × AdainResBlk1d  [{hd}→{hd}, {hd}→{hd//2}(up), {hd//2}→{hd//2}] + Conv1d({hd//2},1)")
print(f"  N  head:  3 × AdainResBlk1d  [{hd}→{hd}, {hd}→{hd//2}(up), {hd//2}→{hd//2}] + Conv1d({hd//2},1)")
pp_params = count_params(flat, "predictor")
print(f"\n  Total params: {pp_params:,}")
print()
module_param_table(flat, "predictor", depth=3)

# ── iSTFTNet Decoder ────────────────────────────────────────────────────────────

sep("DECODER  (iSTFTNet)")
ist = cfg["istftnet"]
print(f"  upsample_rates:          {ist['upsample_rates']}")
print(f"  upsample_kernel_sizes:   {ist['upsample_kernel_sizes']}")
print(f"  upsample_initial_channel:{ist['upsample_initial_channel']}")
print(f"  resblock_kernel_sizes:   {ist['resblock_kernel_sizes']}")
print(f"  resblock_dilation_sizes: {ist['resblock_dilation_sizes']}")
print(f"  gen_istft_n_fft:         {ist['gen_istft_n_fft']}")
print(f"  gen_istft_hop_size:      {ist['gen_istft_hop_size']}")
total_upsample = math.prod(ist["upsample_rates"]) * ist["gen_istft_hop_size"]
print(f"  total upsampling factor: {total_upsample}× (@ 24kHz → 300 samples/frame, 80.0 frames/sec)")
print()
print(f"  Encoder: AdainResBlk1d({cfg['dim_in']+2} → 1024, style_dim={sty})")
print(f"  Decode stack: 3× AdainResBlk1d(1024+2+64 → 1024) + 1× →512 (upsample)")
print(f"  F0_conv: Conv1d(1,1,k=3,stride=2)")
print(f"  N_conv:  Conv1d(1,1,k=3,stride=2)")
print(f"  asr_res: Conv1d(512→64,k=1)")
print()
# Generator channel schedule
ch = ist["upsample_initial_channel"]
print(f"  Generator (iSTFT-based HiFiGAN):")
for i, (u, k) in enumerate(zip(ist["upsample_rates"], ist["upsample_kernel_sizes"])):
    ch_out = ch // (2 ** (i + 1))
    print(f"    ups[{i}]: ConvTranspose1d({ch//(2**i)} → {ch_out}, k={k}, stride={u})")
    for j, (rk, rd) in enumerate(zip(ist["resblock_kernel_sizes"], ist["resblock_dilation_sizes"])):
        print(f"      resblock[{i*3+j}]: AdaINResBlock1({ch_out}, k={rk}, dil={rd})")
print(f"    post conv: Conv1d({ch//(2**len(ist['upsample_rates']))} → {ist['gen_istft_n_fft']+2}, k=7)")
print(f"    iSTFT: n_fft={ist['gen_istft_n_fft']}, hop={ist['gen_istft_hop_size']}")
print(f"    SineGen harmonics: 8  (NSF source module)")

dec_params = count_params(flat, "decoder")
print(f"\n  Total decoder params: {dec_params:,}")
print()
module_param_table(flat, "decoder", depth=3)

# ── Per-submodule summary ───────────────────────────────────────────────────────

sep("PARAMETER BUDGET SUMMARY")
rows = []
for mod in ["bert", "bert_encoder", "text_encoder", "predictor", "decoder"]:
    n = count_params(flat, mod)
    rows.append((mod, n))

total_check = sum(r[1] for r in rows)
print(f"  {'Module':<20} {'Params':>12}   {'Share':>7}")
print(f"  {'─'*20} {'─'*12}   {'─'*7}")
for mod, n in rows:
    print(f"  {mod:<20} {n:>12,}   {n/total_check:>6.1%}")
print(f"  {'─'*20} {'─'*12}")
print(f"  {'TOTAL':<20} {total_check:>12,}")

# ── Weight statistics per submodule ────────────────────────────────────────────

sep("WEIGHT STATISTICS PER SUBMODULE")
for mod in ["bert", "bert_encoder", "text_encoder", "predictor", "decoder"]:
    tensors = [v.float() for k, v in flat.items() if k.startswith(mod + ".")]
    if not tensors:
        continue
    cat = torch.cat([t.reshape(-1) for t in tensors])
    sparsity = (cat == 0).float().mean().item()
    print(f"\n  [{mod}]")
    print(f"    count:    {len(tensors)} tensors")
    print(f"    mean:     {cat.mean():.6f}")
    print(f"    std:      {cat.std():.6f}")
    print(f"    min:      {cat.min():.6f}")
    print(f"    max:      {cat.max():.6f}")
    print(f"    sparsity: {sparsity:.4%}")
    # weight norm distribution
    norms = [t.norm().item() for t in tensors]
    print(f"    tensor L2 norms — min={min(norms):.4f}  max={max(norms):.4f}  mean={sum(norms)/len(norms):.4f}")

sep()
print("Done.")
