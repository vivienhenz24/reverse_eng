"""
Cross-model comparison: Kokoro-82M vs StyleTTS2 LJSpeech vs StyleTTS2 LibriTTS.

Covers:
  - Architecture lineage and shared DNA
  - Parameter budget side-by-side
  - Shared submodules: TextEncoder, ProsodyPredictor, style_dim
  - Decoder strategy: iSTFTNet vs HiFiGAN
  - Style conditioning mechanism differences
  - Voice conditioning: Kokoro voicepacks vs StyleTTS2 reference audio
  - Training objective differences
  - Direct weight-space comparison for shared structures
"""

import json, math, sys
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent

KOKORO_CFG    = ROOT / "kokoro/weights/config.json"
KOKORO_MODEL  = ROOT / "kokoro/weights/kokoro-v1_0.pth"
VOICES_DIR    = ROOT / "kokoro/weights/voices"

LJS_CKPT      = ROOT / "StyleTTS2/Models/LJSpeech/Models/LJSpeech/epoch_2nd_00100.pth"
LJS_CFG       = ROOT / "StyleTTS2/Models/LJSpeech/Models/LJSpeech/config.yml"
LIB_CKPT      = ROOT / "StyleTTS2/Models/LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth"
LIB_CFG       = ROOT / "StyleTTS2/Models/LibriTTS/Models/LibriTTS/config.yml"

def sep(title="", width=72, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)

def flatten_ckpt(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_ckpt(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, torch.Tensor):
        out[prefix] = obj
    return out

# ── Load ────────────────────────────────────────────────────────────────────────

print("=" * 72)
print("  CROSS-MODEL COMPARISON: Kokoro-82M vs StyleTTS2")
print("=" * 72)

with open(KOKORO_CFG) as f:
    kok_cfg = json.load(f)
kok_raw = torch.load(KOKORO_MODEL, map_location="cpu", weights_only=True)
kok_flat = {}
for mod, sd in kok_raw.items():
    for k, v in sd.items():
        kok_flat[f"{mod}.{k}"] = v

with open(LJS_CFG) as f:
    ljs_cfg = yaml.safe_load(f)
ljs_flat = flatten_ckpt(torch.load(LJS_CKPT, map_location="cpu", weights_only=False))

with open(LIB_CFG) as f:
    lib_cfg = yaml.safe_load(f)
lib_flat = flatten_ckpt(torch.load(LIB_CKPT, map_location="cpu", weights_only=False))

kok_params = sum(v.numel() for v in kok_flat.values())
ljs_params = sum(v.numel() for v in ljs_flat.values())
lib_params = sum(v.numel() for v in lib_flat.values())

# ── Architecture overview ───────────────────────────────────────────────────────

sep("ARCHITECTURE OVERVIEW")
print("""
  All three models share the StyleTTS2 lineage:
    Text  →  PL-BERT (CustomAlbert)  →  TextEncoder (CNN+LSTM)
          →  ProsodyPredictor (DurationEncoder, F0/N heads)
          →  Alignment  →  Decoder (iSTFTNet or HiFiGAN)

  Style conditioning uses AdaIN (Adaptive Instance Normalization)
  throughout the decoder, controlled by a 128-dim style vector.

  ┌──────────────────────────────────────────────────────────────┐
  │  Component           │ Kokoro-82M │ StyleTTS2-LJS │ StyleTTS2-Lib│
  ├──────────────────────────────────────────────────────────────┤
  │  PL-BERT backbone    │  yes (12L) │  yes (12L)    │  yes (12L)   │
  │  TextEncoder         │  yes       │  yes          │  yes         │
  │  ProsodyPredictor    │  yes       │  yes          │  yes         │
  │  Style diffusion     │  NO        │  yes          │  yes         │
  │  WavLM discriminator │  NO        │  yes (train)  │  yes (train) │
  │  Speaker embedding   │  NO        │  NO           │  YES         │
  │  Decoder             │  iSTFTNet  │  iSTFTNet     │  HiFiGAN     │
  │  Voice conditioning  │  .pt vpack │  ref audio    │  ref audio   │
  │  Multispeaker        │  yes       │  NO           │  YES         │
  └──────────────────────────────────────────────────────────────┘
""")

# ── Hyperparameter table ────────────────────────────────────────────────────────

sep("HYPERPARAMETER COMPARISON")
ljs_mp = ljs_cfg["model_params"]
lib_mp = lib_cfg["model_params"]
ljs_pp = ljs_cfg["preprocess_params"]
lib_pp = lib_cfg["preprocess_params"]

rows = [
    ("hidden_dim",       kok_cfg["hidden_dim"],         ljs_mp["hidden_dim"],        lib_mp["hidden_dim"]),
    ("style_dim",        kok_cfg["style_dim"],           ljs_mp["style_dim"],         lib_mp["style_dim"]),
    ("n_layer",          kok_cfg["n_layer"],             ljs_mp["n_layer"],           lib_mp["n_layer"]),
    ("n_token",          kok_cfg["n_token"],             ljs_mp["n_token"],           lib_mp["n_token"]),
    ("n_mels",           kok_cfg["n_mels"],              ljs_mp["n_mels"],            lib_mp["n_mels"]),
    ("dropout",          kok_cfg["dropout"],             ljs_mp["dropout"],           lib_mp["dropout"]),
    ("max_dur",          kok_cfg["max_dur"],             ljs_mp["max_dur"],           lib_mp["max_dur"]),
    ("multispeaker",     kok_cfg["multispeaker"],        ljs_mp["multispeaker"],      lib_mp["multispeaker"]),
    ("sample_rate",      24000,                          ljs_pp["sr"],                lib_pp["sr"]),
    ("hop_length",       "300 (via upsample)",           ljs_pp["spect_params"]["hop_length"], lib_pp["spect_params"]["hop_length"]),
    ("decoder type",     "iSTFTNet",                     ljs_mp["decoder"]["type"],   lib_mp["decoder"]["type"]),
    ("upsample_rates",   str(kok_cfg["istftnet"]["upsample_rates"]), str(ljs_mp["decoder"]["upsample_rates"]), str(lib_mp["decoder"]["upsample_rates"])),
    ("PLBERT layers",    kok_cfg["plbert"]["num_hidden_layers"],  "12 (PLBERT)",   "12 (PLBERT)"),
    ("PLBERT hidden",    kok_cfg["plbert"]["hidden_size"],        "768",            "768"),
    ("PLBERT attn heads",kok_cfg["plbert"]["num_attention_heads"],"12",            "12"),
    ("style diffusion",  "NO",                           "YES (3 layers, 8 heads)",   "YES (3 layers, 8 heads)"),
]

print(f"\n  {'Parameter':<25} {'Kokoro-82M':>14}  {'StyleTTS2-LJS':>15}  {'StyleTTS2-Lib':>15}")
print(f"  {'─'*25} {'─'*14}  {'─'*15}  {'─'*15}")
for name, kok, ljs, lib in rows:
    same = "✓" if str(kok) == str(ljs) == str(lib) else " "
    print(f"  {name:<25} {str(kok):>14}  {str(ljs):>15}  {str(lib):>15}  {same}")

# ── Parameter budget ────────────────────────────────────────────────────────────

sep("TOTAL PARAMETER BUDGET")
print(f"  Kokoro-82M:       {kok_params:>12,}  (~{kok_params/1e6:.1f}M)")
print(f"  StyleTTS2-LJS:    {ljs_params:>12,}  (~{ljs_params/1e6:.1f}M)")
print(f"  StyleTTS2-LibTTS: {lib_params:>12,}  (~{lib_params/1e6:.1f}M)")

# Estimate Kokoro submodule params (mirroring what analyze_kokoro.py computed)
def kok_sub(prefix):
    return sum(v.numel() for k, v in kok_flat.items() if k.startswith(prefix + "."))

kok_subs = {
    "bert":         kok_sub("bert"),
    "bert_encoder": kok_sub("bert_encoder"),
    "text_encoder": kok_sub("text_encoder"),
    "predictor":    kok_sub("predictor"),
    "decoder":      kok_sub("decoder"),
}

def ljs_sub(prefix):
    return sum(v.numel() for k, v in ljs_flat.items() if k.split(".")[0] == prefix)

def lib_sub(prefix):
    return sum(v.numel() for k, v in lib_flat.items() if k.split(".")[0] == prefix)

ljs_subs_dict = {}
lib_subs_dict = {}
for k in ljs_flat:
    top = k.split(".")[0]
    ljs_subs_dict[top] = ljs_subs_dict.get(top, 0) + ljs_flat[k].numel()
for k in lib_flat:
    top = k.split(".")[0]
    lib_subs_dict[top] = lib_subs_dict.get(top, 0) + lib_flat[k].numel()

sep("SUBMODULE PARAMS: Kokoro vs StyleTTS2")
# StyleTTS2 checkpoints store everything under "net.*" prefix
def count_by_prefix(flat, prefix):
    return sum(v.numel() for k, v in flat.items() if k.startswith(prefix + ".") or k == prefix)

# (label, kokoro_prefix, ljs_prefix, lib_prefix)
mappings = [
    ("bert (PLBERT)",     "bert",         "net.bert",          "net.bert"),
    ("bert_encoder",      "bert_encoder", "net.bert_encoder",  "net.bert_encoder"),
    ("text_encoder",      "text_encoder", "net.text_encoder",  "net.text_encoder"),
    ("predictor",         "predictor",    "net.predictor",     "net.predictor"),
    ("decoder",           "decoder",      "net.decoder",       "net.decoder"),
    ("style_encoder",     None,           "net.style_encoder", "net.style_encoder"),
    ("text_aligner",      None,           "net.text_aligner",  "net.text_aligner"),
    ("diffusion",         None,           "net.diffusion",     "net.diffusion"),
    ("mpd discriminator", None,           "net.mpd",           "net.mpd"),
    ("msd discriminator", None,           "net.msd",           "net.msd"),
    ("mwd discriminator", None,           "net.mwd",           "net.mwd"),
    ("wd discriminator",  None,           "net.wd",            "net.wd"),
]

print(f"  {'Module':<22} {'Kokoro-82M':>12}  {'StyleTTS2-LJS':>13}  {'StyleTTS2-Lib':>13}")
print(f"  {'─'*22} {'─'*12}  {'─'*13}  {'─'*13}")
for label, kok_pfx, ljs_pfx, lib_pfx in mappings:
    kv = count_by_prefix(kok_flat, kok_pfx) if kok_pfx else 0
    lv = count_by_prefix(ljs_flat, ljs_pfx) if ljs_pfx else 0
    rv = count_by_prefix(lib_flat, lib_pfx) if lib_pfx else 0
    kv_s = f"{kv:,}" if kv else "—"
    lv_s = f"{lv:,}" if lv else "—"
    rv_s = f"{rv:,}" if rv else "—"
    print(f"  {label:<22} {kv_s:>12}  {lv_s:>13}  {rv_s:>13}")

# ── Decoder comparison ─────────────────────────────────────────────────────────

sep("DECODER ARCHITECTURE COMPARISON")
print("""
  Kokoro-82M & StyleTTS2-LJSpeech share iSTFTNet decoder:
    • upsample_rates   = [10, 6]
    • upsample_k_sizes = [20, 12]
    • gen_istft_n_fft  = 20
    • gen_istft_hop    = 5
    • total upsample   = 10 × 6 × 5 = 300 samples/frame @ 24kHz
    • HiFiGAN-style AdaINResBlock + Snake activation
    • NSF source: SineGen (8 harmonics) for pitch-aware excitation

  StyleTTS2-LibriTTS uses HiFiGAN decoder (4-stage):
    • upsample_rates   = [10, 5, 3, 2]
    • total upsample   = 10 × 5 × 3 × 2 = 300 (same!)
    • Standard LeakyReLU resblocks (no Snake activation)
    • No NSF/SineGen — relies on F0 input differently
    • More stages → finer temporal resolution at each step
""")

# ── Style conditioning ─────────────────────────────────────────────────────────

sep("STYLE CONDITIONING MECHANISM")
print(f"""
  All models: style_dim = {kok_cfg['style_dim']} (128)

  Kokoro-82M:
    • ref_s is a 256-dim vector from a voicepack .pt file
    • ref_s[:, :128] → decoder (iSTFTNet AdaIN layers)
    • ref_s[:, 128:] → predictor (DurationEncoder AdaLayerNorm)
    • Style is FIXED at inference — no diffusion needed
    • 54 voice packs available (different speakers/languages)
    • Voice pack tensors are pre-computed style embeddings

  StyleTTS2-LJSpeech:
    • style_encoder encodes reference mel-spectrogram → 128-dim
    • Diffusion model generates style from text (no ref needed)
    • Single speaker — style variation is expressiveness only
    • Style fed into: TextEncoder (via predictor), Decoder (AdaIN)

  StyleTTS2-LibriTTS:
    • Same as LJSpeech but adds speaker embedding
    • Diffusion model conditioned on speaker + text
    • Multi-speaker: style encodes both identity + expression
    • sigma_data much lower (0.199 vs 0.457) — tighter prior
      because speaker distribution is more structured
""")

# ── Voice conditioning deep dive ───────────────────────────────────────────────

sep("VOICE CONDITIONING: Kokoro voicepacks vs StyleTTS2 reference")
voices_raw = {}
for vf in sorted(VOICES_DIR.glob("*.pt")):
    voices_raw[vf.stem] = torch.load(vf, map_location="cpu", weights_only=True)
# shape: (510, 1, 256) — 510 reference utterance style frames per voice
voices_mean = {n: t.squeeze(1).float().mean(dim=0) for n, t in voices_raw.items()}  # (256,)
all_voices = torch.stack(list(voices_mean.values()))   # (N, 256)

print(f"\n  Kokoro voice pack statistics ({len(voices_raw)} voices):")
print(f"    Raw tensor shape: (510, 1, 256) per voice")
print(f"      510 = length-indexed entries used as pack[len(phonemes)-1]")
print(f"        1 = batch dim")
print(f"      256 = style_dim*2 (decoder:128 + prosody:128)")
print(f"    Mean-pooled stats (across 510 frames × 54 voices):")
print(f"    Overall mean:     {all_voices.mean():.6f}")
print(f"    Overall std:      {all_voices.std():.6f}")
print(f"    Norm range:       {all_voices.norm(dim=1).min():.4f} – {all_voices.norm(dim=1).max():.4f}")
print(f"    Decoder half std: {all_voices[:, :128].std():.6f}")
print(f"    Prosody half std: {all_voices[:, 128:].std():.6f}")
adj = torch.nn.functional.cosine_similarity(
    torch.stack([t.squeeze(1)[:-1].float() for t in voices_raw.values()]),
    torch.stack([t.squeeze(1)[1:].float() for t in voices_raw.values()]),
    dim=-1,
)
print(f"    Adjacent-index cosine mean: {adj.mean():.6f}")
print(f"\n  StyleTTS2 reference audio style (inferred at runtime):")
print(f"    Style dim: 128 (single vector per reference utterance)")
print(f"    Extracted via style_encoder (mel → latent)")
print(f"    LJSpeech sigma_data ≈ 0.457 (wide style distribution)")
print(f"    LibriTTS sigma_data ≈ 0.199 (compact — many speakers)")
print(f"\n  Key difference:")
print(f"    Kokoro ships 54 pre-computed 256-dim voice tensors with a")
print(f"    smooth length-conditioned axis, avoiding runtime style extraction.")
print(f"    This is why")
print(f"    Kokoro has no style_encoder or diffusion module at inference.")

# ── Weight-space similarity for shared TextEncoder ─────────────────────────────

sep("WEIGHT-SPACE ANALYSIS: TextEncoder embedding (Kokoro vs StyleTTS2-LJS)")
try:
    kok_emb = kok_flat.get("text_encoder.module.embedding.weight")
    ljs_emb = ljs_flat.get("net.text_encoder.module.embedding.weight")

    if kok_emb is not None and ljs_emb is not None:
        print(f"\n  Kokoro   embedding: {list(kok_emb.shape)}")
        print(f"  LJS      embedding: {list(ljs_emb.shape)}")
        if kok_emb.shape == ljs_emb.shape:
            diff = (kok_emb.float() - ljs_emb.float())
            print(f"  L2 diff norm:    {diff.norm():.4f}")
            print(f"  Mean abs diff:   {diff.abs().mean():.6f}")
            cos = torch.nn.functional.cosine_similarity(
                kok_emb.float().reshape(1, -1),
                ljs_emb.float().reshape(1, -1)
            )
            print(f"  Cosine similarity (flattened): {cos.item():.6f}")
            print(f"  → {'Very similar' if cos > 0.99 else 'Diverged significantly'}")
        else:
            print(f"  Shapes differ — cannot directly compare")
    else:
        print(f"  Embedding keys not found under expected paths")
        kok_emb_k = [k for k in kok_flat if "embedding" in k]
        ljs_emb_k = [k for k in ljs_flat if "embedding" in k]
        print(f"  Kokoro embedding keys: {kok_emb_k}")
        print(f"  LJS    embedding keys: {ljs_emb_k}")
except Exception as e:
    print(f"  Skipped: {e}")

# ── Training objectives ────────────────────────────────────────────────────────

sep("TRAINING OBJECTIVES COMPARISON")
print("""
  Kokoro-82M:
    • Derived from StyleTTS2 architecture, trained by hexgrad
    • No public training code — inference-only release
    • Likely trained with a close relative of the StyleTTS2 losses
      (mel, dur, F0, N, alignment), but the exact Kokoro recipe is unknown
    • Voice style is baked into pre-computed packs at inference
    • Any stronger claim about SLM adversarial training is still inference,
      not directly recoverable from the released Kokoro checkpoint

  StyleTTS2-LJSpeech (from config):
    Loss weights:
      lambda_mel=5.0   (mel reconstruction — dominant)
      lambda_dur=1.0   (duration prediction)
      lambda_F0 =1.0   (pitch prediction)
      lambda_gen=1.0   (GAN generator)
      lambda_slm=1.0   (WavLM SLM adversarial)
      lambda_diff=1.0  (style diffusion)
      lambda_sty=1.0   (style consistency)
      lambda_ce =20.0  (cross-entropy alignment — very high)
      lambda_s2s=1.0   (sequence-to-sequence)
      lambda_mono=1.0  (monotonic alignment)
      lambda_norm=1.0  (normalization)
    TMA epoch: 50  (text-mel alignment starts at epoch 50)
    Diff epoch: 20 (diffusion training starts at epoch 20)
    Joint epoch: 50 (joint training starts at epoch 50)

  StyleTTS2-LibriTTS (from config):
    Same loss weights, but:
      TMA epoch: 4    (much earlier — smaller dataset)
      Diff/Joint: 0   (start immediately — pretrained weights)
      Smaller max_len (300 vs 400 frames)
      Larger slmadv iter (20 vs 10) — more SLM adversarial steps
""")

sep("INFERENCE PIPELINE COMPARISON")
print("""
  Kokoro-82M:
    Text → G2P (misaki) → phonemes → PL-BERT → bert_encoder
         → TextEncoder + ProsodyPredictor (with voicepack style)
         → alignment + F0/N prediction
         → iSTFTNet Decoder (with voicepack style)
         → 24kHz waveform

  StyleTTS2 (LJSpeech):
    Text → phonemizer → PL-BERT → TextEncoder + ProsodyPredictor
         → style diffusion (text-only, no reference needed)
         → alignment + F0/N prediction
         → iSTFTNet Decoder (with diffused style)
         → 24kHz waveform

  StyleTTS2 (LibriTTS):
    Text + reference_audio → phonemizer → PL-BERT → TextEncoder
         → style_encoder(ref mel) OR diffusion (speaker-conditioned)
         → alignment + F0/N prediction
         → HiFiGAN Decoder (with style)
         → 24kHz waveform

  Key speed difference:
    Kokoro skips style diffusion entirely (pre-computed voicepacks),
    making inference significantly faster. StyleTTS2 runs a diffusion
    denoising loop (default ~5–20 steps) at each inference call.
""")

sep()
print("Done.")
