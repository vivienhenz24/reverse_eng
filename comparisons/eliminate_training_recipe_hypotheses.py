"""
Eliminate Turkish training-recipe hypotheses using:
  - the released Kokoro checkpoint contents
  - Kokoro's direct voicepack lookup inference path
  - StyleTTS2's published training losses

The goal is not to prove every historical detail of Kokoro's private training.
The goal is to identify the smallest plausible recipe for adding Turkish now.
"""

from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT / "kokoro/weights/kokoro-v1_0.pth"


def sep(title="", width=88, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)


ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
released_modules = set(ckpt.keys())

recipe = [
    {
        "name": "diffusion loss",
        "status": "eliminated",
        "reason": "The direct voice-table hypothesis removes the need to sample style latents during Turkish finetuning.",
    },
    {
        "name": "style reconstruction loss against s_trg",
        "status": "eliminated",
        "reason": "This depends on StyleTTS2 style encoders and diffusion targets that are not part of the current direct-table path.",
    },
    {
        "name": "text aligner / monotonic alignment training",
        "status": "eliminated",
        "reason": "The released Kokoro checkpoint does not contain a text aligner; for practical Turkish finetuning, use existing durations from the predictor path or external alignments instead.",
    },
    {
        "name": "style_encoder training",
        "status": "eliminated",
        "reason": "No style encoder is present in the released Kokoro checkpoint, and the direct voice-table path does not require one.",
    },
    {
        "name": "predictor_encoder training",
        "status": "eliminated",
        "reason": "Same reason as style_encoder: the direct voice-table path bypasses it.",
    },
    {
        "name": "mel / STFT reconstruction loss",
        "status": "required",
        "reason": "You need a direct audio reconstruction objective for Turkish finetuning.",
    },
    {
        "name": "duration loss",
        "status": "required",
        "reason": "Kokoro's predictor still has to learn token-to-frame timing for Turkish.",
    },
    {
        "name": "F0 loss",
        "status": "likely",
        "reason": "The released predictor explicitly generates F0; keeping pitch supervision is the safest direct carry-over.",
    },
    {
        "name": "norm / energy loss",
        "status": "likely",
        "reason": "The released predictor explicitly generates N; matching energy/noise remains structurally aligned with Kokoro.",
    },
    {
        "name": "GAN discriminator / generator loss",
        "status": "optional",
        "reason": "Helpful for quality, but not needed for the first Turkish intelligibility proof of concept.",
    },
    {
        "name": "SLM / WavLM loss",
        "status": "optional",
        "reason": "Helpful later, but too much moving machinery for the first Turkish recovery attempt.",
    },
    {
        "name": "voicepack smoothness regularization",
        "status": "recommended",
        "reason": "The released voice tables are extremely smooth over length; regularizing adjacent slots preserves that structure.",
    },
]

print("=" * 88)
print("  TRAINING RECIPE HYPOTHESIS ELIMINATION")
print("=" * 88)

sep("RELEASED MODULES")
for name in sorted(released_modules):
    print(f"  {name}")

sep("LOSS / COMPONENT VERDICTS")
for row in recipe:
    print(f"  {row['name']:<36} {row['status']}")
    print(f"    {row['reason']}")

sep("MINIMAL TURKISH RECIPE")
print("  Train:")
print("    - Turkish voicepack table")
print("    - bert_encoder")
print("    - text_encoder")
print("    - predictor")
print("  Start with decoder frozen.")
print("  Use losses:")
print("    - mel / STFT reconstruction")
print("    - duration")
print("    - F0")
print("    - norm / energy")
print("    - voicepack smoothness regularization")
print("  Add GAN / SLM only after intelligibility is working.")

sep("TAKEAWAY")
print("  For adding Turkish now, most of the large StyleTTS2 training recipe can be")
print("  dropped. The smallest plausible path is a direct Kokoro finetune with a")
print("  trainable Turkish voice table plus core reconstruction/prosody losses.")

sep()
print("Done.")
