"""
Prototype the simplest training-time voicepack mechanism consistent with
Kokoro inference:

    ref_s = voice_table[voice_id, len(phonemes) - 1]

This does not prove Kokoro used this exact code. It makes the strongest current
hypothesis concrete and checks that the parameter count and tensor contract are
fully compatible with the released artifacts.
"""

from dataclasses import dataclass


@dataclass
class VoicepackTableSpec:
    num_voices: int = 54
    max_length: int = 510
    style_dim: int = 256

    @property
    def total_params(self) -> int:
        return self.num_voices * self.max_length * self.style_dim

    @property
    def fp32_mib(self) -> float:
        return self.total_params * 4 / (1024 * 1024)


def sep(title="", width=88, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)


spec = VoicepackTableSpec()

print("=" * 88)
print("  PROTOTYPE TRAINABLE VOICEPACK TABLE")
print("=" * 88)

sep("TABLE SHAPE")
print(f"  voices:       {spec.num_voices}")
print(f"  max length:   {spec.max_length}")
print(f"  style dim:    {spec.style_dim}")
print(f"  tensor shape: ({spec.num_voices}, {spec.max_length}, {spec.style_dim})")

sep("PARAMETER COST")
print(f"  parameters:   {spec.total_params:,}")
print(f"  fp32 size:    {spec.fp32_mib:.2f} MiB")

sep("INFERENCE CONTRACT")
print("  Kokoro runtime already does exactly this lookup shape-wise:")
print("    ref_s = pack[len(phonemes) - 1]")
print("  So a per-voice training-time table is fully compatible with the released")
print("  artifacts and the shipped inference path.")

sep("WHY THIS IS PLAUSIBLE")
print("  - The table is small enough to train directly.")
print("  - It removes the need to ship style encoder and diffusion modules.")
print("  - It matches public wording about 'trained voicepacks'.")
print("  - It explains why voicepacks can be averaged or manipulated directly.")

sep("CANDIDATE TRAINING-TIME USE")
print("  For each batch item:")
print("    1. choose voice_id")
print("    2. compute phoneme length index = min(len(ps), 510) - 1")
print("    3. fetch ref_s = voice_table[voice_id, length_index]")
print("    4. run Kokoro backbone exactly as in inference, conditioned on ref_s")
print("    5. backprop into model weights and voice_table")

sep("KEY UNKNOWN")
print("  This still leaves one unresolved choice:")
print("    was the table learned from scratch as a parameter,")
print("    or initialized from StyleTTS2-like style features and then optimized?")

sep()
print("Done.")
