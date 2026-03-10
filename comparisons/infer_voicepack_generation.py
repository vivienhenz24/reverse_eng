"""
Infer plausible Kokoro voicepack generation recipes from local code + weights.

This script does not claim ground truth. It combines:
  - StyleTTS2 training code paths
  - Kokoro runtime voicepack usage
  - Measured voicepack statistics

to rank hypotheses for how Kokoro's (510, 1, 256) voice packs were likely built.
"""

from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = ROOT / "kokoro/weights/voices"


def sep(title="", width=72, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)


voices = {}
for vf in sorted(VOICES_DIR.glob("*.pt")):
    voices[vf.stem] = torch.load(vf, map_location="cpu", weights_only=True).squeeze(1).float()

names = sorted(voices)
stack = torch.stack([voices[n] for n in names])  # (V, 510, 256)

adj = F.cosine_similarity(stack[:, :-1, :], stack[:, 1:, :], dim=-1)
first_last = F.cosine_similarity(stack[:, 0, :], stack[:, -1, :], dim=-1)
per_len_voice_var = stack.var(dim=0).mean(dim=1)
delta = (stack[:, 1:, :] - stack[:, :-1, :]).norm(dim=-1).mean(dim=0)
dec = stack[:, :, :128]
pro = stack[:, :, 128:]

stats = {
    "n_voices": stack.shape[0],
    "n_slots": stack.shape[1],
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

hypotheses = [
    {
        "name": "Direct utterance-style stack from 510 reference clips",
        "supports": [
            "Would explain a 510-long axis if each row came from a different utterance.",
        ],
        "conflicts": [
            "Kokoro runtime uses pack[len(phonemes)-1], so row index is treated as phoneme length.",
            "Adjacent-row cosine is almost 1.0; independent utterance embeddings should not form such a smooth trajectory.",
            "Variance and step size are front-loaded at very short lengths, which looks like a lookup-table schedule, not a shuffled bank.",
        ],
        "score": 0.05,
    },
    {
        "name": "Single style vector repeated 510 times",
        "supports": [
            "Would be easy to export from a standard StyleTTS2-style embedding.",
        ],
        "conflicts": [
            "First-vs-last cosine is only about 0.53 on average, so the rows are not constant.",
            "Prosody half varies substantially more than decoder half across the length axis.",
        ],
        "score": 0.10,
    },
    {
        "name": "Length-conditioned interpolation from one or a few base style vectors",
        "supports": [
            "Matches Kokoro runtime indexing by phoneme length.",
            "Matches extremely high adjacent cosine with gradual long-range drift.",
            "Matches strong changes at the first few length slots and tapering later.",
            "Matches stronger variation in the prosody half than decoder half.",
        ],
        "conflicts": [
            "Does not by itself explain where the base style vectors came from.",
        ],
        "score": 0.88,
    },
    {
        "name": "Diffusion-generated style table conditioned on length and speaker/reference",
        "supports": [
            "StyleTTS2 already trains a diffusion model over 256-dim style targets.",
            "Could produce a smooth table if sampled or denoised over a length-conditioned schedule.",
            "Compatible with Kokoro removing diffusion only at inference time.",
        ],
        "conflicts": [
            "No public Kokoro code exposes such a generation step directly.",
            "This is a training/export hypothesis, not something observable in the released inference checkpoint alone.",
        ],
        "score": 0.72,
    },
    {
        "name": "Teacher-forced extraction over many text lengths using StyleEncoder + PredictorEncoder",
        "supports": [
            "StyleTTS2 training explicitly forms 256-dim targets by concatenating acoustic and prosodic style encoders.",
            "A fixed speaker/reference combined with varying text-length contexts could yield a per-length table.",
            "Prosody half being more length-sensitive fits the predictor-side role in Kokoro inference.",
        ],
        "conflicts": [
            "StyleEncoder itself uses global average pooling, so raw encoder output is utterance-level; some extra export logic is still needed.",
        ],
        "score": 0.80,
    },
]


print("=" * 72)
print("  KOKORO VOICEPACK GENERATION INFERENCE")
print("=" * 72)

sep("OBSERVED FACTS")
print(f"  voice count:                     {stats['n_voices']}")
print(f"  slots per voicepack:            {stats['n_slots']}")
print(f"  adjacent-step cosine mean:      {stats['adj_mean']:.6f}")
print(f"  adjacent-step cosine min:       {stats['adj_min']:.6f}")
print(f"  first-vs-last cosine mean:      {stats['first_last_mean']:.6f}")
print(f"  per-length voice variance @1:   {stats['first_len_var']:.6f}")
print(f"  per-length voice variance @10:  {stats['len_10_var']:.6f}")
print(f"  per-length voice variance @100: {stats['len_100_var']:.6f}")
print(f"  per-length voice variance @510: {stats['last_len_var']:.6f}")
print(f"  mean delta L2 1->2:             {stats['first_delta']:.6f}")
print(f"  mean delta L2 10->11:           {stats['delta_10']:.6f}")
print(f"  mean delta L2 100->101:         {stats['delta_100']:.6f}")
print(f"  decoder-half length variance:   {stats['decoder_len_var']:.6f}")
print(f"  prosody-half length variance:   {stats['prosody_len_var']:.6f}")

sep("CODE FACTS")
print("  StyleTTS2 training builds a 256-dim style target as:")
print("    s_trg = concat(style_encoder(mel), predictor_encoder(mel))")
print("  StyleEncoder uses AdaptiveAvgPool2d(1), so its output is utterance-level.")
print("  Kokoro inference does not run style encoders or diffusion.")
print("  Kokoro inference selects ref_s with pack[len(phonemes)-1].")

sep("HYPOTHESIS RANKING")
for rank, h in enumerate(sorted(hypotheses, key=lambda x: x["score"], reverse=True), start=1):
    print(f"  {rank}. {h['name']}  score={h['score']:.2f}")
    print("     supports:")
    for s in h["supports"]:
        print(f"       - {s}")
    print("     conflicts:")
    for s in h["conflicts"]:
        print(f"       - {s}")

sep("WORKING CONCLUSION")
print("  Most likely: Kokoro voicepacks are exported length-conditioned style tables.")
print("  The underlying 256-dim style space is still consistent with StyleTTS2's")
print("  acoustic-style + prosodic-style split, but an extra offline export step")
print("  appears to map a voice/reference identity into 510 phoneme-length slots.")
print("  The next reverse-engineering target is that export step.")

sep("NEXT QUESTIONS")
print("  1. Was the table produced by diffusion sampling, by deterministic interpolation,")
print("     or by teacher-forced extraction over many length contexts?")
print("  2. What reference audio or speaker statistics seeded each shipped voice?")
print("  3. Was the same table used during training, or only compiled post-training for inference?")

sep()
print("Done.")
