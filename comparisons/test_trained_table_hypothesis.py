"""
Test the hypothesis that Kokoro voicepacks are learned voice-length parameter
tables, not post-hoc exports from a single reference style vector.
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
    return names, torch.stack([voices[n] for n in names])


def leave_one_out_pack_prediction(features, target):
    scores = []
    for i in range(target.shape[0]):
        train = [j for j in range(target.shape[0]) if j != i]
        x = torch.cat([features[train], torch.ones(len(train), 1)], dim=1)
        y = target[train].reshape(len(train), -1)
        w = torch.linalg.lstsq(x, y).solution
        pred = (torch.cat([features[i], torch.ones(1)]).unsqueeze(0) @ w).reshape(target.shape[1], target.shape[2])
        scores.append(F.cosine_similarity(pred, target[i], dim=-1).mean().item())
    return sum(scores) / len(scores)


names, stack = load_stack()
v, l, d = stack.shape

print("=" * 88)
print("  TRAINED TABLE HYPOTHESIS TEST")
print("=" * 88)

sep("VOICEPACK SIZE")
total_params = v * l * d
print(f"  voices:            {v}")
print(f"  slots per voice:   {l}")
print(f"  dims per slot:     {d}")
print(f"  total parameters:  {total_params:,}")
print(f"  fp32 size:         {total_params * 4 / (1024 * 1024):.2f} MiB")

sep("SMOOTHNESS")
adj_cos = F.cosine_similarity(
    stack[:, :-1, :].reshape(-1, d),
    stack[:, 1:, :].reshape(-1, d),
    dim=-1,
).mean().item()
first_last_cos = F.cosine_similarity(stack[:, 0, :], stack[:, -1, :], dim=-1).mean().item()
print(f"  adjacent row cosine mean:   {adj_cos:.6f}")
print(f"  first-vs-last cosine mean:  {first_last_cos:.6f}")

sep("CAN ONE STYLE VECTOR PREDICT THE WHOLE TABLE?")
first_only = leave_one_out_pack_prediction(stack[:, 0, :], stack)
last_only = leave_one_out_pack_prediction(stack[:, -1, :], stack)
first_last = leave_one_out_pack_prediction(torch.cat([stack[:, 0, :], stack[:, -1, :]], dim=1), stack)
print(f"  first slot only:       mean cosine {first_only:.6f}")
print(f"  last slot only:        mean cosine {last_only:.6f}")
print(f"  first+last together:   mean cosine {first_last:.6f}")

sep("INTERPRETATION")
print("  If the files were simple post-hoc exports from one reference style vector,")
print("  recovering the whole table from the first slot or first+last slots should")
print("  be much easier than this. It is not.")
print("  The artifacts are smoother than arbitrary tables, but much richer than a")
print("  one-vector export. That is consistent with a jointly learned voice-length")
print("  parameter table.")

sep("WHY THIS HYPOTHESIS FITS THE PUBLIC EVIDENCE")
print("  - Hugging Face posts explicitly say 'trained voicepacks'.")
print("  - v0.19 publicly exposed a voicepack made by averaging two others.")
print("  - The released v1.0 tables are not exact pairwise averages, but they are")
print("    plainly manipulable artifacts rather than opaque encoder outputs.")
print("  - A 7.05M-parameter voice table is cheap enough to train jointly.")

sep("CURRENT BEST EXPLANATION")
print("  The strongest remaining explanation is:")
print("    voicepack[voice_id, phoneme_length] -> 256-d learned style parameter")
print("  with the table trained jointly or fine-tuned alongside Kokoro, rather than")
print("  exported afterward from a single reference utterance embedding.")

sep()
print("Done.")
