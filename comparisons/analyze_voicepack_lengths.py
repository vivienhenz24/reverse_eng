"""
Analyze the length-conditioned axis inside Kokoro voice packs.

Kokoro inference uses:
  ref_s = pack[len(phonemes)-1]

So this script treats the first axis of each voice tensor as a lookup table
over phoneme length and measures how that axis behaves.
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
step0 = stack[:, 0, :]
step_last = stack[:, -1, :]

print("=" * 72)
print("  KOKORO VOICEPACK LENGTH-AXIS ANALYSIS")
print("=" * 72)

sep("GLOBAL")
print(f"  voices:        {stack.shape[0]}")
print(f"  length slots:  {stack.shape[1]}  (usable for phoneme lengths 1..510)")
print(f"  style dim:     {stack.shape[2]}")
print(f"  adjacent cosine mean: {adj.mean():.6f}")
print(f"  adjacent cosine std:  {adj.std():.6f}")
print(f"  adjacent cosine min:  {adj.min():.6f}")
first_last = F.cosine_similarity(step0, step_last, dim=-1)
print(f"  first-last cosine mean: {first_last.mean():.6f}")
print(f"  first-last cosine std:  {first_last.std():.6f}")

sep("PER-VOICE DRIFT")
print(f"  {'Voice':<25} {'adj_cos':>9} {'first_last':>11} {'len_var':>10}")
print(f"  {'─'*25} {'─'*9} {'─'*11} {'─'*10}")
for i, name in enumerate(names):
    len_var = stack[i].var(dim=0).mean().item()
    print(
        f"  {name:<25} "
        f"{adj[i].mean().item():>9.6f} "
        f"{first_last[i].item():>11.6f} "
        f"{len_var:>10.6f}"
    )

sep("WHICH LENGTHS CHANGE MOST ACROSS VOICES")
per_len_voice_var = stack.var(dim=0).mean(dim=1)
top = torch.topk(per_len_voice_var, 15)
print(f"  {'Len':>5} {'Mean voice-var':>14}")
print(f"  {'─'*5} {'─'*14}")
for idx, val in zip(top.indices.tolist(), top.values.tolist()):
    print(f"  {idx + 1:>5} {val:>14.6f}")

sep("WHICH LENGTH TRANSITIONS CHANGE MOST")
delta = (stack[:, 1:, :] - stack[:, :-1, :]).norm(dim=-1).mean(dim=0)
top = torch.topk(delta, 15)
print(f"  {'From->To':>9} {'Mean delta L2':>14}")
print(f"  {'─'*9} {'─'*14}")
for idx, val in zip(top.indices.tolist(), top.values.tolist()):
    print(f"  {idx + 1:>4}->{idx + 2:<4} {val:>14.6f}")

sep("DECODER VS PROSODY HALF")
dec = stack[:, :, :128]
pro = stack[:, :, 128:]
dec_adj = F.cosine_similarity(dec[:, :-1, :], dec[:, 1:, :], dim=-1)
pro_adj = F.cosine_similarity(pro[:, :-1, :], pro[:, 1:, :], dim=-1)
print(f"  decoder adjacent cosine mean: {dec_adj.mean():.6f}")
print(f"  prosody adjacent cosine mean: {pro_adj.mean():.6f}")
print(f"  decoder length variance mean: {dec.var(dim=1).mean().item():.6f}")
print(f"  prosody length variance mean: {pro.var(dim=1).mean().item():.6f}")

sep("INTERPRETATION")
print("  The length axis is extremely smooth locally and drifts gradually over long ranges.")
print("  That supports a length-conditioned lookup-table interpretation, not independent samples.")

sep()
print("Done.")
