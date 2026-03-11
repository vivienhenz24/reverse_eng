"""
Find the minimum terminal-centered basis size that round-trips the shipped
voicepacks exactly at float32 precision.
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
        voices[vf.stem] = torch.load(vf, map_location="cpu", weights_only=True).squeeze(1)
    names = sorted(voices)
    return names, torch.stack([voices[n] for n in names])


def reconstruct(stack, k):
    stack64 = stack.double()
    terminal = stack64[:, -1, :]
    residual = stack64 - terminal.unsqueeze(1)
    mat = residual.permute(1, 0, 2).reshape(stack.shape[1], -1)
    u, s, _ = torch.linalg.svd(mat, full_matrices=False)
    basis = u[:, :k]
    coeff = torch.einsum("vld,lk->vkd", residual, basis)
    recon = terminal.unsqueeze(1) + torch.einsum("lk,vkd->vld", basis, coeff)
    return recon.float(), s


names, stack = load_stack()

print("=" * 88)
print("  EXACT VOICEPACK ROUNDTRIP")
print("=" * 88)

sep("SEARCH")
lo, hi = 1, stack.shape[1]
answer = None
while lo <= hi:
    mid = (lo + hi) // 2
    recon, _ = reconstruct(stack, mid)
    if torch.equal(recon, stack):
        answer = mid
        hi = mid - 1
    else:
        lo = mid + 1

print(f"  minimal exact round-trip k: {answer}")

sep("CHECKPOINTS")
for k in [4, 8, 11, 16, 32, 64, 128, 256, answer]:
    recon, _ = reconstruct(stack, k)
    mismatches = int((recon != stack).sum().item())
    max_abs = float((recon - stack).abs().max().item())
    print(f"  k={k:<4} mismatches={mismatches:<9} max abs={max_abs:.9f}")

sep("TAKEAWAY")
print("  No compact low-rank factorization reproduces the stored float32 files exactly.")
print("  Exact round-trip requires essentially the full terminal-centered row rank.")

sep()
print("Done.")
