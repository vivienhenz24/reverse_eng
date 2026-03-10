"""
Verify how similar Kokoro's released weights are to StyleTTS2 checkpoints.

This script compares shared module tensors between:
  - Kokoro vs StyleTTS2 LJSpeech
  - Kokoro vs StyleTTS2 LibriTTS
  - StyleTTS2 LJSpeech vs StyleTTS2 LibriTTS

It answers:
  - Which shared tensors are exact copies?
  - Which modules stayed close in weight space?
  - Which modules clearly diverged despite matching shapes?
"""

from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent

KOKORO_MODEL = ROOT / "kokoro/weights/kokoro-v1_0.pth"
LJS_CKPT = ROOT / "StyleTTS2/Models/LJSpeech/Models/LJSpeech/epoch_2nd_00100.pth"
LIB_CKPT = ROOT / "StyleTTS2/Models/LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth"

SHARED_MODULES = ["bert", "bert_encoder", "text_encoder", "predictor", "decoder"]


def sep(title="", width=72, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, torch.Tensor):
        out[prefix] = obj
    return out


def load_kokoro():
    raw = torch.load(KOKORO_MODEL, map_location="cpu", weights_only=True)
    flat = {}
    for mod, sd in raw.items():
        for k, v in sd.items():
            flat[f"{mod}.{k}"] = v
    return flat


def load_styletts(path):
    return flatten(torch.load(path, map_location="cpu", weights_only=False))


def shared_tensor_pairs(flat_a, flat_b, prefixes_a, prefixes_b):
    pairs = []
    for prefix_a, prefix_b in zip(prefixes_a, prefixes_b):
        keys_a = {
            k[len(prefix_a) + 1:]: v
            for k, v in flat_a.items()
            if k.startswith(prefix_a + ".")
        }
        keys_b = {
            k[len(prefix_b) + 1:]: v
            for k, v in flat_b.items()
            if k.startswith(prefix_b + ".")
        }
        for rel in sorted(set(keys_a) & set(keys_b)):
            ta = keys_a[rel]
            tb = keys_b[rel]
            if ta.shape != tb.shape:
                continue
            pairs.append((prefix_a, rel, ta, tb))
    return pairs


def compare(name, flat_a, flat_b, prefixes_a, prefixes_b):
    pairs = shared_tensor_pairs(flat_a, flat_b, prefixes_a, prefixes_b)
    rows = []
    for module, rel, ta, tb in pairs:
        module_label = module.split(".")[-1]
        a = ta.float().reshape(-1)
        b = tb.float().reshape(-1)
        diff = a - b
        cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        cos = max(min(cos, 1.0), -1.0)
        rows.append({
            "module": module_label,
            "rel": rel,
            "numel": a.numel(),
            "exact": bool(torch.equal(ta, tb)),
            "cos": cos,
            "mae": diff.abs().mean().item(),
            "max_abs": diff.abs().max().item(),
            "l2": diff.norm().item(),
        })

    sep(name)
    print(f"  shared tensors compared: {len(rows)}")
    print(f"  shared parameters:       {sum(r['numel'] for r in rows):,}")

    sep("MODULE SUMMARY", char="·")
    print(f"  {'Module':<14} {'Tensors':>8} {'Exact':>8} {'Mean cos':>10} {'Mean MAE':>10}")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
    for module in SHARED_MODULES:
        subset = [r for r in rows if r["module"] == module]
        if not subset:
            continue
        exact = sum(r["exact"] for r in subset)
        mean_cos = sum(r["cos"] for r in subset) / len(subset)
        mean_mae = sum(r["mae"] for r in subset) / len(subset)
        print(f"  {module:<14} {len(subset):>8} {exact:>8} {mean_cos:>10.6f} {mean_mae:>10.6f}")

    exact_rows = [r for r in rows if r["exact"]]
    sep("TOP NEAREST TENSORS", char="·")
    print(f"  {'Cosine':>8} {'MAE':>10}  Tensor")
    nearest = sorted(rows, key=lambda r: (-r["cos"], r["mae"]))[:15]
    for r in nearest:
        print(f"  {r['cos']:8.6f} {r['mae']:10.6f}  {r['module']}.{r['rel']}")

    sep("TOP MOST DIVERGED TENSORS", char="·")
    print(f"  {'Cosine':>8} {'MAE':>10}  Tensor")
    farthest = sorted(rows, key=lambda r: (r["cos"], -r["mae"]))[:15]
    for r in farthest:
        print(f"  {r['cos']:8.6f} {r['mae']:10.6f}  {r['module']}.{r['rel']}")

    sep("EXACT MATCHES", char="·")
    print(f"  exact tensor matches: {len(exact_rows)}")
    for r in exact_rows[:20]:
        print(f"  {r['module']}.{r['rel']}")
    if len(exact_rows) > 20:
        print(f"  ... and {len(exact_rows) - 20} more")


print("=" * 72)
print("  WEIGHT SHARING VERIFICATION")
print("=" * 72)

kok = load_kokoro()
ljs = load_styletts(LJS_CKPT)
lib = load_styletts(LIB_CKPT)

compare(
    "Kokoro vs StyleTTS2 LJSpeech",
    kok,
    ljs,
    SHARED_MODULES,
    [f"net.{m}" for m in SHARED_MODULES],
)
compare(
    "Kokoro vs StyleTTS2 LibriTTS",
    kok,
    lib,
    SHARED_MODULES,
    [f"net.{m}" for m in SHARED_MODULES],
)
compare(
    "StyleTTS2 LJSpeech vs LibriTTS",
    ljs,
    lib,
    [f"net.{m}" for m in SHARED_MODULES],
    [f"net.{m}" for m in SHARED_MODULES],
)

sep()
print("Done.")
