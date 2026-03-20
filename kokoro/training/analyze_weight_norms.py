"""
analyze_weight_norms.py

Analysis 1: Norm of text_encoder layers vs. other components.

Shows how much "capacity" each component holds, which guides
how aggressively to set learning rates when finetuning for Turkish.

High norms in early layers → those layers have strong priors, need
lower LR to avoid disrupting them.  Low norms → more plastic, can
tolerate higher LR.

Usage:
    python kokoro/training/analyze_weight_norms.py
    python kokoro/training/analyze_weight_norms.py --verbose
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHECKPOINT_PATH, CONFIG_PATH, load_local_kmodel


def layer_stats(params: list[torch.Tensor]) -> dict:
    norms = [p.data.norm(2).item() for p in params]
    total_params = sum(p.numel() for p in params)
    zero_count = sum((p.data.abs() < 1e-6).sum().item() for p in params)
    mean_norm = sum(norms) / len(norms)
    var = sum((n - mean_norm) ** 2 for n in norms) / len(norms)
    return {
        "n_tensors": len(norms),
        "total_params": total_params,
        "mean_norm": mean_norm,
        "std_norm": math.sqrt(var),
        "min_norm": min(norms),
        "max_norm": max(norms),
        "sparsity_pct": 100.0 * zero_count / total_params,
    }


def print_component_table(components: dict, model):
    col = "{:<35} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
    header = col.format("Component", "Params", "Mean|W|", "Std|W|", "Min|W|", "Max|W|", "Sparse%")
    print("\n" + "=" * len(header))
    print("COMPONENT SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, module in components.items():
        params = list(module.parameters())
        if not params:
            continue
        s = layer_stats(params)
        print(col.format(
            name,
            f"{s['total_params']:,}",
            f"{s['mean_norm']:.4f}",
            f"{s['std_norm']:.4f}",
            f"{s['min_norm']:.4f}",
            f"{s['max_norm']:.4f}",
            f"{s['sparsity_pct']:.2f}",
        ))
    print("-" * len(header))
    # totals
    all_params = list(model.parameters())
    s = layer_stats(all_params)
    print(col.format(
        "TOTAL",
        f"{s['total_params']:,}",
        f"{s['mean_norm']:.4f}",
        f"{s['std_norm']:.4f}",
        f"{s['min_norm']:.4f}",
        f"{s['max_norm']:.4f}",
        f"{s['sparsity_pct']:.2f}",
    ))


def print_layer_breakdown(name: str, module: torch.nn.Module):
    params = list(module.named_parameters())
    if not params:
        print(f"  (no parameters)")
        return
    col = "  {:<60} {:>12} {:>10}"
    print(col.format("Layer", "Shape", "|W| L2"))
    print("  " + "-" * 86)
    for pname, p in params:
        shape_str = "x".join(str(d) for d in p.shape)
        print(col.format(pname, shape_str, f"{p.data.norm(2).item():.4f}"))


def lr_recommendation(components: dict):
    """
    Simple heuristic: rank components by mean norm.
    Higher norm → stronger prior → suggest lower relative LR.
    """
    print("\n" + "=" * 60)
    print("LR SCALING RECOMMENDATION (relative to base LR)")
    print("=" * 60)
    print("Higher norm = stronger prior = use lower LR multiplier.\n")

    norms = {}
    for name, module in components.items():
        params = list(module.parameters())
        if params:
            norms[name] = layer_stats(params)["mean_norm"]

    max_norm = max(norms.values())
    col = "  {:<35} {:>10} {:>14}"
    print(col.format("Component", "Mean|W|", "Suggested mult"))
    print("  " + "-" * 62)
    for name in sorted(norms, key=norms.get, reverse=True):
        n = norms[name]
        # Simple inverse scaling, clipped to [0.1, 1.0]
        mult = max(0.1, min(1.0, max_norm / (n + 1e-8)))
        print(col.format(name, f"{n:.4f}", f"{mult:.3f}x"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true", help="Print per-layer breakdown for text_encoder and predictor")
    args = parser.parse_args()

    print(f"Loading model from {CHECKPOINT_PATH} ...")
    KModel = load_local_kmodel()
    model = KModel(config=str(CONFIG_PATH), model=str(CHECKPOINT_PATH))
    model.eval()

    components = {
        "bert": model.bert,
        "bert_encoder": model.bert_encoder,
        "text_encoder": model.text_encoder,
        "predictor": model.predictor,
        "predictor.text_encoder": model.predictor.text_encoder,
        "predictor.lstm": model.predictor.lstm,
        "predictor.duration_proj": model.predictor.duration_proj,
        "predictor.shared": model.predictor.shared,
        "predictor.F0": model.predictor.F0,
        "predictor.N": model.predictor.N,
        "decoder": model.decoder,
    }

    print_component_table(components, model)
    lr_recommendation(components)

    if args.verbose:
        for section in ["text_encoder", "predictor"]:
            print(f"\n{'=' * 60}")
            print(f"LAYER BREAKDOWN: {section}")
            print("=" * 60)
            print_layer_breakdown(section, getattr(model, section))


if __name__ == "__main__":
    main()
