"""
analyze_gradient_sensitivity.py

Analysis 2: Gradient sensitivity of decoder vs. predictor on Turkish data.

For each trainable configuration (voicepack_only up to full), runs a
forward + backward pass on a small Turkish batch and reports the L2
gradient norm per model component.

High gradient norm in a component = that component "wants to move"
the most on Turkish data.  Low gradient norm = that component is
already well-suited and changing it may not help (or may hurt via
catastrophic forgetting).

Interpretation guide:
  - If decoder grad >> predictor grad: unfreezing decoder is high-risk
    because it's responding strongly — likely to overfit or destabilize.
  - If predictor grad >> decoder grad: the text/prosody side is the
    bottleneck; focus training there.
  - Roughly equal: safe to train both.

Usage:
    python kokoro/training/analyze_gradient_sensitivity.py
    python kokoro/training/analyze_gradient_sensitivity.py --n-samples 8 --device cpu
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    DEFAULT_CANONICAL_ALIGNMENTS,
    DEFAULT_TURKISH_MANIFEST,
    init_voicepacks,
)
from dataset import TurkishKokoroDataset, collate_turkish_batch
from losses import TurkishKokoroLosses
from train_kokoro_turkish import KokoroTurkishTrainer


def component_grad_norm(module: torch.nn.Module) -> float:
    """Combined L2 norm of all gradients in a module."""
    sq_sum = 0.0
    count = 0
    for p in module.parameters():
        if p.grad is not None:
            sq_sum += p.grad.detach().norm(2).item() ** 2
            count += 1
    if count == 0:
        return float("nan")
    return math.sqrt(sq_sum)


CONFIGS = [
    "voicepack_only",
    "voicepack_predictor",
    "voicepack_predictor_text",
    "voicepack_predictor_text_bertenc",
    "voicepack_predictor_text_decoder",
    "voicepack_predictor_text_bertenc_decoder",
]


def run_config(config: str, batch, device: torch.device, disable_f0_loss: bool) -> dict:
    trainer = KokoroTurkishTrainer(num_voices=2)
    trainer.to(device)
    init_voicepacks(trainer, "mean")
    trainer.set_trainable_configuration(config)
    trainer.train()

    loss_fn = TurkishKokoroLosses(device=device, disable_f0_loss=disable_f0_loss)

    out = trainer.forward_teacher_forced(
        input_ids=batch.input_ids.to(device),
        input_lengths=batch.input_lengths.to(device),
        alignments=batch.alignments.to(device),
        speaker_ids=batch.speaker_ids.to(device),
        phoneme_lengths=batch.phoneme_lengths.to(device),
    )

    loss = loss_fn.compute(
        pred_audio=out["audio"],
        target_audio=batch.waveforms.to(device),
        pred_duration_logits=out["duration_logits"],
        gt_alignments=batch.alignments.to(device),
        input_lengths=batch.input_lengths.to(device),
        target_mel=batch.mel.to(device),
        f0_pred=out["f0_pred"],
        n_pred=out["n_pred"],
        voicepack_table=trainer.voicepacks.table,
    )

    loss.total.backward()

    return {
        "total_loss": loss.total.item(),
        "stft": loss.stft.item(),
        "duration": loss.duration.item(),
        "voicepacks": component_grad_norm(trainer.voicepacks),
        "bert": component_grad_norm(trainer.model.bert),
        "bert_encoder": component_grad_norm(trainer.model.bert_encoder),
        "text_encoder": component_grad_norm(trainer.model.text_encoder),
        "predictor": component_grad_norm(trainer.model.predictor),
        "decoder": component_grad_norm(trainer.model.decoder),
    }


def print_results(results: list[tuple[str, dict]]):
    # Loss table
    print("\n" + "=" * 70)
    print("LOSSES PER CONFIG")
    print("=" * 70)
    col = "{:<45} {:>10} {:>10} {:>10}"
    print(col.format("Config", "Total", "STFT", "Duration"))
    print("-" * 70)
    for config, r in results:
        print(col.format(config, f"{r['total_loss']:.4f}", f"{r['stft']:.4f}", f"{r['duration']:.4f}"))

    # Gradient norm table
    print("\n" + "=" * 110)
    print("GRADIENT NORMS PER COMPONENT (L2, combined across all parameters in component)")
    print("=" * 110)
    col = "{:<45} {:>10} {:>10} {:>12} {:>12} {:>12} {:>12}"
    print(col.format("Config", "voicepacks", "bert_enc", "text_enc", "predictor", "decoder", "bert"))
    print("-" * 110)
    for config, r in results:
        def fmt(v):
            return f"{v:.4f}" if not math.isnan(v) else "  frozen"
        print(col.format(
            config,
            fmt(r["voicepacks"]),
            fmt(r["bert_encoder"]),
            fmt(r["text_encoder"]),
            fmt(r["predictor"]),
            fmt(r["decoder"]),
            fmt(r["bert"]),
        ))

    # Interpretation: for the "full" config, rank components by grad norm
    full_config = "voicepack_predictor_text_bertenc_decoder"
    full = next((r for c, r in results if c == full_config), None)
    if full:
        components = ["voicepacks", "bert_encoder", "text_encoder", "predictor", "decoder", "bert"]
        ranked = sorted(
            [(c, full[c]) for c in components if not math.isnan(full[c])],
            key=lambda x: x[1], reverse=True
        )
        print(f"\n{'=' * 60}")
        print(f"COMPONENT RANKING BY GRAD NORM (full config, highest = most sensitive)")
        print("=" * 60)
        for rank, (comp, norm) in enumerate(ranked, 1):
            bar = "█" * min(40, int(norm / ranked[0][1] * 40))
            print(f"  {rank}. {comp:<20} {norm:>8.4f}  {bar}")

        print("\nINTERPRETATION:")
        top = ranked[0][0]
        bottom = ranked[-1][0]
        print(f"  → '{top}' has the highest gradient response to Turkish data.")
        print(f"  → '{bottom}' has the lowest — likely already well-suited or not in active path.")

        decoder_norm = full.get("decoder", float("nan"))
        predictor_norm = full.get("predictor", float("nan"))
        if not math.isnan(decoder_norm) and not math.isnan(predictor_norm):
            ratio = decoder_norm / (predictor_norm + 1e-8)
            if ratio > 2.0:
                print(f"  → decoder/predictor ratio = {ratio:.2f}x: decoder is high-risk to unfreeze early.")
            elif ratio < 0.5:
                print(f"  → decoder/predictor ratio = {ratio:.2f}x: predictor is the main bottleneck.")
            else:
                print(f"  → decoder/predictor ratio = {ratio:.2f}x: roughly balanced, safe to train both.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=4, help="Number of Turkish samples to use (default: 4)")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--disable-f0-loss", action="store_true", help="Skip F0 loss (faster, no JDC model needed)")
    parser.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS,
                        help="Which training configs to test")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading dataset from {DEFAULT_TURKISH_MANIFEST} ...")
    ds = TurkishKokoroDataset(DEFAULT_TURKISH_MANIFEST, DEFAULT_CANONICAL_ALIGNMENTS)
    ds = Subset(ds, list(range(min(args.n_samples, len(ds)))))
    loader = DataLoader(ds, batch_size=args.n_samples, collate_fn=collate_turkish_batch)
    batch = next(iter(loader))
    print(f"Batch: {len(batch.texts)} samples, max phoneme length {batch.phoneme_lengths.max().item()}")

    results = []
    for config in args.configs:
        print(f"  Running config: {config} ...", end=" ", flush=True)
        r = run_config(config, batch, device, args.disable_f0_loss)
        results.append((config, r))
        print(f"loss={r['total_loss']:.3f}")

    print_results(results)


if __name__ == "__main__":
    main()
