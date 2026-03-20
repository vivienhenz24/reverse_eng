"""
analyze_activation_stats.py

Analysis 3: Activation statistics on Turkish samples.

Runs the unmodified, pretrained Kokoro model on Turkish audio — with
no finetuning — and compares predicted F0/duration/norm against the
ground truth extracted from the same audio.

This tells us how much the model "already works" for Turkish out of
the box.  Strong correlation means the pretrained weights generalize
well and we need less finetuning.  Poor correlation identifies which
prediction heads need the most work.

Metrics reported per sample and averaged:
  - F0:       MAE (Hz), Pearson correlation, voiced-frame coverage
  - Duration: MAE (frames/phoneme), Pearson correlation
  - Norm:     MAE, Pearson correlation

Usage:
    python kokoro/training/analyze_activation_stats.py
    python kokoro/training/analyze_activation_stats.py --n-samples 20 --device cpu
    python kokoro/training/analyze_activation_stats.py --voicepack mean
    python kokoro/training/analyze_activation_stats.py --voicepack voice:af_heart
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    DEFAULT_CANONICAL_ALIGNMENTS,
    DEFAULT_TURKISH_MANIFEST,
    init_voicepacks,
    load_styletts2_symbols_and_losses,
)
from dataset import TurkishKokoroDataset, collate_turkish_batch
from train_kokoro_turkish import KokoroTurkishTrainer


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom < 1e-8:
        return float("nan")
    return (a @ b / denom).item()


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.l1_loss(a.float().flatten(), b.float().flatten()).item()


def voiced_coverage(f0: torch.Tensor, threshold: float = 0.01) -> float:
    """Fraction of frames where predicted F0 > threshold (voiced)."""
    return (f0.flatten() > threshold).float().mean().item()


def duration_from_alignment(alignment: torch.Tensor) -> torch.Tensor:
    """Sum along frames axis to get per-phoneme duration."""
    # alignment: [tokens, frames]
    return alignment.sum(dim=-1).float()


def print_sample_row(idx: int, text: str, metrics: dict):
    text_short = text[:40].replace("\n", " ")
    print(f"  [{idx:>3}] {text_short:<40}  "
          f"f0_mae={metrics['f0_mae']:6.3f}  f0_r={metrics['f0_r']:+.3f}  "
          f"dur_mae={metrics['dur_mae']:5.2f}  dur_r={metrics['dur_r']:+.3f}  "
          f"norm_mae={metrics['norm_mae']:5.3f}  norm_r={metrics['norm_r']:+.3f}  "
          f"voiced={metrics['voiced_pct']:.1%}")


def average_metrics(all_metrics: list[dict]) -> dict:
    keys = [k for k in all_metrics[0] if k != "idx"]
    out = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if not math.isnan(m[k])]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=8, help="Number of Turkish samples to evaluate (default: 8)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--voicepack", default="mean",
                        help="Voicepack init: mean | voice:<name> | auto_gender (default: mean)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    _, load_f0_models, log_norm = load_styletts2_symbols_and_losses()

    print(f"Loading model ...")
    trainer = KokoroTurkishTrainer(num_voices=2)
    trainer.to(device)
    init_voicepacks(trainer, args.voicepack)
    trainer.eval()

    print(f"Loading F0 extractor ...")
    from common import DEFAULT_F0_PATH
    f0_model = load_f0_models(str(DEFAULT_F0_PATH)).to(device).eval()

    print(f"Loading dataset ...")
    ds = TurkishKokoroDataset(DEFAULT_TURKISH_MANIFEST, DEFAULT_CANONICAL_ALIGNMENTS)
    indices = list(range(min(args.n_samples, len(ds))))
    ds_sub = Subset(ds, indices)
    # Process one sample at a time to avoid length-padding issues in metrics
    loader = DataLoader(ds_sub, batch_size=1, collate_fn=collate_turkish_batch)

    print(f"\nRunning {len(indices)} samples through unmodified pretrained model (voicepack={args.voicepack}) ...")
    print(f"{'=' * 120}")
    print(f"  {'idx':>3}  {'text':40}  {'f0_mae':>8} {'f0_r':>7} {'dur_mae':>8} {'dur_r':>7} "
          f"{'norm_mae':>9} {'norm_r':>7} {'voiced':>7}")
    print(f"  {'-' * 116}")

    all_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch_device = type(batch)(
                waveforms=batch.waveforms.to(device),
                waveform_lengths=batch.waveform_lengths.to(device),
                mel=batch.mel.to(device),
                mel_lengths=batch.mel_lengths.to(device),
                input_ids=batch.input_ids.to(device),
                input_lengths=batch.input_lengths.to(device),
                alignments=batch.alignments.to(device),
                speaker_ids=batch.speaker_ids.to(device),
                phoneme_lengths=batch.phoneme_lengths.to(device),
                texts=batch.texts,
                files=batch.files,
            )

            out = trainer.forward_teacher_forced(
                input_ids=batch_device.input_ids,
                input_lengths=batch_device.input_lengths,
                alignments=batch_device.alignments,
                speaker_ids=batch_device.speaker_ids,
                phoneme_lengths=batch_device.phoneme_lengths,
            )

            # Ground truth F0 and norm from mel
            f0_gt, _, _ = f0_model(batch_device.mel.unsqueeze(1))
            n_gt = log_norm(batch_device.mel.unsqueeze(1)).squeeze(1)

            f0_pred = out["f0_pred"]
            n_pred = out["n_pred"]

            # Crop to same length
            t = min(f0_gt.shape[-1], f0_pred.shape[-1])
            f0_gt_c = f0_gt[..., :t]
            f0_pred_c = f0_pred[..., :t]
            t_n = min(n_gt.shape[-1], n_pred.shape[-1])
            n_gt_c = n_gt[..., :t_n]
            n_pred_c = n_pred[..., :t_n]

            # Duration: predicted vs GT from alignment
            gt_dur = duration_from_alignment(batch_device.alignments[0])
            dur_logits = out["duration_logits"][0]  # [seq_len, 1] or [seq_len, max_dur]
            text_len = batch_device.input_lengths[0].item()
            dur_pred = torch.sigmoid(dur_logits[:text_len]).sum(dim=-1)
            gt_dur_trimmed = gt_dur[:text_len]
            # Exclude BOS/EOS (index 0 and last)
            dur_pred_inner = dur_pred[1:text_len - 1]
            gt_dur_inner = gt_dur_trimmed[1:text_len - 1]

            metrics = {
                "idx": i,
                "f0_mae": mae(f0_pred_c, f0_gt_c),
                "f0_r": pearson(f0_pred_c, f0_gt_c),
                "dur_mae": mae(dur_pred_inner, gt_dur_inner),
                "dur_r": pearson(dur_pred_inner, gt_dur_inner),
                "norm_mae": mae(n_pred_c, n_gt_c),
                "norm_r": pearson(n_pred_c, n_gt_c),
                "voiced_pct": voiced_coverage(f0_pred_c),
            }
            all_metrics.append(metrics)
            print_sample_row(indices[i], batch.texts[0], metrics)

    avg = average_metrics(all_metrics)
    print(f"  {'─' * 116}")
    print(f"  {'AVERAGE':>44}  "
          f"f0_mae={avg['f0_mae']:6.3f}  f0_r={avg['f0_r']:+.3f}  "
          f"dur_mae={avg['dur_mae']:5.2f}  dur_r={avg['dur_r']:+.3f}  "
          f"norm_mae={avg['norm_mae']:5.3f}  norm_r={avg['norm_r']:+.3f}  "
          f"voiced={avg['voiced_pct']:.1%}")

    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print("=" * 60)

    def interp_r(name, r, threshold_good=0.5, threshold_ok=0.2):
        if math.isnan(r):
            return f"  {name}: no data"
        if r >= threshold_good:
            return f"  {name} r={r:+.3f}: ✓ already correlates well — model generalizes to Turkish for this head"
        if r >= threshold_ok:
            return f"  {name} r={r:+.3f}: ~ partial correlation — some finetuning needed"
        return f"  {name} r={r:+.3f}: ✗ poor correlation — this head needs significant finetuning"

    print(interp_r("F0  ", avg["f0_r"]))
    print(interp_r("Dur ", avg["dur_r"]))
    print(interp_r("Norm", avg["norm_r"]))

    voiced = avg["voiced_pct"]
    if voiced < 0.3:
        print(f"  Voiced coverage={voiced:.1%}: very low — model may be producing mostly unvoiced output for Turkish")
    elif voiced > 0.9:
        print(f"  Voiced coverage={voiced:.1%}: high — model produces voiced output for most frames")
    else:
        print(f"  Voiced coverage={voiced:.1%}: moderate")


if __name__ == "__main__":
    main()
