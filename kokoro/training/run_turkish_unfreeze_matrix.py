from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
import sys

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import DEFAULT_CANONICAL_ALIGNMENTS, DEFAULT_TURKISH_MANIFEST
    from compare_turkish_audio_variants import build_context, init_voicepacks
    from dataset import TurkishKokoroDataset, collate_turkish_batch
    from losses import TurkishKokoroLosses
    from train_kokoro_turkish import KokoroTurkishTrainer, TurkishBatchToDevice
else:
    from .common import DEFAULT_CANONICAL_ALIGNMENTS, DEFAULT_TURKISH_MANIFEST
    from .compare_turkish_audio_variants import build_context, init_voicepacks
    from .dataset import TurkishKokoroDataset, collate_turkish_batch
    from .losses import TurkishKokoroLosses
    from .train_kokoro_turkish import KokoroTurkishTrainer, TurkishBatchToDevice


CONFIGS = [
    "voicepack_only",
    "voicepack_predictor",
    "voicepack_predictor_text",
    "voicepack_predictor_text_bertenc",
]


def save_wave(path: Path, wav: torch.Tensor, sr: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav.detach().cpu().float().squeeze().numpy(), sr)


def pick_eval_indices(manifest_path: Path) -> list[int]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_speaker: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for idx, row in enumerate(rows):
        by_speaker.setdefault(row["speaker_id"], []).append((idx, row))

    out: list[int] = []
    for speaker, items in sorted(by_speaker.items()):
        ordered = sorted(items, key=lambda pair: int(pair[1]["phoneme_length"]))
        for offset in (100, len(ordered) // 2):
            out.append(ordered[offset][0])
    return out


def make_dataloader(manifest_path: Path, alignment_dir: Path, batch_size: int, max_samples: int):
    ds = TurkishKokoroDataset(manifest_path=manifest_path, canonical_alignment_dir=alignment_dir)
    subset = Subset(ds, range(min(max_samples, len(ds))))
    return ds, DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate_turkish_batch)


def render_eval_set(
    trainer: KokoroTurkishTrainer,
    losses: TurkishKokoroLosses,
    ds: TurkishKokoroDataset,
    eval_indices: list[int],
    out_dir: Path,
    step: int,
):
    trainer.eval()
    with torch.no_grad():
        for idx in eval_indices:
            ctx = build_context(ds, idx, next(trainer.parameters()).device)
            out = trainer.forward_teacher_forced(
                input_ids=ctx.input_ids,
                input_lengths=torch.tensor([ctx.input_ids.shape[1]], device=ctx.input_ids.device),
                alignments=ctx.alignment,
                speaker_ids=ctx.speaker_id,
                phoneme_lengths=ctx.phoneme_length,
            )
            sample_dir = out_dir / f"eval_{idx:05d}"
            save_wave(sample_dir / f"step_{step:04d}_pred.wav", out["audio"][0])
            if step == 0 and not (sample_dir / "target.wav").exists():
                save_wave(sample_dir / "target.wav", ctx.waveform[0])
                meta = [
                    f"file={ctx.file}",
                    f"text={ctx.text}",
                    f"phoneme_length={int(ctx.phoneme_length.item())}",
                ]
                (sample_dir / "meta.txt").write_text("\n".join(meta), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TURKISH_MANIFEST)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_CANONICAL_ALIGNMENTS)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=Path, default=Path("kokoro/training/unfreeze_matrix"))
    args = parser.parse_args()

    device = torch.device(args.device)
    ds, dataloader = make_dataloader(args.manifest, args.alignment_dir, args.batch_size, args.max_samples)
    losses = TurkishKokoroLosses(device=device, disable_f0_loss=False)
    eval_indices = pick_eval_indices(args.manifest)

    summary = [
        f"manifest={args.manifest}",
        f"alignment_dir={args.alignment_dir}",
        f"device={device}",
        f"batch_size={args.batch_size}",
        f"max_samples={args.max_samples}",
        f"steps={args.steps}",
        f"eval_every={args.eval_every}",
        f"eval_indices={eval_indices}",
        "",
    ]

    for config_name in CONFIGS:
        run_dir = args.out_dir / config_name
        run_dir.mkdir(parents=True, exist_ok=True)
        trainer = KokoroTurkishTrainer(num_voices=2).to(device)
        init_voicepacks(trainer, "mean")
        trainer.set_trainable_configuration(config_name)
        trainer.train()
        opt = torch.optim.AdamW([p for p in trainer.parameters() if p.requires_grad], lr=args.lr)

        summary.append(f"[{config_name}]")
        render_eval_set(trainer, losses, ds, eval_indices, run_dir, step=0)

        for step, batch in zip(range(1, args.steps + 1), itertools.cycle(dataloader)):
            batch = TurkishBatchToDevice(batch, device)
            out = trainer.forward_teacher_forced(
                input_ids=batch.input_ids,
                input_lengths=batch.input_lengths,
                alignments=batch.alignments,
                speaker_ids=batch.speaker_ids,
                phoneme_lengths=batch.phoneme_lengths,
            )
            bundle = losses.compute(
                pred_audio=out["audio"],
                target_audio=batch.waveforms,
                pred_duration_logits=out["duration_logits"],
                gt_alignments=batch.alignments,
                input_lengths=batch.input_lengths,
                target_mel=batch.mel,
                f0_pred=out["f0_pred"],
                n_pred=out["n_pred"],
                voicepack_table=trainer.voicepacks.table,
            )
            opt.zero_grad(set_to_none=True)
            bundle.total.backward()
            opt.step()

            summary.append(
                f"step={step} total={float(bundle.total.item()):.6f} stft={float(bundle.stft.item()):.6f} "
                f"dur={float(bundle.duration.item()):.6f} f0={float(bundle.f0.item()):.6f} "
                f"norm={float(bundle.norm.item()):.6f}"
            )
            if step % args.eval_every == 0 or step == args.steps:
                render_eval_set(trainer, losses, ds, eval_indices, run_dir, step=step)

        summary.append("")
        (args.out_dir / "SUMMARY.txt").write_text("\n".join(summary), encoding="utf-8")

    (args.out_dir / "SUMMARY.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"saved matrix results to {args.out_dir}")


if __name__ == "__main__":
    main()
