from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import soundfile as sf
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import DEFAULT_TURKISH_MANIFEST
    from compare_turkish_audio_variants import (
        RenderContext,
        build_context,
        init_voicepacks,
        mean_released_pack,
        named_pack,
        run_teacher_forced,
        short_overfit,
    )
    from dataset import TurkishKokoroDataset
    from losses import TurkishKokoroLosses
    from train_kokoro_turkish import KokoroTurkishTrainer
else:
    from .common import DEFAULT_TURKISH_MANIFEST
    from .compare_turkish_audio_variants import (
        RenderContext,
        build_context,
        init_voicepacks,
        mean_released_pack,
        named_pack,
        run_teacher_forced,
        short_overfit,
    )
    from .dataset import TurkishKokoroDataset
    from .losses import TurkishKokoroLosses
    from .train_kokoro_turkish import KokoroTurkishTrainer


@dataclass(frozen=True)
class SampleSpec:
    index: int
    speaker: str
    phoneme_length: int
    file: str
    text: str


def save_wave(path: Path, wav: torch.Tensor, sr: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav.detach().cpu().float().squeeze().numpy(), sr)


def slugify(text: str) -> str:
    keep = []
    for ch in text.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_"}:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:48] or "sample"


def load_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_default_samples(manifest_path: Path) -> list[SampleSpec]:
    rows = load_rows(manifest_path)
    by_speaker: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for idx, row in enumerate(rows):
        by_speaker.setdefault(row["speaker_id"], []).append((idx, row))

    specs: list[SampleSpec] = []
    for speaker, items in sorted(by_speaker.items()):
        ordered = sorted(items, key=lambda pair: int(pair[1]["phoneme_length"]))
        for offset in (100, len(ordered) // 2, len(ordered) - 100):
            idx, row = ordered[offset]
            specs.append(
                SampleSpec(
                    index=idx,
                    speaker=speaker,
                    phoneme_length=int(row["phoneme_length"]),
                    file=row["file"],
                    text=row["text"],
                )
            )
    return specs


def speaker_voice_init(speaker: str) -> str:
    return "voice:pm_alex" if speaker.startswith("male") else "voice:af_heart"


def render_variant(
    trainer: KokoroTurkishTrainer,
    losses: TurkishKokoroLosses,
    ctx: RenderContext,
    name: str,
    init_mode: str,
    overfit_steps: int,
) -> tuple[torch.Tensor, list[str]]:
    init_voicepacks(trainer, init_mode)
    if overfit_steps > 0:
        audio, logs = short_overfit(trainer, ctx, losses, overfit_steps)
        return audio, logs
    audio, metrics = run_teacher_forced(trainer, ctx, losses, use_gt_fn=False)
    return audio, [f"metrics={metrics}"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TURKISH_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=Path("kokoro/training/listening_suite"))
    parser.add_argument("--overfit-steps", type=int, default=25)
    parser.add_argument("--sample-index", type=int, action="append", default=[])
    parser.add_argument("--include-overfit", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    ds = TurkishKokoroDataset(manifest_path=args.manifest)
    losses = TurkishKokoroLosses(device=device, disable_f0_loss=False)
    rows = load_rows(args.manifest)

    if args.sample_index:
        sample_specs = [
            SampleSpec(
                index=idx,
                speaker=rows[idx]["speaker_id"],
                phoneme_length=int(rows[idx]["phoneme_length"]),
                file=rows[idx]["file"],
                text=rows[idx]["text"],
            )
            for idx in args.sample_index
        ]
    else:
        sample_specs = pick_default_samples(args.manifest)

    variants = [
        ("mean_predfn", "mean", 0),
        ("speaker_predfn", None, 0),
    ]
    if args.include_overfit and args.overfit_steps > 0:
        variants.extend(
            [
                ("mean_predfn_overfit", "mean", args.overfit_steps),
                ("speaker_predfn_overfit", None, args.overfit_steps),
            ]
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        f"manifest={args.manifest}",
        f"device={device}",
        f"overfit_steps={args.overfit_steps}",
        "",
    ]

    for spec in sample_specs:
        ctx = build_context(ds, spec.index, device)
        sample_dir = args.out_dir / f"{spec.index:05d}_{spec.speaker}_{spec.phoneme_length:03d}_{slugify(spec.text)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_wave(sample_dir / "target.wav", ctx.waveform[0])

        summary_lines.append(f"[sample {spec.index}]")
        summary_lines.append(f"speaker={spec.speaker}")
        summary_lines.append(f"phoneme_length={spec.phoneme_length}")
        summary_lines.append(f"file={spec.file}")
        summary_lines.append(f"text={spec.text}")
        summary_lines.append("")

        for variant_name, init_mode, overfit_steps in variants:
            trainer = KokoroTurkishTrainer(num_voices=2).to(device)
            chosen_init = init_mode or speaker_voice_init(spec.speaker)
            audio, logs = render_variant(trainer, losses, ctx, variant_name, chosen_init, overfit_steps)
            save_wave(sample_dir / f"{variant_name}.wav", audio[0])
            summary_lines.append(f"{variant_name}: init={chosen_init} overfit_steps={overfit_steps}")
            summary_lines.extend(logs)
            summary_lines.append(f"saved={sample_dir / f'{variant_name}.wav'}")
            summary_lines.append("")
            (args.out_dir / "SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    (args.out_dir / "SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"saved listening suite to {args.out_dir}")


if __name__ == "__main__":
    main()
