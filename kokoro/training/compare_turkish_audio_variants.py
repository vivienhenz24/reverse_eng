from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import soundfile as sf
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import VOICES_DIR, init_voicepacks, mean_released_pack, named_pack
    from dataset import TurkishKokoroDataset
    from losses import TurkishKokoroLosses
    from train_kokoro_turkish import KokoroTurkishTrainer
else:
    from .common import VOICES_DIR, init_voicepacks, mean_released_pack, named_pack
    from .dataset import TurkishKokoroDataset
    from .losses import TurkishKokoroLosses
    from .train_kokoro_turkish import KokoroTurkishTrainer


@dataclass
class RenderContext:
    waveform: torch.Tensor
    mel: torch.Tensor
    input_ids: torch.Tensor
    alignment: torch.Tensor
    speaker_id: torch.Tensor
    phoneme_length: torch.Tensor
    text: str
    file: str


def save_wave(path: Path, wav: torch.Tensor, sr: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav.detach().cpu().float().squeeze().numpy(), sr)


def build_context(ds: TurkishKokoroDataset, index: int, device: torch.device) -> RenderContext:
    item = ds[index]
    return RenderContext(
        waveform=item["waveform"].to(device).unsqueeze(0),
        mel=item["mel"].to(device).unsqueeze(0),
        input_ids=item["input_ids"].to(device).unsqueeze(0),
        alignment=item["alignment"].to(device).unsqueeze(0),
        speaker_id=torch.tensor([item["speaker_id"]], dtype=torch.long, device=device),
        phoneme_length=torch.tensor([item["phoneme_length"]], dtype=torch.long, device=device),
        text=item["text"],
        file=item["file"],
    )


def run_teacher_forced(
    trainer: KokoroTurkishTrainer,
    ctx: RenderContext,
    losses: TurkishKokoroLosses,
    use_gt_fn: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    trainer.eval()
    with torch.no_grad():
        out = trainer.forward_teacher_forced(
            input_ids=ctx.input_ids,
            input_lengths=torch.tensor([ctx.input_ids.shape[1]], device=ctx.input_ids.device),
            alignments=ctx.alignment,
            speaker_ids=ctx.speaker_id,
            phoneme_lengths=ctx.phoneme_length,
        )
        if use_gt_fn:
            f0_real, _, _ = losses.pitch_extractor(ctx.mel.unsqueeze(1))
            n_real = losses.ensure_bt(losses.log_norm(ctx.mel.unsqueeze(1)).squeeze(1)) if hasattr(losses, "log_norm") else None
            if f0_real.ndim == 3 and f0_real.shape[1] == 1:
                f0_real = f0_real.squeeze(1)
            n_real = torch.log(torch.exp(ctx.mel * 4 - 4).norm(dim=1)).unsqueeze(0) if n_real is None else n_real
            f0_real = losses.ensure_bt(f0_real)
            n_real = losses.ensure_bt(n_real)
            target_len = out["f0_pred"].shape[-1]
            if f0_real.shape[-1] != target_len:
                f0_real = F.interpolate(f0_real.unsqueeze(1), size=target_len, mode="linear", align_corners=False).squeeze(1)
            if n_real.shape[-1] != target_len:
                n_real = F.interpolate(n_real.unsqueeze(1), size=target_len, mode="linear", align_corners=False).squeeze(1)
            asr = trainer.model.text_encoder(
                ctx.input_ids,
                torch.tensor([ctx.input_ids.shape[1]], device=ctx.input_ids.device),
                torch.zeros_like(ctx.input_ids, dtype=torch.bool),
            ) @ ctx.alignment
            audio = trainer.model.decoder(asr, f0_real, n_real, out["ref_s"][:, :128]).squeeze(1)
        else:
            audio = out["audio"]
        metrics = {
            "pred_audio_abs_mean": float(audio.abs().mean().item()),
            "pred_audio_std": float(audio.std().item()),
        }
        return audio, metrics


def short_overfit(
    trainer: KokoroTurkishTrainer,
    ctx: RenderContext,
    losses: TurkishKokoroLosses,
    steps: int,
) -> tuple[torch.Tensor, list[str]]:
    trainer.train()
    trainer.freeze_stage1()
    opt = torch.optim.AdamW([p for p in trainer.parameters() if p.requires_grad], lr=1e-4)
    logs = []
    input_lengths = torch.tensor([ctx.input_ids.shape[1]], device=ctx.input_ids.device)
    for step in range(steps):
        out = trainer.forward_teacher_forced(
            input_ids=ctx.input_ids,
            input_lengths=input_lengths,
            alignments=ctx.alignment,
            speaker_ids=ctx.speaker_id,
            phoneme_lengths=ctx.phoneme_length,
        )
        bundle = losses.compute(
            pred_audio=out["audio"],
            target_audio=ctx.waveform,
            pred_duration_logits=out["duration_logits"],
            gt_alignments=ctx.alignment,
            input_lengths=input_lengths,
            target_mel=ctx.mel,
            f0_pred=out["f0_pred"],
            n_pred=out["n_pred"],
            voicepack_table=trainer.voicepacks.table,
        )
        opt.zero_grad(set_to_none=True)
        bundle.total.backward()
        opt.step()
        logs.append(
            f"step={step} total={float(bundle.total.item()):.6f} stft={float(bundle.stft.item()):.6f} "
            f"dur={float(bundle.duration.item()):.6f} f0={float(bundle.f0.item()):.6f} norm={float(bundle.norm.item()):.6f}"
        )
    trainer.eval()
    with torch.no_grad():
        out = trainer.forward_teacher_forced(
            input_ids=ctx.input_ids,
            input_lengths=input_lengths,
            alignments=ctx.alignment,
            speaker_ids=ctx.speaker_id,
            phoneme_lengths=ctx.phoneme_length,
        )
    return out["audio"], logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("kokoro/training/variant_listen"))
    parser.add_argument("--overfit-steps", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    ds = TurkishKokoroDataset()
    ctx = build_context(ds, args.sample_index, device)
    losses = TurkishKokoroLosses(device=device, disable_f0_loss=False)
    # expose helpers for convenience
    losses.log_norm = staticmethod(lambda x, mean=-4, std=4, dim=2: torch.log(torch.exp(x * std + mean).norm(dim=dim)))

    variants = [
        ("random_predfn", "random", False, False),
        ("random_gtfn", "random", True, False),
        ("mean_predfn", "mean", False, False),
        ("mean_gtfn", "mean", True, False),
        ("afheart_gtfn", "voice:af_heart", True, False),
        ("mean_predfn_overfit", "mean", False, True),
        ("mean_gtfn_overfit", "mean", True, True),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_wave(args.out_dir / "target.wav", ctx.waveform[0])
    summary_lines = [
        f"sample_index={args.sample_index}",
        f"file={ctx.file}",
        f"text={ctx.text}",
        "",
    ]

    for name, init_mode, use_gt_fn, do_overfit in variants:
        trainer = KokoroTurkishTrainer(num_voices=2).to(device)
        init_voicepacks(trainer, init_mode)
        if do_overfit:
            audio, logs = short_overfit(trainer, ctx, losses, args.overfit_steps)
            summary_lines.append(f"[{name}]")
            summary_lines.extend(logs)
        else:
            audio, metrics = run_teacher_forced(trainer, ctx, losses, use_gt_fn=use_gt_fn)
            summary_lines.append(f"[{name}] {metrics}")
        save_wave(args.out_dir / f"{name}.wav", audio[0])
        summary_lines.append(f"saved={args.out_dir / f'{name}.wav'}")
        summary_lines.append("")

    (args.out_dir / "SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"saved variants to {args.out_dir}")


if __name__ == "__main__":
    main()
