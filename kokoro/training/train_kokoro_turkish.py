from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
from pathlib import Path
import re
import sys

import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import CHECKPOINT_PATH, CONFIG_PATH, DEFAULT_CANONICAL_ALIGNMENTS, DEFAULT_TURKISH_MANIFEST, init_voicepacks, load_local_kmodel
    from dataset import TurkishKokoroDataset, collate_turkish_batch
    from losses import TurkishKokoroLosses
else:
    from .common import CHECKPOINT_PATH, CONFIG_PATH, DEFAULT_CANONICAL_ALIGNMENTS, DEFAULT_TURKISH_MANIFEST, init_voicepacks, load_local_kmodel
    from .dataset import TurkishKokoroDataset, collate_turkish_batch
    from .losses import TurkishKokoroLosses


KModel = load_local_kmodel()


class TrainableVoicepackTable(nn.Module):
    def __init__(self, num_voices: int, max_length: int = 510, style_dim: int = 256):
        super().__init__()
        self.table = nn.Parameter(torch.empty(num_voices, max_length, style_dim))
        nn.init.normal_(self.table, mean=0.0, std=0.02)

    def forward(self, voice_ids: torch.LongTensor, phoneme_lengths: torch.LongTensor) -> torch.FloatTensor:
        idx = phoneme_lengths.clamp(min=1, max=self.table.shape[1]) - 1
        return self.table[voice_ids, idx]


class KokoroTurkishTrainer(nn.Module):
    def __init__(self, num_voices: int):
        super().__init__()
        self.model = KModel(config=str(CONFIG_PATH), model=str(CHECKPOINT_PATH))
        self.voicepacks = TrainableVoicepackTable(num_voices=num_voices)

    def freeze_stage1(self):
        for p in self.model.decoder.parameters():
            p.requires_grad = False
        for p in self.model.bert.parameters():
            p.requires_grad = False

    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.voicepacks.parameters():
            p.requires_grad = False

    def set_trainable_configuration(self, config_name: str):
        self.freeze_all()

        # The released decoder and PL-BERT stay frozen in all current Turkish
        # adaptation probes. We only vary the text/prosody stack plus voicepacks.
        for p in self.voicepacks.parameters():
            p.requires_grad = True

        if config_name == "voicepack_only":
            return
        if config_name == "voicepack_predictor":
            for p in self.model.predictor.parameters():
                p.requires_grad = True
            return
        if config_name == "voicepack_predictor_text":
            for p in self.model.predictor.parameters():
                p.requires_grad = True
            for p in self.model.text_encoder.parameters():
                p.requires_grad = True
            return
        if config_name == "voicepack_predictor_text_bertenc":
            for p in self.model.predictor.parameters():
                p.requires_grad = True
            for p in self.model.text_encoder.parameters():
                p.requires_grad = True
            for p in self.model.bert_encoder.parameters():
                p.requires_grad = True
            return
        raise ValueError(f"Unknown training config: {config_name}")

    def forward_teacher_forced(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        alignments: torch.FloatTensor,
        speaker_ids: torch.LongTensor,
        phoneme_lengths: torch.LongTensor,
    ):
        ref_s = self.voicepacks(speaker_ids, phoneme_lengths)
        text_mask = torch.arange(input_lengths.max(), device=input_ids.device).unsqueeze(0)
        text_mask = text_mask.expand(input_lengths.shape[0], -1)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(input_ids.device)

        bert_dur = self.model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)
        s_dur = ref_s[:, 128:]
        d = self.model.predictor.text_encoder(d_en, s_dur, input_lengths, text_mask)
        x, _ = self.model.predictor.lstm(d)
        duration_logits = self.model.predictor.duration_proj(x)

        p_en = d.transpose(-1, -2) @ alignments
        f0_pred, n_pred = self.model.predictor.F0Ntrain(p_en, s_dur)

        t_en = self.model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ alignments
        pred_audio = self.model.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze(1)

        return {
            "audio": pred_audio,
            "duration_logits": duration_logits,
            "f0_pred": f0_pred,
            "n_pred": n_pred,
            "ref_s": ref_s,
        }


def make_dataloader(
    manifest_path: Path,
    alignment_dir: Path,
    batch_size: int,
    max_samples: int | None,
    num_workers: int,
    pin_memory: bool,
):
    dataset = TurkishKokoroDataset(manifest_path=manifest_path, canonical_alignment_dir=alignment_dir)
    if max_samples is not None and max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_turkish_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def format_loss(name: str, value: torch.Tensor) -> str:
    return f"{name}={float(value.detach().item()):.6f}"


def save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = waveform.detach().cpu().float().squeeze().numpy()
    sf.write(path, wav, sample_rate)


def save_training_checkpoint(
    save_dir: Path,
    step: int,
    trainer: KokoroTurkishTrainer,
    optimizer: torch.optim.Optimizer,
    args,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "train_config": args.train_config,
        "voicepack_init": args.voicepack_init,
        "model_state_dict": trainer.model.state_dict(),
        "voicepack_table": trainer.voicepacks.table.detach().cpu(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, save_dir / f"checkpoint_step_{step:04d}.pt")
    torch.save(trainer.voicepacks.table.detach().cpu(), save_dir / f"voicepacks_step_{step:04d}.pt")


def load_training_checkpoint(
    checkpoint_path: Path,
    trainer: KokoroTurkishTrainer,
    optimizer: torch.optim.Optimizer,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(ckpt["model_state_dict"], strict=False)
    trainer.voicepacks.table.data.copy_(ckpt["voicepack_table"].to(trainer.voicepacks.table.device))
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt["step"]) + 1


def find_latest_checkpoint(save_dir: Path) -> Path | None:
    if not save_dir.exists():
        return None
    checkpoints = sorted(save_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return None
    return max(
        checkpoints,
        key=lambda p: int(re.search(r"checkpoint_step_(\d+)\.pt$", p.name).group(1)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TURKISH_MANIFEST)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_CANONICAL_ALIGNMENTS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--disable-f0-loss", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--train-config",
        type=str,
        default="voicepack_predictor_text",
        choices=[
            "voicepack_only",
            "voicepack_predictor",
            "voicepack_predictor_text",
            "voicepack_predictor_text_bertenc",
        ],
    )
    parser.add_argument("--voicepack-init", type=str, default="mean")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataloader = make_dataloader(
        args.manifest,
        args.alignment_dir,
        args.batch_size,
        args.max_samples,
        args.num_workers,
        args.pin_memory,
    )
    trainer = KokoroTurkishTrainer(num_voices=2).to(device)
    init_voicepacks(trainer, args.voicepack_init)
    trainer.set_trainable_configuration(args.train_config)
    trainer.train()

    losses = TurkishKokoroLosses(device=device, disable_f0_loss=args.disable_f0_loss)
    params = [p for p in trainer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, foreach=False)

    print("=" * 88)
    print("  TRAIN KOKORO TURKISH")
    print("=" * 88)
    print(f"manifest:             {args.manifest}")
    print(f"alignment dir:        {args.alignment_dir}")
    print(f"device:               {device}")
    print(f"batch size:           {args.batch_size}")
    print(f"max samples:          {args.max_samples}")
    print(f"max steps:            {args.max_steps}")
    print(f"lr:                   {args.lr}")
    print(f"grad clip:            {args.grad_clip}")
    print(f"num workers:          {args.num_workers}")
    print(f"pin memory:           {args.pin_memory}")
    print(f"disable F0 loss:      {args.disable_f0_loss}")
    print(f"save dir:             {args.save_dir}")
    print(f"train config:         {args.train_config}")
    print(f"voicepack init:       {args.voicepack_init}")

    step = 0
    if args.resume and args.save_dir is not None:
        latest = find_latest_checkpoint(args.save_dir)
        if latest is not None:
            step = load_training_checkpoint(latest, trainer, optimizer)
            print(f"resumed from:         {latest}")
            print(f"resume step:          {step}")

    for batch in itertools.cycle(dataloader):
        if step >= args.max_steps:
            break
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

        if not torch.isfinite(bundle.total):
            print(
                f"step={step} non_finite_loss "
                f"total={bundle.total} stft={bundle.stft} dur={bundle.duration} "
                f"f0={bundle.f0} norm={bundle.norm} vp={bundle.voicepack_smooth}"
            )
            break

        optimizer.zero_grad(set_to_none=True)
        bundle.total.backward()
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            if not torch.isfinite(grad_norm):
                print(f"step={step} non_finite_grad_norm={grad_norm}")
                break
        optimizer.step()

        if args.save_dir is not None:
            should_save = step == 0 or step == args.max_steps - 1 or (args.save_every > 0 and step % args.save_every == 0)
            if should_save:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                base = args.save_dir / f"step_{step:04d}"
                save_waveform(base.with_name(base.name + "_pred.wav"), out["audio"][0])
                save_waveform(base.with_name(base.name + "_target.wav"), batch.waveforms[0])
                save_training_checkpoint(args.save_dir, step, trainer, optimizer, args)

        print(
            f"step={step} "
            f"{format_loss('total', bundle.total)} "
            f"{format_loss('stft', bundle.stft)} "
            f"{format_loss('dur', bundle.duration)} "
            f"{format_loss('f0', bundle.f0)} "
            f"{format_loss('norm', bundle.norm)} "
            f"{format_loss('vp', bundle.voicepack_smooth)}"
        )
        step += 1

    if args.save_dir is not None and step > 0:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.model.state_dict(), args.save_dir / "final_model_state_dict.pt")
        torch.save(trainer.voicepacks.table.detach().cpu(), args.save_dir / "final_voicepacks.pt")


@dataclass
class TurkishBatchToDevice:
    waveforms: torch.FloatTensor
    waveform_lengths: torch.LongTensor
    mel: torch.FloatTensor
    mel_lengths: torch.LongTensor
    input_ids: torch.LongTensor
    input_lengths: torch.LongTensor
    alignments: torch.FloatTensor
    speaker_ids: torch.LongTensor
    phoneme_lengths: torch.LongTensor
    texts: list[str]
    files: list[str]

    def __init__(self, batch, device: torch.device):
        self.waveforms = batch.waveforms.to(device)
        self.waveform_lengths = batch.waveform_lengths.to(device)
        self.mel = batch.mel.to(device)
        self.mel_lengths = batch.mel_lengths.to(device)
        self.input_ids = batch.input_ids.to(device)
        self.input_lengths = batch.input_lengths.to(device)
        self.alignments = batch.alignments.to(device)
        self.speaker_ids = batch.speaker_ids.to(device)
        self.phoneme_lengths = batch.phoneme_lengths.to(device)
        self.texts = batch.texts
        self.files = batch.files


if __name__ == "__main__":
    main()
