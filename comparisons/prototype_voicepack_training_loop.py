"""
Prototype a Kokoro training loop with a trainable voice-length table.

This is the most concrete reverse-engineering step so far:
  - keep Kokoro's released inference backbone unchanged
  - replace missing style/diffusion inference machinery with a trainable table
  - fetch ref_s by (voice_id, phoneme_length)
  - run the exact Kokoro forward path already used at inference

The script is intentionally lightweight. It dry-runs the shapes and shows where
losses would attach; it does not attempt full dataset training on its own.
"""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "kokoro/weights/config.json"
CHECKPOINT_PATH = ROOT / "kokoro/weights/kokoro-v1_0.pth"
VOICES_DIR = ROOT / "kokoro/weights/voices"


def load_local_kmodel():
    pkg_name = "_local_kokoro_pkg"
    pkg_root = ROOT / "kokoro/kokoro"

    if "kokoro" not in sys.modules:
        public_pkg = types.ModuleType("kokoro")
        public_pkg.__path__ = [str(pkg_root)]
        sys.modules["kokoro"] = public_pkg

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_root)]
        sys.modules[pkg_name] = pkg

    for name in ["custom_stft", "istftnet", "modules", "model"]:
        full_name = f"{pkg_name}.{name}"
        if full_name in sys.modules:
            continue
        path = pkg_root / f"{name}.py"
        spec = importlib.util.spec_from_file_location(full_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        sys.modules[f"kokoro.{name}"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return sys.modules[f"{pkg_name}.model"].KModel


KModel = load_local_kmodel()


def sep(title="", width=88, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)


@dataclass
class Batch:
    input_ids: torch.LongTensor      # [B, T]
    phoneme_lengths: torch.LongTensor  # [B]
    voice_ids: torch.LongTensor      # [B]
    target_audio: Optional[torch.FloatTensor] = None
    target_duration: Optional[torch.LongTensor] = None


class TrainableVoicepackTable(nn.Module):
    def __init__(self, num_voices: int, max_length: int = 510, style_dim: int = 256):
        super().__init__()
        self.num_voices = num_voices
        self.max_length = max_length
        self.style_dim = style_dim
        self.table = nn.Parameter(torch.empty(num_voices, max_length, style_dim))
        nn.init.normal_(self.table, mean=0.0, std=0.02)

    @classmethod
    def from_released_voicepacks(cls, voice_names):
        sample = torch.load(VOICES_DIR / f"{voice_names[0]}.pt", map_location="cpu", weights_only=True).squeeze(1)
        mod = cls(num_voices=len(voice_names), max_length=sample.shape[0], style_dim=sample.shape[1])
        with torch.no_grad():
            rows = []
            for name in voice_names:
                rows.append(torch.load(VOICES_DIR / f"{name}.pt", map_location="cpu", weights_only=True).squeeze(1).float())
            mod.table.copy_(torch.stack(rows))
        return mod

    def forward(self, voice_ids: torch.LongTensor, phoneme_lengths: torch.LongTensor) -> torch.FloatTensor:
        idx = phoneme_lengths.clamp(min=1, max=self.max_length) - 1
        return self.table[voice_ids, idx]


class KokoroWithTrainableVoicepacks(nn.Module):
    def __init__(self, voice_names):
        super().__init__()
        self.voice_names = list(voice_names)
        self.model = KModel(config=str(CONFIG_PATH), model=str(CHECKPOINT_PATH))
        self.voicepacks = TrainableVoicepackTable.from_released_voicepacks(self.voice_names)

    def forward_with_tokens_trainable(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float = 1.0):
        model = self.model
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long,
        )
        text_mask = torch.arange(input_lengths.max(), device=input_ids.device).unsqueeze(0)
        text_mask = text_mask.expand(input_lengths.shape[0], -1)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(input_ids.device)

        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze(0)

        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=input_ids.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=input_ids.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=input_ids.device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg
        f0_pred, n_pred = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = model.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    def forward(self, batch: Batch):
        ref_s = self.voicepacks(batch.voice_ids, batch.phoneme_lengths)
        audio, pred_dur = self.forward_with_tokens_trainable(batch.input_ids, ref_s, speed=1.0)
        return audio, pred_dur, ref_s

    def training_step(self, batch: Batch):
        audio, pred_dur, ref_s = self(batch)
        losses = {}

        if batch.target_audio is not None:
            # Placeholder: real training would use multi-resolution STFT / mel / adversarial losses.
            target = batch.target_audio[..., : audio.shape[-1]]
            pred = audio[..., : target.shape[-1]]
            losses["audio_l1"] = F.l1_loss(pred, target)

        if batch.target_duration is not None:
            losses["dur_l1"] = F.l1_loss(pred_dur.float(), batch.target_duration.float())

        # Optional regularizers suggested by the released artifact geometry.
        table = self.voicepacks.table
        losses["smooth_len"] = (table[:, 1:] - table[:, :-1]).pow(2).mean()
        losses["small_norm"] = table.pow(2).mean()

        total = sum(losses.values()) if losses else torch.tensor(0.0, device=ref_s.device)
        return total, losses, ref_s


def main():
    voice_names = sorted(p.stem for p in VOICES_DIR.glob("*.pt"))
    trainer = KokoroWithTrainableVoicepacks(voice_names)
    trainer.eval()

    batch = Batch(
        input_ids=torch.randint(1, 178, (1, 32)),
        phoneme_lengths=torch.tensor([32]),
        voice_ids=torch.tensor([0]),
    )
    with torch.no_grad():
        audio, pred_dur, ref_s = trainer(batch)

    print("=" * 88)
    print("  PROTOTYPE VOICEPACK TRAINING LOOP")
    print("=" * 88)

    sep("TABLE")
    print(f"  voices:       {len(voice_names)}")
    print(f"  shape:        {list(trainer.voicepacks.table.shape)}")
    print(f"  params:       {trainer.voicepacks.table.numel():,}")

    sep("DRY RUN")
    print(f"  input_ids:    {list(batch.input_ids.shape)}")
    print(f"  ref_s:        {list(ref_s.shape)}")
    print(f"  audio:        {list(audio.shape)}")
    print(f"  pred_dur:     {list(pred_dur.shape)}")

    sep("TRAINING INTERPRETATION")
    print("  Candidate loop:")
    print("    1. Batch carries (voice_id, phoneme sequence, target audio)")
    print("    2. Look up ref_s = voicepack_table[voice_id, len(ps)-1]")
    print("    3. Run Kokoro exactly as in released inference")
    print("    4. Backprop audio / duration / auxiliary losses into model + table")
    print("    5. Optionally regularize adjacent length slots for smoothness")

    sep("WHY THIS FITS THE EVIDENCE")
    print("  - It matches the released inference contract exactly.")
    print("  - It matches the public phrase 'trained voicepacks'.")
    print("  - It explains why voicepacks are separate files and directly averageable.")
    print("  - It removes any need to assume a hidden one-shot ref_s exporter.")

    sep()
    print("Done.")


if __name__ == "__main__":
    main()
