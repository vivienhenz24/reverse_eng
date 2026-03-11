from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import List

import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import CONFIG_PATH, DEFAULT_CANONICAL_ALIGNMENTS, DEFAULT_TURKISH_MANIFEST, ROOT
else:
    from .common import CONFIG_PATH, DEFAULT_CANONICAL_ALIGNMENTS, DEFAULT_TURKISH_MANIFEST, ROOT


MEL_MEAN = -4.0
MEL_STD = 4.0


def load_vocab() -> dict[str, int]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["vocab"]


def phonemes_to_ids(phonemes: str, vocab: dict[str, int]) -> torch.LongTensor:
    token_ids = [0]
    token_ids.extend(vocab[ch] for ch in phonemes)
    token_ids.append(0)
    return torch.tensor(token_ids, dtype=torch.long)


def build_mel_transform():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        n_mels=80,
    )


def waveform_to_mel(waveform: torch.Tensor, to_mel: torchaudio.transforms.MelSpectrogram) -> torch.Tensor:
    mel = to_mel(waveform)
    mel = (torch.log(1e-5 + mel) - MEL_MEAN) / MEL_STD
    return mel


@dataclass
class TurkishBatch:
    waveforms: torch.FloatTensor
    waveform_lengths: torch.LongTensor
    mel: torch.FloatTensor
    mel_lengths: torch.LongTensor
    input_ids: torch.LongTensor
    input_lengths: torch.LongTensor
    alignments: torch.FloatTensor
    speaker_ids: torch.LongTensor
    phoneme_lengths: torch.LongTensor
    texts: List[str]
    files: List[str]


class TurkishKokoroDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path = DEFAULT_TURKISH_MANIFEST,
        canonical_alignment_dir: Path = DEFAULT_CANONICAL_ALIGNMENTS,
    ):
        self.manifest_path = Path(manifest_path)
        self.canonical_alignment_dir = Path(canonical_alignment_dir)
        self.vocab = load_vocab()
        self.to_mel = build_mel_transform()

        with self.manifest_path.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        wav_path = self.resolve_audio_path(row["file"])
        waveform, sample_rate = sf.read(wav_path)
        if waveform.ndim == 2:
            waveform = waveform[:, 0]
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if sample_rate != 24000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 24000)
        mel = waveform_to_mel(waveform, self.to_mel)

        input_ids = phonemes_to_ids(row["phonemes"], self.vocab)
        alignment_path = self.canonical_alignment_dir / Path(row["alignment"]).name
        alignment = torch.load(alignment_path, map_location="cpu", weights_only=True).float()

        expected_rows = input_ids.shape[0]
        if alignment.shape[0] != expected_rows:
            raise ValueError(
                f"Alignment rows {alignment.shape[0]} != token count {expected_rows} for {row['file']}"
            )

        return {
            "waveform": waveform,
            "mel": mel,
            "input_ids": input_ids,
            "alignment": alignment,
            "speaker_id": int(row["speaker_index"]),
            "phoneme_length": int(row["phoneme_length"]),
            "text": row["text"],
            "file": row["file"],
        }

    @staticmethod
    def resolve_audio_path(manifest_path: str) -> Path:
        direct = ROOT / manifest_path
        if direct.exists():
            return direct
        fallback = ROOT / "combined_dataset" / Path(manifest_path).name
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Could not resolve audio path {manifest_path!r}")


def collate_turkish_batch(items: list[dict]) -> TurkishBatch:
    waveforms = [item["waveform"] for item in items]
    waveform_lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
    waveforms = pad_sequence(waveforms, batch_first=True)

    mels = [item["mel"].transpose(0, 1) for item in items]
    mel_lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    mels = pad_sequence(mels, batch_first=True).transpose(1, 2)

    input_ids = [item["input_ids"] for item in items]
    input_lengths = torch.tensor([x.shape[0] for x in input_ids], dtype=torch.long)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    alignments = [item["alignment"].transpose(0, 1) for item in items]
    max_frames = max(a.shape[0] for a in alignments)
    max_tokens = max(a.shape[1] for a in alignments)
    padded_alignments = torch.zeros((len(items), max_frames, max_tokens), dtype=torch.float32)
    for i, a in enumerate(alignments):
        padded_alignments[i, : a.shape[0], : a.shape[1]] = a
    alignments = padded_alignments.transpose(1, 2)

    speaker_ids = torch.tensor([item["speaker_id"] for item in items], dtype=torch.long)
    phoneme_lengths = torch.tensor([item["phoneme_length"] for item in items], dtype=torch.long)

    return TurkishBatch(
        waveforms=waveforms,
        waveform_lengths=waveform_lengths,
        mel=mels,
        mel_lengths=mel_lengths,
        input_ids=input_ids,
        input_lengths=input_lengths,
        alignments=alignments,
        speaker_ids=speaker_ids,
        phoneme_lengths=phoneme_lengths,
        texts=[item["text"] for item in items],
        files=[item["file"] for item in items],
    )
