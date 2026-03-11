from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import DEFAULT_F0_PATH, load_styletts2_symbols_and_losses
else:
    from .common import DEFAULT_F0_PATH, load_styletts2_symbols_and_losses


MultiResolutionSTFTLoss, load_F0_models, log_norm = load_styletts2_symbols_and_losses()


@dataclass
class TurkishLossBundle:
    total: torch.Tensor
    stft: torch.Tensor
    duration: torch.Tensor
    f0: torch.Tensor
    norm: torch.Tensor
    voicepack_smooth: torch.Tensor


class TurkishKokoroLosses:
    def __init__(
        self,
        device: torch.device,
        f0_model_path=DEFAULT_F0_PATH,
        lambda_stft: float = 1.0,
        lambda_dur: float = 1.0,
        lambda_f0: float = 1.0,
        lambda_norm: float = 1.0,
        lambda_voicepack_smooth: float = 1e-4,
        disable_f0_loss: bool = False,
    ):
        self.device = device
        self.lambda_stft = lambda_stft
        self.lambda_dur = lambda_dur
        self.lambda_f0 = lambda_f0
        self.lambda_norm = lambda_norm
        self.lambda_voicepack_smooth = lambda_voicepack_smooth
        self.disable_f0_loss = disable_f0_loss

        self.stft_loss = MultiResolutionSTFTLoss().to(device)
        self.pitch_extractor = None
        if not disable_f0_loss:
            self.pitch_extractor = load_F0_models(str(f0_model_path)).to(device).eval()
            for param in self.pitch_extractor.parameters():
                param.requires_grad = False

    def compute_duration_targets(self, alignments: torch.Tensor, input_lengths: torch.LongTensor) -> torch.Tensor:
        gt_dur = alignments.sum(dim=-1)
        # ignore BOS/EOS durations in the loss by zeroing them here; caller will slice
        return gt_dur

    def extract_ground_truth_conditioning(self, target_mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pitch_extractor is None:
            raise RuntimeError("Ground-truth acoustic conditioning requires pitch extraction to be enabled")
        with torch.no_grad():
            f0_real, _, _ = self.pitch_extractor(target_mel.unsqueeze(1))
            n_real = log_norm(target_mel.unsqueeze(1)).squeeze(1)
        return self.ensure_bt(f0_real), self.ensure_bt(n_real)

    @staticmethod
    def crop_time_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_len = min(a.shape[-1], b.shape[-1])
        return a[..., :target_len].contiguous(), b[..., :target_len].contiguous()

    @staticmethod
    def ensure_bt(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x.unsqueeze(0)
        if x.ndim == 2:
            return x
        return x.reshape(x.shape[0], -1)

    def crop_waveforms(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_len = min(pred_audio.shape[-1], target_audio.shape[-1])
        return pred_audio[..., :target_len].contiguous(), target_audio[..., :target_len].contiguous()

    def compute(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        pred_duration_logits: torch.Tensor,
        gt_alignments: torch.Tensor,
        input_lengths: torch.LongTensor,
        target_mel: torch.Tensor,
        f0_pred: torch.Tensor,
        n_pred: torch.Tensor,
        voicepack_table: Optional[torch.Tensor] = None,
    ) -> TurkishLossBundle:
        pred_audio, target_audio = self.crop_waveforms(pred_audio, target_audio)
        stft = self.stft_loss(pred_audio, target_audio)

        gt_dur = self.compute_duration_targets(gt_alignments, input_lengths)
        dur_losses = []
        for dur_logits_i, dur_gt_i, text_len in zip(pred_duration_logits, gt_dur, input_lengths):
            dur_pred = torch.sigmoid(dur_logits_i[:text_len]).sum(dim=-1)
            dur_losses.append(F.l1_loss(dur_pred[1 : text_len - 1], dur_gt_i[1 : text_len - 1]))
        duration = torch.stack(dur_losses).mean()

        if self.pitch_extractor is None:
            f0 = torch.zeros((), device=self.device)
            norm = torch.zeros((), device=self.device)
        else:
            with torch.no_grad():
                f0_real, _, _ = self.pitch_extractor(target_mel.unsqueeze(1))
                n_real = log_norm(target_mel.unsqueeze(1)).squeeze(1)
            f0_real = self.ensure_bt(f0_real)
            f0_pred = self.ensure_bt(f0_pred)
            n_real = self.ensure_bt(n_real)
            n_pred = self.ensure_bt(n_pred)
            f0_real, f0_pred = self.crop_time_pair(f0_real, f0_pred)
            n_real, n_pred = self.crop_time_pair(n_real, n_pred)
            f0 = F.smooth_l1_loss(f0_real.contiguous(), f0_pred.contiguous()) / 10.0
            norm = F.smooth_l1_loss(n_real.contiguous(), n_pred.contiguous())

        if voicepack_table is None:
            voicepack_smooth = torch.zeros((), device=self.device)
        else:
            voicepack_smooth = (voicepack_table[:, 1:] - voicepack_table[:, :-1]).pow(2).mean()

        total = (
            self.lambda_stft * stft
            + self.lambda_dur * duration
            + self.lambda_f0 * f0
            + self.lambda_norm * norm
            + self.lambda_voicepack_smooth * voicepack_smooth
        )

        return TurkishLossBundle(
            total=total,
            stft=stft,
            duration=duration,
            f0=f0,
            norm=norm,
            voicepack_smooth=voicepack_smooth,
        )
