# Kokoro Turkish Loss Mapping

This note maps StyleTTS2's training losses onto the stripped Kokoro backbone.

## Best Repo Placement

Yes: the actual Turkish training implementation should live under `kokoro/`, not `comparisons/`.

Use `comparisons/` for reverse-engineering evidence and audits.
Use `kokoro/training/` for real training code and training-specific docs.

## Exact StyleTTS2 Loss Anchors

Primary source:

- [train_second.py](/Users/vivienhenz/reverse_eng/StyleTTS2/train_second.py#L383)
- [train_second.py](/Users/vivienhenz/reverse_eng/StyleTTS2/train_second.py#L400)
- [train_second.py](/Users/vivienhenz/reverse_eng/StyleTTS2/train_second.py#L419)
- [train_second.py](/Users/vivienhenz/reverse_eng/StyleTTS2/train_second.py#L427)
- [train_second.py](/Users/vivienhenz/reverse_eng/StyleTTS2/train_second.py#L443)

Observed generator-side losses:

- `loss_mel = stft_loss(y_rec, wav)`
- `loss_F0_rec = smooth_l1(F0_real, F0_fake) / 10`
- `loss_norm_rec = smooth_l1(N_real, N_fake)`
- `loss_dur = l1(pred_duration, gt_duration)`
- `loss_ce = BCEWithLogits(pred_duration_logits, duration_step_targets)`
- optional `loss_gen_all` adversarial loss
- optional `loss_lm` SLM loss
- optional `loss_sty` style reconstruction loss
- optional `loss_diff` diffusion loss

## What Survives For Kokoro

Keep:

- multi-resolution STFT reconstruction loss
- duration L1 loss from the canonicalized Turkish alignments
- F0 smooth L1 loss
- norm/noise smooth L1 loss

Optional:

- duration-frame BCE if we expose a stepwise duration-logit path compatible with the external alignments
- voicepack smoothness regularization
- adversarial / SLM losses later, only if needed

Drop for the first Turkish path:

- diffusion loss
- style reconstruction loss
- style encoder loss
- predictor encoder loss
- text aligner loss

Reason:

Kokoro inference no longer depends on diffusion, style encoder, predictor encoder, or text aligner.
Turkish already has external phonemized text and external alignments.

## Current Best First Objective

For Turkish stage 1:

```text
L =
  lambda_stft * L_stft(y_pred, y_gt)
  + lambda_dur * L1(pred_dur, gt_dur)
  + lambda_f0 * smooth_l1(F0_pred, F0_gt)
  + lambda_norm * smooth_l1(N_pred, N_gt)
  + lambda_vp_smooth * smoothness(voicepack_table)
```

Where:

- `gt_dur` comes from canonicalized `alignments_kokoro_tr/*.pt`
- `F0_gt` comes from the same pitch extractor StyleTTS2 uses
- `N_gt` comes from `log_norm(...)` on target mels

## What Still Needs To Be Implemented

Under `kokoro/training/`:

1. `dataset.py`
- load `combined_dataset/kokoro_turkish_manifest.csv`
- load canonicalized alignments
- map normalized phonemes to token ids

2. `losses.py`
- STFT loss wrapper
- duration loss from canonical alignments
- F0 extraction + loss
- norm target + loss
- voicepack smoothness regularizer

3. `train_kokoro_turkish.py`
- train Turkish voicepack table
- finetune `bert_encoder`, `text_encoder`, `predictor`
- keep `decoder` frozen first

## Best Next Test

Do a tiny overfit run with:

- STFT
- duration
- F0
- norm
- voicepack smoothness

If that overfits cleanly and produces intelligible Turkish, only then consider:

- unfreezing decoder
- adding adversarial loss
- adding SLM loss
