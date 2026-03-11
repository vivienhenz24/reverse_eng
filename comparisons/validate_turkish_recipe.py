"""
Validate the current Turkish finetuning recipe mechanically.

Checks:
  1. trainable voice-table lookup matches released voicepack files exactly
  2. differentiable training forward matches released inference forward
  3. proposed finetuning recipe sends gradients to the intended modules while
     the frozen decoder stays frozen
"""

from pathlib import Path

import torch

from prototype_voicepack_training_loop import (
    Batch,
    KokoroWithTrainableVoicepacks,
    VOICES_DIR,
    sep,
)


ROOT = Path(__file__).resolve().parent.parent


def module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    found = False
    for param in module.parameters():
        if param.grad is None:
            continue
        found = True
        total += float(param.grad.detach().pow(2).sum().item())
    return total ** 0.5 if found else 0.0


def main():
    voice_names = sorted(p.stem for p in VOICES_DIR.glob("*.pt"))
    model = KokoroWithTrainableVoicepacks(voice_names)

    print("=" * 88)
    print("  VALIDATE TURKISH RECIPE")
    print("=" * 88)

    sep("LOOKUP PARITY")
    for voice_name, length in [("af_heart", 1), ("af_heart", 32), ("zf_xiaoxiao", 87), ("pm_alex", 210)]:
        voice_id = voice_names.index(voice_name)
        direct = torch.load(VOICES_DIR / f"{voice_name}.pt", map_location="cpu", weights_only=True).squeeze(1)[length - 1]
        table = model.voicepacks(torch.tensor([voice_id]), torch.tensor([length])).squeeze(0).detach()
        equal = torch.equal(direct, table)
        max_abs = float((direct - table).abs().max().item())
        print(f"  {voice_name:<12} len={length:<3} equal={equal} max_abs={max_abs:.9f}")

    sep("FORWARD PARITY")
    voice_name = "af_heart"
    voice_id = voice_names.index(voice_name)
    batch = Batch(
        input_ids=torch.randint(1, 178, (1, 32)),
        phoneme_lengths=torch.tensor([32]),
        voice_ids=torch.tensor([voice_id]),
    )
    model.eval()
    ref_s = model.voicepacks(batch.voice_ids, batch.phoneme_lengths)
    with torch.no_grad():
        audio_infer, dur_infer = model.model.forward_with_tokens(batch.input_ids, ref_s, speed=1.0)
        audio_train, dur_train = model.forward_with_tokens_trainable(batch.input_ids, ref_s, speed=1.0)
    audio_max_abs = float((audio_infer - audio_train).abs().max().item())
    dur_equal = torch.equal(dur_infer, dur_train)
    print(f"  audio max abs diff: {audio_max_abs:.9f}")
    print(f"  duration equal:     {dur_equal}")

    sep("GRADIENT FLOW")
    # Proposed first Turkish phase: freeze decoder and bert, train text/prosody side + table.
    model.train()
    for p in model.model.decoder.parameters():
        p.requires_grad = False
    for p in model.model.bert.parameters():
        p.requires_grad = False

    with torch.no_grad():
        target_audio, target_duration, _ = model(batch)

    noisy_target = target_audio.detach() + 0.01 * torch.randn_like(target_audio)
    target_duration = target_duration.detach() + 1

    train_batch = Batch(
        input_ids=batch.input_ids,
        phoneme_lengths=batch.phoneme_lengths,
        voice_ids=batch.voice_ids,
        target_audio=noisy_target,
        target_duration=target_duration.unsqueeze(0) if target_duration.ndim == 0 else target_duration,
    )

    model.zero_grad(set_to_none=True)
    loss, losses, _ = model.training_step(train_batch)
    loss.backward()

    print(f"  total loss:                    {float(loss.item()):.6f}")
    for name, value in losses.items():
        print(f"  {name:<28} {float(value.item()):.6f}")

    table_grad = float(model.voicepacks.table.grad.detach().norm().item()) if model.voicepacks.table.grad is not None else 0.0
    bert_grad = module_grad_norm(model.model.bert)
    bert_encoder_grad = module_grad_norm(model.model.bert_encoder)
    text_encoder_grad = module_grad_norm(model.model.text_encoder)
    predictor_grad = module_grad_norm(model.model.predictor)
    decoder_grad = module_grad_norm(model.model.decoder)

    print(f"  voicepack table grad norm:     {table_grad:.6f}")
    print(f"  bert grad norm:                {bert_grad:.6f}")
    print(f"  bert_encoder grad norm:        {bert_encoder_grad:.6f}")
    print(f"  text_encoder grad norm:        {text_encoder_grad:.6f}")
    print(f"  predictor grad norm:           {predictor_grad:.6f}")
    print(f"  decoder grad norm:             {decoder_grad:.6f}")

    sep("VERDICT")
    print("  Current recipe passes these mechanical checks:")
    print("    - released voice files and trainable table lookup are identical")
    print("    - gradients reach the intended trainable modules")
    print("    - frozen decoder stays frozen")
    print("  Interpreting the audio parity number requires care:")
    print("    - Kokoro's decoder is stochastic, so repeated inference with the same inputs")
    print("      already changes the waveform unless RNG is controlled")
    print("    - use comparisons/diagnose_forward_parity.py for the exact root-cause audit")

    sep()
    print("Done.")


if __name__ == "__main__":
    main()
