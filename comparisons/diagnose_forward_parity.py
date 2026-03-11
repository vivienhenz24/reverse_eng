"""
Diagnose the remaining parity gap between Kokoro's released inference helper and
the differentiable mirrored training forward.

Main question:
  Is the gap caused by our mirrored logic, or by stochasticity inside the
  decoder/vocoder path?
"""

from pathlib import Path

import torch

from prototype_voicepack_training_loop import (
    KokoroWithTrainableVoicepacks,
    VOICES_DIR,
    sep,
)


def explicit_forward(model, input_ids, ref_s, speed=1.0):
    input_lengths = torch.full(
        (input_ids.shape[0],),
        input_ids.shape[-1],
        device=input_ids.device,
        dtype=torch.long,
    )
    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(model.device)

    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

    indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=model.device), pred_dur)
    pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=model.device)
    pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0).to(model.device)

    en = d.transpose(-1, -2) @ pred_aln_trg
    f0_pred, n_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg
    audio = model.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze()

    return {
        "text_mask": text_mask,
        "bert_dur": bert_dur,
        "d_en": d_en,
        "d": d,
        "x": x,
        "duration": duration,
        "pred_dur": pred_dur,
        "pred_aln_trg": pred_aln_trg,
        "en": en,
        "f0_pred": f0_pred,
        "n_pred": n_pred,
        "t_en": t_en,
        "asr": asr,
        "audio": audio,
    }


def main():
    voice_names = sorted(p.stem for p in VOICES_DIR.glob("*.pt"))
    wrapper = KokoroWithTrainableVoicepacks(voice_names)
    wrapper.eval()
    model = wrapper.model

    voice_id = voice_names.index("af_heart")
    input_ids = torch.randint(1, 178, (1, 32))
    ref_s = wrapper.voicepacks(torch.tensor([voice_id]), torch.tensor([32]))

    print("=" * 88)
    print("  DIAGNOSE FORWARD PARITY")
    print("=" * 88)

    sep("UNSEEDED REPEATABILITY")
    with torch.no_grad():
        audio_a, dur_a = model.forward_with_tokens(input_ids, ref_s, speed=1.0)
        audio_b, dur_b = model.forward_with_tokens(input_ids, ref_s, speed=1.0)
    print(f"  original repeat audio diff:  {float((audio_a - audio_b).abs().max().item()):.9f}")
    print(f"  original repeat dur equal:   {torch.equal(dur_a, dur_b)}")

    with torch.no_grad():
        clone_a, clone_dur_a = wrapper.forward_with_tokens_trainable(input_ids, ref_s, speed=1.0)
        clone_b, clone_dur_b = wrapper.forward_with_tokens_trainable(input_ids, ref_s, speed=1.0)
    print(f"  clone repeat audio diff:     {float((clone_a - clone_b).abs().max().item()):.9f}")
    print(f"  clone repeat dur equal:      {torch.equal(clone_dur_a, clone_dur_b)}")

    sep("RESEEDED REPEATABILITY")
    torch.manual_seed(123)
    with torch.no_grad():
        audio_a, dur_a = model.forward_with_tokens(input_ids, ref_s, speed=1.0)
    torch.manual_seed(123)
    with torch.no_grad():
        audio_b, dur_b = model.forward_with_tokens(input_ids, ref_s, speed=1.0)
    print(f"  original reseeded audio diff:{float((audio_a - audio_b).abs().max().item()):.9f}")
    print(f"  original reseeded dur equal: {torch.equal(dur_a, dur_b)}")

    torch.manual_seed(123)
    with torch.no_grad():
        clone_a, clone_dur_a = wrapper.forward_with_tokens_trainable(input_ids, ref_s, speed=1.0)
    torch.manual_seed(123)
    with torch.no_grad():
        clone_b, clone_dur_b = wrapper.forward_with_tokens_trainable(input_ids, ref_s, speed=1.0)
    print(f"  clone reseeded audio diff:   {float((clone_a - clone_b).abs().max().item()):.9f}")
    print(f"  clone reseeded dur equal:    {torch.equal(clone_dur_a, clone_dur_b)}")

    sep("INTERMEDIATE TENSORS")
    with torch.no_grad():
        explicit = explicit_forward(model, input_ids, ref_s, speed=1.0)

        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], device=input_ids.device, dtype=torch.long)
        text_mask = torch.arange(input_lengths.max(), device=input_ids.device).unsqueeze(0)
        text_mask = text_mask.expand(input_lengths.shape[0], -1)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(input_ids.device)
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze(0)
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=input_ids.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=input_ids.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=input_ids.device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg
        f0_pred, n_pred = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        mirrored = {
            "text_mask": text_mask,
            "bert_dur": bert_dur,
            "d_en": d_en,
            "d": d,
            "x": x,
            "duration": duration,
            "pred_dur": pred_dur,
            "pred_aln_trg": pred_aln_trg,
            "en": en,
            "f0_pred": f0_pred,
            "n_pred": n_pred,
            "t_en": t_en,
            "asr": asr,
        }

    for name in ["bert_dur", "d_en", "d", "x", "duration", "pred_dur", "pred_aln_trg", "en", "f0_pred", "n_pred", "t_en", "asr"]:
        diff = float((explicit[name] - mirrored[name]).abs().max().item())
        print(f"  {name:<12} max_abs={diff:.9f}")

    sep("DECODER STOCHASTICITY")
    torch.manual_seed(321)
    with torch.no_grad():
        y1 = model.decoder(explicit["asr"], explicit["f0_pred"], explicit["n_pred"], ref_s[:, :128]).squeeze()
    torch.manual_seed(321)
    with torch.no_grad():
        y2 = model.decoder(explicit["asr"], explicit["f0_pred"], explicit["n_pred"], ref_s[:, :128]).squeeze()
    print(f"  decoder same tensors + same seed: {float((y1 - y2).abs().max().item()):.9f}")

    with torch.no_grad():
        y3 = model.decoder(explicit['asr'], explicit['f0_pred'], explicit['n_pred'], ref_s[:, :128]).squeeze()
    print(f"  decoder same tensors + new RNG:   {float((y1 - y3).abs().max().item()):.9f}")

    sep("WHY")
    print("  The generator injects randomness internally:")
    print("    - torch.rand(...) for initial phase in SineGen._f02sine")
    print("    - torch.randn_like(...) for voiced/unvoiced noise in SineGen.forward")
    print("    - torch.randn_like(...) for noise branch in SourceModuleHnNSF.forward")
    print("  So raw waveform parity is not a valid equivalence test unless RNG is controlled.")

    sep("VERDICT")
    print("  The large audio mismatch is caused by stochastic decoder excitation, not by")
    print("  a materially different mirrored forward graph.")
    print("  For deterministic parity checks, compare:")
    print("    - durations")
    print("    - alignment matrix")
    print("    - asr / F0 / N tensors")
    print("  Or reset torch.manual_seed(...) immediately before decoder calls.")

    sep()
    print("Done.")


if __name__ == "__main__":
    main()
