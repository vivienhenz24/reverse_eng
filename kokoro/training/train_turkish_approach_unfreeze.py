from __future__ import annotations

from train_kokoro_turkish import main


if __name__ == "__main__":
    main(
        defaults={
            # Train voicepacks + predictor + text_encoder + bert_encoder; decoder frozen.
            # Uses predicted F0/N (no gt bootstrap) — predictor learns Turkish prosody directly.
            "train_config": "voicepack_predictor_text_bertenc",
            "voicepack_init": "mean",
            "decoder_f0_source": "pred",
            "decoder_n_source": "pred",
            "lr": 1e-5,
            "grad_clip": 0.5,
            "batch_size": 1,
            "max_steps": 12000,
            "save_audio_every": 200,
            "save_checkpoint_every": 500,
        },
        description="Approach: unfreeze voicepacks + predictor + text_encoder + bert_encoder, decoder frozen",
    )
