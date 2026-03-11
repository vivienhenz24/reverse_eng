from __future__ import annotations

from train_kokoro_turkish import main


if __name__ == "__main__":
    main(
        defaults={
            "train_config": "voicepack_predictor_text",
            "voicepack_init": "mean",
            "decoder_f0_source": "gt",
            "decoder_n_source": "gt",
            "lr": 2e-5,
            "grad_clip": 0.5,
            "batch_size": 2,
            "max_steps": 12000,
            "save_every": 50,
        },
        description="Approach 4: teacher-forced acoustic bootstrap with GT F0/N driving the decoder",
    )
