from __future__ import annotations

from train_kokoro_turkish import main


if __name__ == "__main__":
    main(
        defaults={
            "train_config": "voicepack_predictor_text_decoder",
            "voicepack_init": "mean",
            "lr": 1e-5,
            "grad_clip": 0.3,
            "batch_size": 1,
            "max_steps": 8000,
            "save_every": 25,
        },
        description="Approach 3: direct Turkish adaptation with decoder unfrozen",
    )
