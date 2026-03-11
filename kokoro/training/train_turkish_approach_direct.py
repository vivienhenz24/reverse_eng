from __future__ import annotations

from train_kokoro_turkish import main


if __name__ == "__main__":
    main(
        defaults={
            "train_config": "voicepack_predictor_text",
            "voicepack_init": "mean",
            "lr": 2e-5,
            "grad_clip": 0.5,
            "batch_size": 2,
            "max_steps": 12000,
            "save_every": 50,
        },
        description="Approach 1: direct Turkish voicepack-table adaptation with decoder frozen",
    )
