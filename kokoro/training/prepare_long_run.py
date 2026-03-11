from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", type=str, default="female_speaker")
    parser.add_argument("--max-phonemes", type=int, default=80)
    parser.add_argument("--max-rows", type=int, default=12000)
    parser.add_argument("--out-dir", type=Path, default=Path("kokoro/training/long_run"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = args.out_dir / f"{args.speaker}_p{args.max_phonemes}_rows{args.max_rows}.csv"

    cmd = [
        sys.executable,
        str(ROOT / "kokoro/training/build_turkish_subset_manifest.py"),
        "--speaker",
        args.speaker,
        "--max-phonemes",
        str(args.max_phonemes),
        "--out",
        str(manifest_out),
    ]
    if args.max_rows > 0:
        cmd.extend(["--max-rows", str(args.max_rows)])
    subprocess.run(cmd, check=True)

    run_sh = args.out_dir / f"run_{args.speaker}.sh"
    run_sh.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "DEVICE=${DEVICE:-mps}",
                "export PYTHONUNBUFFERED=1",
                "python - <<'PY'",
                "import torch, os",
                "device = os.environ.get('DEVICE', 'mps')",
                "if device == 'mps' and not torch.backends.mps.is_available():",
                "    print('MPS unavailable, falling back to cpu')",
                "PY",
                f"python kokoro/training/train_kokoro_turkish.py \\",
                f"  --manifest {manifest_out} \\",
                "  --device ${DEVICE} \\",
                "  --batch-size 2 \\",
                "  --max-steps 2000 \\",
                f"  --max-samples {args.max_rows if args.max_rows > 0 else 0} \\",
                "  --lr 2e-5 \\",
                "  --grad-clip 0.5 \\",
                "  --train-config voicepack_predictor_text \\",
                "  --voicepack-init mean \\",
                f"  --save-dir {args.out_dir / 'checkpoints'} \\",
                "  --save-every 50",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_sh.chmod(0o755)

    print(f"manifest={manifest_out}")
    print(f"run_script={run_sh}")


if __name__ == "__main__":
    main()
