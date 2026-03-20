"""
generate_turkish_samples.py

Generate Turkish audio samples from a training checkpoint.
Uses espeak-ng for phonemization and the finetuned voicepack table.

Usage:
    # From latest checkpoint in a run dir:
    python kokoro/training/generate_turkish_samples.py \
        --run-dir kokoro/training/runpod_runs/gt_bootstrap_female_speaker_p140_rows0/checkpoints

    # From a specific checkpoint:
    python kokoro/training/generate_turkish_samples.py \
        --checkpoint path/to/checkpoint_step_5000.pt

    # Custom output dir and sentences:
    python kokoro/training/generate_turkish_samples.py \
        --run-dir ... \
        --out-dir my_samples \
        --sentences "Merhaba, nasılsınız?" "Bugün hava çok güzel."
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHECKPOINT_PATH, CONFIG_PATH, load_local_kmodel

# ── Default test sentences ────────────────────────────────────────────────────
# Mix of short/long, different phoneme patterns, include Turkish-specific chars
DEFAULT_SENTENCES = [
    "Merhaba, nasılsınız?",
    "Bugün hava çok güzel.",
    "Teşekkür ederim.",
    "Hesabınızı açmamı ister misiniz?",
    "Ödeme özetinde görünen tutarı söyleyebilir misiniz?",
    "Bağlantıyı hemen sağlayayım.",
    "Son olarak, bu görüşmeye ait kayıt numaranız oluşturuldu.",
    "Türkçe konuşmayı öğrenmek istiyorum.",
]


def normalize_turkish_phonemes(ps: str) -> str:
    """Same normalization used during training."""
    ps = " ".join(ps.splitlines())
    ps = ps.replace("\u200d", "")
    ps = ps.replace("ɫ", "l")
    ps = " ".join(ps.split())
    return ps


def phonemize_turkish(text: str) -> str | None:
    """Convert Turkish text to IPA phonemes via espeak-ng."""
    try:
        result = subprocess.run(
            ["espeak-ng", "-v", "tr", "-q", "--ipa"],
            input=text,
            capture_output=True,
            text=True,
            check=True,
        )
        ps = normalize_turkish_phonemes(result.stdout.strip())
        return ps if ps else None
    except FileNotFoundError:
        print("ERROR: espeak-ng not found. Install with: apt-get install espeak-ng")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"espeak-ng error for '{text}': {e.stderr}")
        return None


def load_vocab() -> dict[str, int]:
    import json
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["vocab"]


def phonemes_to_ids(ps: str, vocab: dict[str, int]) -> torch.LongTensor | None:
    """Convert phoneme string to token ids, skipping any out-of-vocab chars."""
    ids = [0]
    skipped = []
    for ch in ps:
        if ch in vocab:
            ids.append(vocab[ch])
        else:
            skipped.append(repr(ch))
    ids.append(0)
    if skipped:
        print(f"  Warning: skipped {len(skipped)} out-of-vocab chars: {', '.join(set(skipped))}")
    if len(ids) <= 2:
        return None
    return torch.tensor(ids, dtype=torch.long)


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("checkpoint_step_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(re.search(r"checkpoint_step_(\d+)\.pt", p.name).group(1)))


def load_checkpoint(checkpoint_path: Path, model, device: torch.device):
    """Load model weights and voicepack table from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    voicepack_table = ckpt["voicepack_table"].to(device).float()  # [num_voices, 510, 256]
    step = int(ckpt["step"])
    train_config = ckpt.get("train_config", "unknown")
    return voicepack_table, step, train_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="Path to a specific checkpoint .pt file")
    src.add_argument("--run-dir", type=Path, help="Run dir — will use the latest checkpoint found inside")

    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for wav files (default: next to checkpoint)")
    parser.add_argument("--sentences", nargs="+", default=DEFAULT_SENTENCES,
                        help="Turkish sentences to synthesize")
    parser.add_argument("--voice-index", type=int, default=0,
                        help="Which row of the voicepack table to use (0=first speaker, 1=second)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--also-baseline", action="store_true",
                        help="Also generate from the unmodified pretrained model for comparison")
    args = parser.parse_args()

    # Resolve checkpoint
    if args.run_dir:
        ckpt_path = find_latest_checkpoint(args.run_dir)
        if ckpt_path is None:
            print(f"No checkpoints found in {args.run_dir}")
            sys.exit(1)
    else:
        ckpt_path = args.checkpoint

    # Output dir
    out_dir = args.out_dir or ckpt_path.parent / f"samples_step_{re.search(r'checkpoint_step_(\d+)', ckpt_path.name).group(1)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    vocab = load_vocab()

    print(f"Loading model ...")
    KModel = load_local_kmodel()
    model = KModel(config=str(CONFIG_PATH), model=str(CHECKPOINT_PATH)).to(device).eval()

    print(f"Loading checkpoint: {ckpt_path}")
    voicepack_table, step, train_config = load_checkpoint(ckpt_path, model, device)
    print(f"  step={step}  train_config={train_config}  voicepack_table={list(voicepack_table.shape)}")

    # Baseline model uses the pretrained weights — reload them
    if args.also_baseline:
        baseline_model = KModel(config=str(CONFIG_PATH), model=str(CHECKPOINT_PATH)).to(device).eval()
        from common import mean_released_pack
        baseline_pack = mean_released_pack().to(device).float()  # [510, 256]

    print(f"\nGenerating {len(args.sentences)} samples → {out_dir}\n")
    col = "  {:<5} {:<50} {:<6} {}"
    print(col.format("idx", "text", "ps_len", "phonemes"))
    print("  " + "-" * 90)

    for i, text in enumerate(args.sentences):
        ps = phonemize_turkish(text)
        if ps is None:
            print(f"  [{i:>2}] SKIP (phonemization failed): {text}")
            continue

        input_ids = phonemes_to_ids(ps, vocab)
        if input_ids is None:
            print(f"  [{i:>2}] SKIP (empty after vocab filter): {text}")
            continue

        ps_len = len(input_ids) - 2  # exclude BOS/EOS
        print(col.format(f"[{i}]", text[:48], ps_len, ps[:60] + ("…" if len(ps) > 60 else "")))

        # Finetuned inference
        ref_s = voicepack_table[args.voice_index, min(ps_len - 1, 509)]  # [256]
        with torch.no_grad():
            output = model(ps, ref_s, args.speed)

        audio = output.audio if hasattr(output, "audio") else output
        if audio is None:
            print(f"    WARNING: model returned no audio for sentence {i}")
            continue

        wav = audio.squeeze().cpu().float().numpy()
        slug = re.sub(r"[^\w]", "_", text[:40]).strip("_")
        out_path = out_dir / f"{i:02d}_{slug}_finetuned.wav"
        sf.write(out_path, wav, 24000)

        # Baseline comparison
        if args.also_baseline:
            ref_s_base = baseline_pack[min(ps_len - 1, 509)]
            with torch.no_grad():
                output_base = baseline_model(ps, ref_s_base, args.speed)
            wav_base = (output_base.audio if hasattr(output_base, "audio") else output_base).squeeze().cpu().float().numpy()
            out_path_base = out_dir / f"{i:02d}_{slug}_baseline.wav"
            sf.write(out_path_base, wav_base, 24000)

    n = len(list(out_dir.glob("*_finetuned.wav")))
    print(f"\nSaved {n} finetuned wav files to {out_dir}")
    if args.also_baseline:
        print(f"Saved {n} baseline wav files alongside for comparison")


if __name__ == "__main__":
    main()
