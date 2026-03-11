from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import os
from pathlib import Path
import re
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import DEFAULT_TURKISH_MANIFEST, ROOT
    from compare_turkish_audio_variants import build_context, init_voicepacks
    from dataset import TurkishKokoroDataset
    from train_kokoro_turkish import KokoroTurkishTrainer
else:
    from .common import DEFAULT_TURKISH_MANIFEST, ROOT
    from .compare_turkish_audio_variants import build_context, init_voicepacks
    from .dataset import TurkishKokoroDataset
    from .train_kokoro_turkish import KokoroTurkishTrainer


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zçğıöşü0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pick_indices(manifest_path: Path, per_speaker: int) -> list[int]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_speaker: dict[str, list[tuple[int, int]]] = {}
    for idx, row in enumerate(rows):
        by_speaker.setdefault(row["speaker_id"], []).append((idx, int(row["phoneme_length"])))

    picks: list[int] = []
    for speaker, items in sorted(by_speaker.items()):
        ordered = sorted(items, key=lambda x: x[1])
        step = max(1, len(ordered) // per_speaker)
        chosen = [ordered[min(i * step, len(ordered) - 1)][0] for i in range(per_speaker)]
        picks.extend(chosen)
    return sorted(set(picks))


def load_asr(device: str):
    cache_dir = ROOT / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    from transformers import pipeline

    hf_device = -1 if device == "cpu" else 0
    return pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-tiny",
        device=hf_device,
        model_kwargs={"cache_dir": str(cache_dir / "transformers")},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TURKISH_MANIFEST)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--per-speaker", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=Path("kokoro/training/phoneme_audit"))
    args = parser.parse_args()

    ds = TurkishKokoroDataset(manifest_path=args.manifest)
    trainer = KokoroTurkishTrainer(num_voices=2).to(args.device)
    init_voicepacks(trainer, "mean")
    trainer.set_trainable_configuration("voicepack_only")
    trainer.eval()
    asr = load_asr(args.device)

    indices = pick_indices(args.manifest, args.per_speaker)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, str]] = []

    for idx in indices:
        ctx = build_context(ds, idx, torch.device(args.device))
        with torch.no_grad():
            out = trainer.forward_teacher_forced(
                input_ids=ctx.input_ids,
                input_lengths=torch.tensor([ctx.input_ids.shape[1]], device=ctx.input_ids.device),
                alignments=ctx.alignment,
                speaker_ids=ctx.speaker_id,
                phoneme_lengths=ctx.phoneme_length,
            )
        pred_wav = out["audio"][0].detach().cpu().float().numpy()
        pred_path = args.out_dir / f"{idx:05d}_pred.wav"
        target_path = args.out_dir / f"{idx:05d}_target.wav"
        import soundfile as sf

        sf.write(pred_path, pred_wav, 24000)
        sf.write(target_path, ctx.waveform[0].detach().cpu().float().numpy(), 24000)

        pred_text = asr({"array": pred_wav, "sampling_rate": 24000}, generate_kwargs={"language": "turkish", "task": "transcribe"})["text"]
        target_text = asr(
            {"array": ctx.waveform[0].detach().cpu().float().numpy(), "sampling_rate": 24000},
            generate_kwargs={"language": "turkish", "task": "transcribe"},
        )["text"]

        intended = normalize_text(ctx.text)
        heard = normalize_text(pred_text)
        heard_target = normalize_text(target_text)
        sim = SequenceMatcher(a=intended, b=heard).ratio()
        sim_target = SequenceMatcher(a=intended, b=heard_target).ratio()

        rows_out.append(
            {
                "index": str(idx),
                "file": ctx.file,
                "text": ctx.text,
                "phonemes": "".join(ds.rows[idx]["phonemes"]),
                "pred_asr": pred_text.strip(),
                "target_asr": target_text.strip(),
                "pred_similarity": f"{sim:.4f}",
                "target_similarity": f"{sim_target:.4f}",
                "pred_wav": str(pred_path),
                "target_wav": str(target_path),
            }
        )

    csv_path = args.out_dir / "audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    summary = [
        f"indices={indices}",
        f"avg_pred_similarity={sum(float(r['pred_similarity']) for r in rows_out) / len(rows_out):.4f}",
        f"avg_target_similarity={sum(float(r['target_similarity']) for r in rows_out) / len(rows_out):.4f}",
        f"csv={csv_path}",
    ]
    (args.out_dir / "SUMMARY.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"saved phoneme audit to {args.out_dir}")


if __name__ == "__main__":
    main()
