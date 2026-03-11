"""
Build a cleaned Turkish manifest suitable for Kokoro finetuning experiments.

This script:
  - joins manifest.csv with manifest_phonemized.csv
  - repairs the known mangled phonemized filename row via (text, speaker_id)
  - normalizes phoneme strings into Kokoro's released vocab
  - validates alignment availability
  - optionally filters rare alignment-row outliers
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "combined_dataset"
ALIGN_DIR = ROOT / "alignments"
BASE_MANIFEST = DATA_DIR / "manifest.csv"
PHON_MANIFEST = DATA_DIR / "manifest_phonemized.csv"
CONFIG = ROOT / "kokoro/weights/config.json"
DEFAULT_OUT = DATA_DIR / "kokoro_turkish_manifest.csv"


def normalize_phonemes(phonemes: str) -> str:
    phonemes = " ".join(phonemes.splitlines())
    phonemes = phonemes.replace("ɫ", "l")
    phonemes = " ".join(phonemes.split())
    return phonemes


def load_base_manifest_map():
    base = {}
    with BASE_MANIFEST.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            base[(row["text"], row["speaker_id"])] = row["file"]
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--keep-row-diffs",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Keep samples whose alignment row count differs from normalized phoneme length by one of these values.",
    )
    args = parser.parse_args()

    vocab = set(json.loads(CONFIG.read_text(encoding="utf-8"))["vocab"].keys())
    base_manifest_map = load_base_manifest_map()
    row_diff_counts = Counter()
    speaker_counts = Counter()
    filtered_out = 0
    repaired = 0
    written = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PHON_MANIFEST.open(newline="", encoding="utf-8") as src, args.out.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        fieldnames = [
            "file",
            "text",
            "speaker_id",
            "speaker_index",
            "phonemes",
            "phoneme_length",
            "alignment",
            "align_rows",
            "align_frames",
            "row_diff",
        ]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        speaker_to_index = {}

        for row_index, row in enumerate(reader, start=2):
            audio_file = row["file"]
            align_file = ALIGN_DIR / Path(audio_file).name.replace(".wav", ".pt")
            if not align_file.exists():
                repaired_file = base_manifest_map.get((row["text"], row["speaker_id"]))
                if repaired_file is None:
                    raise FileNotFoundError(
                        f"Row {row_index}: could not repair missing file {row['file']!r}"
                    )
                repaired += 1
                audio_file = repaired_file
                align_file = ALIGN_DIR / Path(audio_file).name.replace(".wav", ".pt")

            phonemes = normalize_phonemes(row["phonemes"])
            missing = sorted(set(ch for ch in phonemes if ch not in vocab))
            if missing:
                raise ValueError(
                    f"Row {row_index}: normalized phonemes still contain out-of-vocab symbols: {missing}"
                )

            align = torch_load_shape(align_file)
            row_diff = align[0] - len(phonemes)
            row_diff_counts[row_diff] += 1
            if row_diff not in args.keep_row_diffs:
                filtered_out += 1
                continue

            if row["speaker_id"] not in speaker_to_index:
                speaker_to_index[row["speaker_id"]] = len(speaker_to_index)
            speaker_index = speaker_to_index[row["speaker_id"]]
            speaker_counts[row["speaker_id"]] += 1

            writer.writerow(
                {
                    "file": audio_file,
                    "text": row["text"],
                    "speaker_id": row["speaker_id"],
                    "speaker_index": speaker_index,
                    "phonemes": phonemes,
                    "phoneme_length": len(phonemes),
                    "alignment": f"alignments/{Path(audio_file).name.replace('.wav', '.pt')}",
                    "align_rows": align[0],
                    "align_frames": align[1],
                    "row_diff": row_diff,
                }
            )
            written += 1

    print("=" * 88)
    print("  BUILD TURKISH TRAINING MANIFEST")
    print("=" * 88)
    print(f"output:              {args.out.relative_to(ROOT)}")
    print(f"written rows:        {written:,}")
    print(f"filtered out:        {filtered_out:,}")
    print(f"repaired rows:       {repaired:,}")
    print(f"keep row diffs:      {sorted(args.keep_row_diffs)}")
    print("speakers:")
    for speaker_id, count in speaker_counts.items():
        print(f"  {speaker_id:<20} {count:,}")
    print("row diff histogram:")
    for diff, count in row_diff_counts.most_common(8):
        print(f"  {diff:<2} -> {count:,}")


def torch_load_shape(path: Path) -> tuple[int, int]:
    import torch

    x = torch.load(path, map_location="cpu", weights_only=True)
    return int(x.shape[0]), int(x.shape[1])


if __name__ == "__main__":
    main()
