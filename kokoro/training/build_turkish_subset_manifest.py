from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("combined_dataset/kokoro_turkish_manifest.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speaker", type=str, default=None, help="Filter to one speaker ID, or omit for all speakers")
    parser.add_argument("--min-phonemes", type=int, default=1)
    parser.add_argument("--max-phonemes", type=int, default=80)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys()

    filtered = [
        row
        for row in rows
        if (args.speaker is None or row["speaker_id"] == args.speaker)
        and args.min_phonemes <= int(row["phoneme_length"]) <= args.max_phonemes
    ]
    if args.max_rows > 0:
        filtered = filtered[: args.max_rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)

    print(f"speaker={args.speaker or 'all'}")
    print(f"rows={len(filtered)}")
    print(f"min_phonemes={args.min_phonemes}")
    print(f"max_phonemes={args.max_phonemes}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
