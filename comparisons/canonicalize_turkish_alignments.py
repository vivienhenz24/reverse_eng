"""
Canonicalize Turkish alignment tensors into Kokoro training format.

The cleaned manifest already filters to the dominant row-count regimes:
  diff = alignment_rows - len(phonemes) in {0, 1, 2}

This script converts each raw alignment into a strict one-hot matrix with
row count len(phonemes) + 2, matching Kokoro's BOS/EOS-expanded token path.

Canonicalization rules discovered from the dataset:
  - diff == 0: pad zero rows on both sides
  - diff == 1: append one zero row on the right
  - diff == 2: keep row count as-is

Rare corruption:
  - 80 tensors have a malformed final frame column with multiple active rows
  - collapse that final column onto the last active row to preserve monotonicity
"""

import argparse
import csv
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "combined_dataset/kokoro_turkish_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "alignments_kokoro_tr"


def collapse_non_onehot_columns(aln: torch.Tensor) -> tuple[torch.Tensor, int]:
    aln = aln.clone()
    repaired = 0
    col_sums = aln.sum(0)
    bad_cols = (col_sums != 1).nonzero().flatten().tolist()
    for col_idx in bad_cols:
        active = (aln[:, col_idx] != 0).nonzero().flatten()
        if len(active) == 0:
            continue
        keep = int(active[-1])
        aln[:, col_idx] = 0
        aln[keep, col_idx] = 1
        repaired += 1
    return aln, repaired


def canonicalize_alignment(aln: torch.Tensor, phoneme_length: int) -> tuple[torch.Tensor, int, int]:
    raw_rows, frames = int(aln.shape[0]), int(aln.shape[1])
    diff = raw_rows - phoneme_length

    aln, repaired_cols = collapse_non_onehot_columns(aln)
    zero = torch.zeros((1, frames), dtype=aln.dtype)

    if diff == 0:
        out = torch.cat([zero, aln, zero], dim=0)
    elif diff == 1:
        out = torch.cat([aln, zero], dim=0)
    elif diff == 2:
        out = aln
    else:
        raise ValueError(f"Unsupported row diff {diff} for phoneme_length={phoneme_length}, raw_rows={raw_rows}")

    target_rows = phoneme_length + 2
    if out.shape[0] != target_rows:
        raise ValueError(f"Canonicalized rows {out.shape[0]} != target_rows {target_rows}")

    col_sums = out.sum(0)
    if not torch.allclose(col_sums, torch.ones_like(col_sums)):
        raise ValueError(f"Canonicalized alignment is not one-hot by frame; max diff={float((col_sums - 1).abs().max())}")

    return out, diff, repaired_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = 0
    repaired_tensors = 0
    repaired_cols = 0
    diff_counts = {0: 0, 1: 0, 2: 0}

    with args.manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows += 1
            phoneme_length = int(row["phoneme_length"])
            raw_alignment = ROOT / row["alignment"]
            out_path = args.out_dir / raw_alignment.name

            aln = torch.load(raw_alignment, map_location="cpu", weights_only=True)
            canonical, diff, col_repairs = canonicalize_alignment(aln, phoneme_length)
            diff_counts[diff] += 1
            if col_repairs:
                repaired_tensors += 1
                repaired_cols += col_repairs
            torch.save(canonical, out_path)

    print("=" * 88)
    print("  CANONICALIZE TURKISH ALIGNMENTS")
    print("=" * 88)
    print(f"manifest:             {args.manifest.relative_to(ROOT)}")
    print(f"out dir:              {args.out_dir.relative_to(ROOT)}")
    print(f"rows written:         {rows:,}")
    print(f"row diff 0:           {diff_counts[0]:,}")
    print(f"row diff 1:           {diff_counts[1]:,}")
    print(f"row diff 2:           {diff_counts[2]:,}")
    print(f"repaired tensors:     {repaired_tensors:,}")
    print(f"repaired columns:     {repaired_cols:,}")


if __name__ == "__main__":
    main()
