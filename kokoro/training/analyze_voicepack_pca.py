"""
analyze_voicepack_pca.py

Analysis 4: Voicepack PCA / rank analysis.

Loads all released voicepack .pt files and performs PCA to determine:
  - How many dimensions actually matter for voice identity
  - The effective rank of the voicepack space
  - Whether male and female voices cluster separately
  - How "far" a Turkish init (mean, gender-matched) sits from existing voices

This informs the gt_bootstrap vs. voicepack_bootstrap choice:
  - Low effective rank → voicepack_bootstrap is safe; a gender-matched
    init already captures most voice variation.
  - High effective rank → gt_bootstrap or random init may be needed to
    reach regions of voice space not covered by existing packs.

Usage:
    python kokoro/training/analyze_voicepack_pca.py
    python kokoro/training/analyze_voicepack_pca.py --plot          (requires matplotlib)
    python kokoro/training/analyze_voicepack_pca.py --top-k 20
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import VOICES_DIR, mean_released_pack, named_pack


# Prefix → gender label
def gender_of(name: str) -> str:
    if name.startswith("af") or name.startswith("bf") or name.startswith("ef") or \
       name.startswith("ff") or name.startswith("if") or name.startswith("zf"):
        return "female"
    if name.startswith("am") or name.startswith("bm") or name.startswith("em") or \
       name.startswith("jm") or name.startswith("pm") or name.startswith("zm"):
        return "male"
    return "unknown"


def load_all_packs() -> tuple[list[str], list[str], torch.Tensor]:
    """Returns (names, genders, stacked_tensor [n_voices, 510, 256])."""
    paths = sorted(VOICES_DIR.glob("*.pt"))
    names, genders, packs = [], [], []
    for p in paths:
        t = torch.load(p, map_location="cpu", weights_only=True).squeeze(1).float()
        if t.shape != (510, 256):
            print(f"  Warning: {p.stem} has unexpected shape {list(t.shape)}, skipping")
            continue
        names.append(p.stem)
        genders.append(gender_of(p.stem))
        packs.append(t)
    return names, genders, torch.stack(packs)  # [n, 510, 256]


def effective_rank(singular_values: torch.Tensor) -> float:
    """Effective rank via entropy of normalized singular value distribution."""
    p = singular_values / singular_values.sum()
    p = p[p > 1e-10]
    return math.exp(-(p * p.log()).sum().item())


def pca_analysis(matrix: torch.Tensor, top_k: int) -> dict:
    """
    matrix: [N, D]
    Returns dict with singular values, explained variance, effective rank.
    """
    mu = matrix.mean(0, keepdim=True)
    centered = matrix - mu
    # SVD
    try:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    except Exception:
        U, S, Vh = torch.svd(centered)
    variance = S ** 2
    total_var = variance.sum().item()
    explained = (variance / total_var).cumsum(0)
    eff_rank = effective_rank(S)

    dims_90 = (explained < 0.90).sum().item() + 1
    dims_95 = (explained < 0.95).sum().item() + 1
    dims_99 = (explained < 0.99).sum().item() + 1

    return {
        "S": S,
        "U": U,
        "Vh": Vh,
        "mu": mu,
        "explained": explained,
        "total_var": total_var,
        "effective_rank": eff_rank,
        "dims_90pct": dims_90,
        "dims_95pct": dims_95,
        "dims_99pct": dims_99,
        "top_k": min(top_k, S.shape[0]),
    }


def project(pack: torch.Tensor, pca: dict) -> torch.Tensor:
    """Project a [510, 256] pack onto top-k PCA components. Returns [k] coordinates."""
    flat = pack.flatten().unsqueeze(0)  # [1, D]
    centered = flat - pca["mu"]
    k = pca["top_k"]
    Vh_k = pca["Vh"][:k]  # [k, D]
    return (centered @ Vh_k.T).squeeze(0)  # [k]


def dist_to_cluster(coords: torch.Tensor, cluster_coords: torch.Tensor) -> float:
    """Mean L2 distance from a point to a set of points in PCA space."""
    diffs = cluster_coords - coords.unsqueeze(0)
    return diffs.norm(dim=-1).mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=10, help="Number of top PCA components to show (default: 10)")
    parser.add_argument("--plot", action="store_true", help="Plot PCA scatter (requires matplotlib)")
    args = parser.parse_args()

    print(f"Loading voicepacks from {VOICES_DIR} ...")
    names, genders, packs = load_all_packs()
    n_voices = packs.shape[0]
    print(f"Loaded {n_voices} voicepacks, shape per pack: {list(packs.shape[1:])}\n")

    # ── Analysis 1: PCA over voice identity (mean across length axis) ──────────
    # Collapse length axis → [n_voices, 256] to study voice identity dimensions
    voice_means = packs.mean(dim=1)  # [n, 256]

    print("=" * 60)
    print("VOICE IDENTITY PCA  (mean-pooled across length axis, [n_voices, 256])")
    print("=" * 60)
    pca_voice = pca_analysis(voice_means, args.top_k)
    print(f"  Effective rank:       {pca_voice['effective_rank']:.2f}  (out of {n_voices} voices x 256 dims)")
    print(f"  Dims for 90% var:     {pca_voice['dims_90pct']}")
    print(f"  Dims for 95% var:     {pca_voice['dims_95pct']}")
    print(f"  Dims for 99% var:     {pca_voice['dims_99pct']}")

    print(f"\n  Top-{pca_voice['top_k']} singular values and explained variance:")
    col = "    {:>4}  {:>12}  {:>14}  {:>14}"
    print(col.format("PC", "Singular val", "% var explained", "Cumul % var"))
    print("    " + "-" * 50)
    for i in range(pca_voice["top_k"]):
        sv = pca_voice["S"][i].item()
        pct = 100.0 * (pca_voice["S"][i] ** 2 / pca_voice["total_var"])
        cum = 100.0 * pca_voice["explained"][i].item()
        bar = "█" * int(pct / 2)
        print(col.format(f"PC{i+1}", f"{sv:.4f}", f"{pct:.2f}%", f"{cum:.2f}%") + f"  {bar}")

    # ── Analysis 2: PCA over flattened packs (n_voices * 510, 256) ────────────
    # Study the full length-aware structure
    flat_all = packs.reshape(-1, 256)  # [n*510, 256]
    print(f"\n{'=' * 60}")
    print("LENGTH-AWARE PCA  (all length slots flattened, [n_voices*510, 256])")
    print("=" * 60)
    pca_full = pca_analysis(flat_all, args.top_k)
    print(f"  Effective rank:       {pca_full['effective_rank']:.2f}  (out of {n_voices * 510} rows x 256 dims)")
    print(f"  Dims for 90% var:     {pca_full['dims_90pct']}")
    print(f"  Dims for 95% var:     {pca_full['dims_95pct']}")
    print(f"  Dims for 99% var:     {pca_full['dims_99pct']}")

    # ── Analysis 3: Gender clustering in PC space ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("GENDER CLUSTERING  (top-2 PC coords per voice)")
    print("=" * 60)

    k = min(2, pca_voice["top_k"])
    Vh_k = pca_voice["Vh"][:k]
    mu_v = pca_voice["mu"]
    coords = ((voice_means - mu_v) @ Vh_k.T)  # [n_voices, k]

    female_idx = [i for i, g in enumerate(genders) if g == "female"]
    male_idx = [i for i, g in enumerate(genders) if g == "male"]
    female_coords = coords[female_idx] if female_idx else None
    male_coords = coords[male_idx] if male_idx else None

    col2 = "  {:<20} {:>8} {:>8}"
    print(col2.format("Voice", "PC1", "PC2"))
    print("  " + "-" * 38)
    for i, (name, gender) in enumerate(zip(names, genders)):
        marker = "F" if gender == "female" else "M" if gender == "male" else "?"
        c = coords[i]
        pc1 = c[0].item() if k > 0 else 0.0
        pc2 = c[1].item() if k > 1 else 0.0
        print(col2.format(f"[{marker}] {name}", f"{pc1:.3f}", f"{pc2:.3f}"))

    if female_coords is not None and male_coords is not None:
        fc = female_coords.mean(0)
        mc = male_coords.mean(0)
        separation = (fc - mc).norm().item()
        within_f = (female_coords - fc.unsqueeze(0)).norm(dim=-1).mean().item()
        within_m = (male_coords - mc.unsqueeze(0)).norm(dim=-1).mean().item()
        print(f"\n  Male centroid:        {[f'{v:.3f}' for v in mc.tolist()]}")
        print(f"  Female centroid:      {[f'{v:.3f}' for v in fc.tolist()]}")
        print(f"  Centroid separation:  {separation:.4f}")
        print(f"  Within-female spread: {within_f:.4f}")
        print(f"  Within-male spread:   {within_m:.4f}")
        ratio = separation / ((within_f + within_m) / 2 + 1e-8)
        print(f"  Separation ratio:     {ratio:.2f}x  {'(well separated)' if ratio > 1.5 else '(overlapping)'}")

    # ── Analysis 4: Turkish init distance from existing voices ────────────────
    print(f"\n{'=' * 60}")
    print("TURKISH INIT DISTANCE FROM EXISTING VOICE CLUSTERS")
    print("=" * 60)
    print("How far each candidate Turkish init sits from the male/female cluster.\n")

    inits = {
        "mean (all voices)": mean_released_pack().mean(dim=0, keepdim=True).expand(1, 256),
        "af_heart (female)": named_pack("af_heart").mean(dim=0, keepdim=True).expand(1, 256),
        "pm_alex (male)": named_pack("pm_alex").mean(dim=0, keepdim=True).expand(1, 256),
    }

    col3 = "  {:<25} {:>14} {:>14} {:>14}"
    print(col3.format("Init strategy", "Dist→female", "Dist→male", "Dist→all"))
    print("  " + "-" * 70)
    for init_name, init_vec in inits.items():
        init_vec = init_vec.squeeze(0)  # [256]
        centered = (init_vec.unsqueeze(0) - mu_v) @ Vh_k.T  # [1, k]
        d_female = dist_to_cluster(centered.squeeze(0), female_coords) if female_coords is not None else float("nan")
        d_male = dist_to_cluster(centered.squeeze(0), male_coords) if male_coords is not None else float("nan")
        d_all = dist_to_cluster(centered.squeeze(0), coords)
        print(col3.format(init_name, f"{d_female:.4f}", f"{d_male:.4f}", f"{d_all:.4f}"))

    # ── Summary / recommendation ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY & RECOMMENDATION")
    print("=" * 60)
    eff = pca_voice["effective_rank"]
    d90 = pca_voice["dims_90pct"]
    if eff <= 5:
        print(f"  Effective rank = {eff:.1f}: very low-rank voicepack space.")
        print(f"  → voicepack_bootstrap (gender-matched init) is strongly justified.")
        print(f"  → A new Turkish voice will likely lie near existing same-gender voices.")
        print(f"  → gt_bootstrap carries higher risk of landing far from the learned manifold.")
    elif eff <= 15:
        print(f"  Effective rank = {eff:.1f}: moderate-rank voicepack space.")
        print(f"  → voicepack_bootstrap is a reasonable default init.")
        print(f"  → gt_bootstrap may be needed if Turkish voice characteristics are unusual.")
    else:
        print(f"  Effective rank = {eff:.1f}: high-rank voicepack space.")
        print(f"  → The voicepack space is rich; gender-matched init may not be close enough.")
        print(f"  → Consider gt_bootstrap or random init with smooth regularization.")
    print(f"\n  90% of variance explained by {d90} dimensions → keep at least {d90} dims unfrozen in voicepack.")

    # ── Optional plot ─────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, (name, gender) in enumerate(zip(names, genders)):
                color = "tab:red" if gender == "female" else "tab:blue" if gender == "male" else "gray"
                c = coords[i]
                ax.scatter(c[0].item(), c[1].item() if k > 1 else 0, color=color, zorder=3)
                ax.annotate(name, (c[0].item(), c[1].item() if k > 1 else 0), fontsize=6)
            # Plot inits
            markers = {"mean (all voices)": "s", "af_heart (female)": "^", "pm_alex (male)": "D"}
            for init_name, init_vec in inits.items():
                init_vec_flat = init_vec.squeeze(0)
                proj = ((init_vec_flat.unsqueeze(0) - mu_v) @ Vh_k.T).squeeze(0)
                ax.scatter(proj[0].item(), proj[1].item() if k > 1 else 0,
                           color="black", marker=markers.get(init_name, "*"), s=100,
                           label=f"init: {init_name}", zorder=5)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.set_title("Voicepack PCA — voice identity (mean-pooled)")
            from matplotlib.lines import Line2D
            legend_elems = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', label='Female'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', label='Male'),
            ]
            ax.legend(handles=legend_elems + ax.get_legend_handles_labels()[0], fontsize=8)
            plt.tight_layout()
            out_path = Path(__file__).parent / "voicepack_pca.png"
            plt.savefig(out_path, dpi=150)
            print(f"\n  Plot saved to {out_path}")
            plt.show()
        except ImportError:
            print("  matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
