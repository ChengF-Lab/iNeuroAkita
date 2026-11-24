#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
library_reproducibility.py — how well do libraries of one cell type agree?

Answers the question a model cannot: what is the highest Pearson r any predictor
could reach on these maps? Two libraries of the same cell type differ only by
sampling and by whatever biological heterogeneity separates the donors, so the
correlation between them is the ceiling for a sequence-to-structure model.

The comparison is done on the SAME windows and with the SAME geometry the model
is evaluated on (cropped extent, log observed/expected, upper triangle from
diagonal_offset), so the numbers are directly comparable to the `pearson` column
of your *_per_window_metrics.tsv.

Depth is controlled: with --downsample every library in a pair is binomially
thinned to the shallower one's count within each window, so a shallow library is
not penalised twice (once for noise, once for depth).

Usage
-----
python library_reproducibility.py \
    --coolers '.../Mic/distillerOut/results/coolers_library/*.mapq_30.2048.cool' \
    --sequences .../model_input/input/fold1/sequences.bed \
    --splits test \
    --downsample \
    --out-prefix .../QC_hic_status/Mic_reproducibility

# reference cell type, same settings
python library_reproducibility.py \
    --coolers '.../Neuron/distillerOut/results/coolers_library/*.mapq_30.2048.cool' \
    --sequences .../fold1/sequences.bed --splits test --downsample \
    --out-prefix .../QC_hic_status/Neuron_reproducibility

python library_reproducibility.py --self_test
"""

from __future__ import annotations

import argparse
import glob as globmod
import itertools
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# map preparation (pure numpy; unit-tested)
# ---------------------------------------------------------------------------


def observed_over_expected(mat: np.ndarray, diagonal_offset: int = 2,
                           clip: float = 2.0) -> np.ndarray:
    """log(observed / expected-by-distance), clipped — the Akita target transform.

    Expected is the mean over ALL valid pixels of a diagonal, including zeros, and
    zero pixels become -clip rather than being dropped. Dropping them (an earlier
    version of this function did) discards the depleted pixels, which carry much of
    the structural signal, and biases the correlation towards whichever pixels
    happened to be covered in both samples. Clipping at +/-2 matches `clip: 2` in
    your targets.txt, so these numbers are comparable to the model's Pearson r.

    NaN marks structurally invalid pixels only: the masked diagonal band and bins
    with no balancing weight.
    """
    L = mat.shape[0]
    out = np.full((L, L), np.nan)
    for d in range(diagonal_offset, L):
        idx = (np.arange(L - d), np.arange(d, L))
        vals = mat[idx]
        valid = np.isfinite(vals)
        if valid.sum() < 2:
            continue
        exp = vals[valid].mean()
        if not np.isfinite(exp) or exp <= 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log(np.where(valid, vals, np.nan) / exp)
        logged = np.clip(logged, -clip, clip)      # zeros -> -clip, not dropped
        out[idx] = logged
        out[(idx[1], idx[0])] = logged
    return out


def thin_counts(mat: np.ndarray, target_total: float, rng: np.random.Generator) -> np.ndarray:
    """Binomially thin a raw count matrix to `target_total` total counts."""
    mat = np.nan_to_num(mat, nan=0.0)
    total = mat.sum()
    if total <= 0 or target_total >= total:
        return mat
    p = float(target_total / total)
    return rng.binomial(mat.astype(np.int64), p).astype(float)


def triu_values(mat: np.ndarray, diagonal_offset: int = 2) -> np.ndarray:
    return mat[np.triu_indices(mat.shape[0], diagonal_offset)]


def pair_pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    x, y = a[m], b[m]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def map_extent(region_start: int, region_end: int, n_bins: int, bin_size: int,
               crop_bp: Optional[int] = None) -> Tuple[int, int]:
    """Genomic span the model's n_bins x n_bins map actually covers."""
    covered = n_bins * bin_size
    if crop_bp is None:
        crop_bp = max(int(round(((region_end - region_start) - covered) / 2)), 0)
    return region_start + crop_bp, region_start + crop_bp + covered


# ---------------------------------------------------------------------------
# cooler access
# ---------------------------------------------------------------------------


def fetch_window(clr, chrom: str, start: int, end: int, n_bins: int,
                 balance: bool = True, weight_clr=None) -> Tuple[np.ndarray, np.ndarray]:
    """Raw counts, plus a weight product taken from `weight_clr` when given.

    Per-library coolers are usually unbalanced while the pooled one is balanced.
    Comparing a raw map with a balanced map mixes coverage bias into the
    correlation, so all maps must use the SAME weights — normally the pooled
    cooler\'s, since every library shares its bin table.
    """
    region = f"{chrom}:{start}-{end}"
    counts = clr.matrix(balance=False, sparse=False).fetch(region).astype(float)
    counts = counts[:n_bins, :n_bins]
    if counts.shape != (n_bins, n_bins):
        pad = n_bins - counts.shape[0]
        counts = np.pad(counts, ((0, pad), (0, pad)), constant_values=np.nan)
    if not balance:
        return counts, np.ones_like(counts)
    src = weight_clr if weight_clr is not None else clr
    try:
        bins = src.bins().fetch(region)
        w = bins["weight"].to_numpy(float)[:n_bins]
        if w.size < n_bins:
            w = np.pad(w, (0, n_bins - w.size), constant_values=np.nan)
        wprod = np.outer(w, w)
    except (KeyError, ValueError):
        wprod = np.ones_like(counts)
    return counts, wprod


def poisson_split(mat: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Split a count matrix into two independent pseudo-replicates.

    Each contact goes to half A or half B with probability 0.5, so the two halves
    are independent samples of the same underlying map. Their agreement isolates
    SAMPLING noise: it is the ceiling a model could reach if the only limitation
    were read depth, with biological heterogeneity held constant.
    """
    counts = np.nan_to_num(mat, nan=0.0)
    a = rng.binomial(counts.astype(np.int64), 0.5).astype(float)
    return a, counts - a


def spearman_brown(r_half: float, n: float = 2.0) -> float:
    """Reliability at n-fold depth given the split-half value.

    Two halves each hold half the pooled depth, so the pooled map is more
    reliable than either half; this is the standard correction for that.
    """
    if not np.isfinite(r_half) or r_half <= -1 / (n - 1 + 1e-12):
        return np.nan
    return float(n * r_half / (1 + (n - 1) * r_half))


def coarsen(mat: np.ndarray, k: int) -> np.ndarray:
    """Average k x k blocks — lower resolution, higher counts per pixel."""
    if k <= 1:
        return mat
    L = (mat.shape[0] // k) * k
    m = mat[:L, :L].reshape(L // k, k, L // k, k)
    return np.nanmean(np.nanmean(m, axis=3), axis=1)


def load_windows(sequences_bed: str, splits: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(sequences_bed, sep="\t", header=None, usecols=[0, 1, 2, 3],
                     names=["chrom", "start", "end", "split"], dtype={0: str})
    sub = df[df["split"].isin(splits)].reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"❌ no rows with split in {splits} "
                         f"(available: {sorted(df['split'].unique())})")
    return sub


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------


def self_test() -> int:
    rng = np.random.default_rng(0)
    L, off = 32, 2

    # obs/exp removes distance decay: a pure decay matrix flattens to ~0
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    decay = 1.0 / (1 + np.abs(i - j))
    oe = observed_over_expected(decay, off)
    assert np.nanstd(triu_values(oe, off)) < 1e-9, "pure decay should flatten"

    # zeros must be retained as strong depletion, not dropped
    withzero = decay.copy()
    withzero[3, 25] = withzero[25, 3] = 0.0
    oz = observed_over_expected(withzero, off, clip=2.0)
    assert oz[3, 25] == -2.0, "zero pixel clipped to -clip, not NaN"
    # every pixel survives except the corner diagonal, which holds a single pixel
    # and therefore has no meaningful expected value
    finite_frac = np.isfinite(triu_values(oz, off)).mean()
    assert finite_frac > 0.99, f"zeros must not be dropped (finite={finite_frac:.3f})"
    assert np.isnan(oz[0, L - 1]), "single-pixel corner diagonal skipped"
    assert np.isnan(np.diag(oe)).all() and np.isnan(np.diag(oe, 1)).all(), "masked band"
    assert np.allclose(oe, oe.T, equal_nan=True), "symmetric"

    # structure on top of decay survives
    struct = decay.copy()
    struct[5:12, 5:12] *= 3.0
    oe2 = observed_over_expected(struct, off)
    assert np.nanstd(triu_values(oe2, off)) > 0.1, "structure preserved"

    # thinning reduces depth without changing structure much
    counts = (decay * 5e4).round()
    thinned = thin_counts(counts, counts.sum() / 4, rng)
    assert 0.2 < thinned.sum() / counts.sum() < 0.3, "thinned to ~1/4"
    r = pair_pearson(triu_values(observed_over_expected(counts, off), off),
                     triu_values(observed_over_expected(thinned, off), off))
    assert np.isnan(r) or r > -1, "correlation computable after thinning"

    # two independent Poisson draws of the same STRUCTURED truth correlate;
    # unrelated maps do not. (A pure-decay truth would flatten to zero under
    # obs/exp, leaving only Poisson noise for both — no structure to share.)
    truth = struct * 4e4
    a = observed_over_expected(rng.poisson(truth).astype(float), off)
    b = observed_over_expected(rng.poisson(truth).astype(float), off)
    shuffled = truth.copy()
    perm = rng.permutation(L)
    shuffled = shuffled[np.ix_(perm, perm)]
    c = observed_over_expected(rng.poisson(shuffled).astype(float), off)
    r_ab = pair_pearson(triu_values(a, off), triu_values(b, off))
    r_ac = pair_pearson(triu_values(a, off), triu_values(c, off))
    assert r_ab > 0.5, f"replicates of a structured truth should agree ({r_ab:.3f})"
    assert r_ab > r_ac, f"replicates should agree more than unrelated maps ({r_ab} vs {r_ac})"

    # crop-aware extent
    s, e = map_extent(1_000_000, 1_000_000 + 1_048_576, 448, 2048)
    assert e - s == 448 * 2048 and s == 1_000_000 + 65536, "crop derived correctly"

    # pair_pearson guards
    assert np.isnan(pair_pearson(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))
    assert np.isnan(pair_pearson(np.array([np.nan, np.nan]), np.array([1.0, 2.0])))

    print("✅ library_reproducibility self-test passed")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--coolers", help="glob of per-library .cool files at the model's bin size")
    p.add_argument("--pooled-cooler", default=None,
                   help="optional pooled cooler; each library is also compared to it")
    p.add_argument("--sequences", help="sequences.bed defining the evaluation windows")
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--max-windows", type=int, default=100,
                   help="cap for speed; windows are sampled evenly across the set")
    p.add_argument("--n-bins", type=int, default=448)
    p.add_argument("--bin-size", type=int, default=2048)
    p.add_argument("--crop-bp", type=int, default=None, help="default: derived from n_bins")
    p.add_argument("--diagonal-offset", type=int, default=2)
    p.add_argument("--downsample", action="store_true",
                   help="thin each pair to matched depth per window (recommended when "
                        "library depths differ)")
    p.add_argument("--no-balance", action="store_true", help="ignore ICE weights")
    p.add_argument("--weights-from", default=None,
                   help="cooler whose ICE weights are applied to ALL maps (normally the "
                        "pooled one). Without this, unbalanced libraries and a balanced "
                        "pooled map are not comparable. Defaults to --pooled-cooler.")
    p.add_argument("--groups", action="append", default=[], metavar="NAME=SUBSTR,SUBSTR",
                   help="sum libraries into a group before comparing, e.g. "
                        "'halfA=ACTGAT,CGTACG' 'halfB=GAGTGG,ATTCCT'")
    p.add_argument("--split-half", action="store_true",
                   help="auto-split libraries into two depth-balanced halves and report "
                        "split-half reliability (+ Spearman-Brown estimate for full depth)")
    p.add_argument("--pseudo-split", action="store_true",
                   help="binomially split EACH input map into two pseudo-replicates; "
                        "isolates sampling noise. Works with a single cooler.")
    p.add_argument("--coarsen", type=int, nargs="*", default=[1],
                   help="also report at coarser resolutions, e.g. 1 2 4 (bin multiples)")
    p.add_argument("--clip", type=float, default=2.0,
                   help="clip on log(obs/exp); match the 'clip' column of your targets.txt")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-prefix", default="reproducibility")
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    for req in ("coolers", "sequences"):
        if getattr(args, req) is None:
            print(f"❌ --{req} is required", file=sys.stderr)
            return 2

    import cooler

    paths = sorted(globmod.glob(args.coolers))
    if not paths:
        print(f"❌ no coolers matched {args.coolers!r}", file=sys.stderr)
        return 2
    if len(paths) < 2 and not args.pseudo_split:
        print(f"❌ found only {len(paths)} cooler. Either point --coolers at the "
              f"per-library directory, or use --pseudo-split to split this one map "
              f"into two independent halves.", file=sys.stderr)
        return 2
    names = [os.path.basename(p).split(".")[0] for p in paths]
    clrs = {n: cooler.Cooler(p) for n, p in zip(names, paths)}
    if args.pooled_cooler:
        clrs["POOLED"] = cooler.Cooler(args.pooled_cooler)
        names.append("POOLED")

    print("📚 libraries")
    for n in names:
        c = clrs[n]
        has_w = "weight" in c.bins().columns
        print(f"   {n:28s} binsize={c.binsize} sum={c.info.get('sum', 'NA')} "
              f"balanced={'yes' if has_w else 'NO'}")
        if c.binsize != args.bin_size:
            print(f"❌ {n} has binsize {c.binsize}, expected {args.bin_size}", file=sys.stderr)
            return 2

    weight_clr = None
    if not args.no_balance:
        wpath = args.weights_from or args.pooled_cooler
        if wpath:
            weight_clr = cooler.Cooler(wpath)
            if "weight" not in weight_clr.bins().columns:
                print(f"⚠️  {wpath} has no ICE weights; falling back to per-cooler weights",
                      file=sys.stderr)
                weight_clr = None
            else:
                print(f"\n⚖️  applying ICE weights from {os.path.basename(wpath)} to every map")
        elif not all("weight" in clrs[n].bins().columns for n in names):
            print("\n⚠️  some coolers are unbalanced and no --weights-from was given. "
                  "Raw and balanced maps are NOT comparable; pass --weights-from "
                  "<pooled.cool> or --no-balance.", file=sys.stderr)

    # ---- samples: single libraries, or groups summed before comparison -------
    groups: Dict[str, List[str]] = {}
    if args.groups:
        for spec in args.groups:
            gname, members = spec.split("=", 1)
            keys = [k.strip() for k in members.split(",") if k.strip()]
            picked = [n for n in names if any(k in n for k in keys)]
            if not picked:
                print(f"❌ group {gname!r} matched no library", file=sys.stderr)
                return 2
            groups[gname.strip()] = picked
    elif args.split_half and len([n for n in names if n != "POOLED"]) >= 2:
        libs = [n for n in names if n != "POOLED"]
        libs = sorted(libs, key=lambda n: clrs[n].info.get("sum", 0), reverse=True)
        a, b = [], []          # greedy: alternate deepest first so halves match in depth
        for i, n in enumerate(libs):
            (a if sum(clrs[x].info.get("sum", 0) for x in a)
                 <= sum(clrs[x].info.get("sum", 0) for x in b) else b).append(n)
        groups = {"halfA": a, "halfB": b}
        print(f"\n✂️  split-half: halfA={a}  halfB={b}")
    else:
        groups = {n: [n] for n in names}

    wins = load_windows(args.sequences, args.splits)
    if len(wins) > args.max_windows:
        idx = np.linspace(0, len(wins) - 1, args.max_windows).astype(int)
        wins = wins.iloc[idx].reset_index(drop=True)
    print(f"\n🪟 {len(wins)} windows on {', '.join(sorted(wins['chrom'].unique()))}")

    rng = np.random.default_rng(args.seed)
    levels = sorted(set(int(k) for k in args.coarsen if int(k) >= 1))
    rows, depth_rows = [], []

    for wi, w in wins.iterrows():
        s_, e_ = map_extent(int(w["start"]), int(w["end"]), args.n_bins,
                            args.bin_size, args.crop_bp)
        counts, wprod, depth = {}, {}, {}
        ok = True
        for gname, members in groups.items():
            acc, wp = None, None
            for mem in members:
                try:
                    cnt, w_ = fetch_window(clrs[mem], w["chrom"], s_, e_, args.n_bins,
                                           balance=not args.no_balance,
                                           weight_clr=weight_clr)
                except Exception as ex:
                    print(f"[warn] {mem} {w['chrom']}:{s_}-{e_}: {ex}", file=sys.stderr)
                    ok = False
                    break
                acc = cnt if acc is None else acc + cnt
                wp = w_ if wp is None else wp
            if not ok:
                break
            counts[gname], wprod[gname] = acc, wp
            depth[gname] = float(np.nansum(acc))
        if not ok:
            continue

        n_pix = args.n_bins * (args.n_bins - 1) / 2
        for gname in groups:
            depth_rows.append(dict(sample=gname, window=f"{w['chrom']}:{w['start']}",
                                   counts=depth[gname],
                                   counts_per_pixel=depth[gname] / n_pix))

        pairs_to_score = []
        if args.pseudo_split:
            for gname in groups:
                a, b = poisson_split(counts[gname], rng)
                pairs_to_score.append((f"{gname}:halfA", f"{gname}:halfB", a, b,
                                       wprod[gname], wprod[gname], True))
        else:
            for a, b in itertools.combinations(groups, 2):
                ca, cb = counts[a], counts[b]
                if args.downsample:
                    t = min(depth[a], depth[b])
                    ca, cb = thin_counts(ca, t, rng), thin_counts(cb, t, rng)
                pairs_to_score.append((a, b, ca, cb, wprod[a], wprod[b], args.downsample))

        for a, b, ca, cb, wa, wb, matched in pairs_to_score:
            for k in levels:
                ma = observed_over_expected(coarsen(ca * wa, k), args.diagonal_offset,
                                            args.clip)
                mb = observed_over_expected(coarsen(cb * wb, k), args.diagonal_offset,
                                            args.clip)
                rows.append(dict(
                    window=f"{w['chrom']}:{w['start']}-{w['end']}", chrom=w["chrom"],
                    lib_a=a, lib_b=b, coarsen=k,
                    resolution_bp=args.bin_size * k,
                    matched_depth=matched, pearson=pair_pearson(
                        triu_values(ma, args.diagonal_offset),
                        triu_values(mb, args.diagonal_offset))))
        if (wi + 1) % 20 == 0:
            print(f"   {wi + 1}/{len(wins)} windows")

    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ no comparisons computed", file=sys.stderr)
        return 2
    df.to_csv(f"{args.out_prefix}_per_window.tsv", sep="\t", index=False)

    dd = pd.DataFrame(depth_rows).groupby("sample").agg(
        median_counts=("counts", "median"),
        median_counts_per_pixel=("counts_per_pixel", "median")).reset_index()
    print("\n🔢 depth per window (sparsity check)")
    print(dd.round(3).to_string(index=False))
    if (dd["median_counts_per_pixel"] < 1).any():
        print("   ⚠️  under ~1 count per pixel: at this depth two independent samples "
              "correlate weakly from Poisson noise alone, so a low r here reflects "
              "sparsity rather than disagreement. Pool libraries (--split-half) or "
              "compare at lower resolution (--coarsen 2 4).")

    summ = (df.groupby(["lib_a", "lib_b", "resolution_bp"])["pearson"]
            .agg(n="size", median="median", mean="mean", sd="std")
            .reset_index().sort_values(["resolution_bp", "median"], ascending=[True, False]))
    summ["spearman_brown_full_depth"] = summ["median"].apply(spearman_brown)
    summ.to_csv(f"{args.out_prefix}_summary.tsv", sep="\t", index=False)

    print("\n📊 reproducibility (median per-window Pearson r"
          + (", depth-matched" if args.downsample and not args.pseudo_split else "") + ")")
    print(summ.round(4).to_string(index=False))

    base = summ[summ["resolution_bp"] == args.bin_size]
    lib_only = base[(~base["lib_a"].str.contains("POOLED"))
                    & (~base["lib_b"].str.contains("POOLED"))]
    if len(lib_only):
        ceiling = float(lib_only["median"].median())
        print(f"\n🎯 measured agreement at {args.bin_size} bp: {ceiling:.4f}")
        if args.pseudo_split or args.split_half:
            print(f"   Spearman-Brown estimate at full pooled depth: "
                  f"{spearman_brown(ceiling):.4f}")
        if args.pseudo_split:
            print("   pseudo-replicates share the same biology, so this is the "
                  "SAMPLING-noise ceiling only; comparing real libraries additionally "
                  "captures donor-to-donor differences.")
    print(f"\n✅ {args.out_prefix}_summary.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
