#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_hic_stats.py — side-by-side comparison of pairtools .stats files

Reads any number of pairtools stats files and reports the quantities that decide
whether two Hi-C datasets are comparable: duplication rate, cis/trans balance,
and the long-range cis fraction. Equal total read counts can hide very different
usable signal, which is exactly the situation these numbers expose.

Usage
-----
# pooled comparison across cell types
python compare_hic_stats.py \
    --label Mic=.../Mic/distillerOut/results/stats_library_group/all_libraries.hg38.stats \
    --label NPC=.../NPC/distillerOut/results/stats_library_group/all_libraries.hg38.stats \
    --label Neuron=.../Neuron/distillerOut/results/stats_library_group/all_libraries.hg38.stats

# per-library within one cell type, flagging outliers
python compare_hic_stats.py --glob '.../Mic/distillerOut/results/pairs_library/*.dedup.stats' \
    --flag-outliers

python compare_hic_stats.py --self_test
"""

from __future__ import annotations

import argparse
import glob as globmod
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_pairtools_stats(path: str) -> Dict[str, float]:
    """Key/value lines from a pairtools stats file. Nested keys (chrom_freq/…,
    dist_freq/…, pair_types/…) are skipped except pair_types/UU."""
    out: Dict[str, float] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            key, val = line.split("\t", 1)
            if key.startswith(("chrom_freq/", "dist_freq/")):
                continue
            if key.startswith("pair_types/") and key != "pair_types/UU":
                continue
            try:
                out[key] = float(val)
            except ValueError:
                continue
    if not out:
        raise ValueError(f"no key/value pairs parsed from {path}")
    return out


def derive(stats: Dict[str, float]) -> Dict[str, float]:
    """Rates and fractions. Missing keys give NaN rather than a crash, since
    pairtools versions differ in which cis_Nkb+ counters they emit."""
    g = lambda k: stats.get(k, np.nan)  # noqa: E731
    total = g("total")
    mapped = g("total_mapped")
    dups = g("total_dups")
    nodups = g("total_nodups")
    cis = g("cis")
    trans = g("trans")
    cis20 = g("cis_20kb+")
    cis10 = g("cis_10kb+")
    cis1 = g("cis_1kb+")

    d = dict(
        total_M=total / 1e6,
        mapped_M=mapped / 1e6,
        nodups_M=nodups / 1e6,
        map_rate=mapped / total if total else np.nan,
        dup_rate=dups / mapped if mapped else np.nan,
        cis_frac=cis / nodups if nodups else np.nan,
        trans_frac=trans / nodups if nodups else np.nan,
        cis_trans=cis / trans if trans else np.nan,
        cis_1kb_of_cis=cis1 / cis if cis else np.nan,
        cis_10kb_of_cis=cis10 / cis if cis else np.nan,
        cis_20kb_of_cis=cis20 / cis if cis else np.nan,
        cis_20kb_M=cis20 / 1e6,
    )
    # usable long-range cis pairs are what a 2 kb contact map is actually built from
    d["usable_M"] = d["cis_20kb_M"]
    return d


def collect(paths: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for label, path in paths.items():
        if not os.path.exists(path):
            print(f"[warn] missing {path}", file=sys.stderr)
            continue
        try:
            rec = derive(parse_pairtools_stats(path))
        except ValueError as e:
            print(f"[warn] {e}", file=sys.stderr)
            continue
        rec["sample"] = label
        rec["path"] = path
        rows.append(rec)
    if not rows:
        raise SystemExit("❌ nothing parsed")
    df = pd.DataFrame(rows)
    cols = ["sample", "total_M", "mapped_M", "nodups_M", "map_rate", "dup_rate",
            "cis_frac", "trans_frac", "cis_trans", "cis_1kb_of_cis",
            "cis_10kb_of_cis", "cis_20kb_of_cis", "usable_M", "path"]
    return df[[c for c in cols if c in df.columns]]


def flag_outliers(df: pd.DataFrame, z: float = 1.5) -> pd.DataFrame:
    """Median-absolute-deviation flags, so one bad library cannot hide behind the
    mean it is inflating."""
    out = df.copy()
    for col, worse in (("dup_rate", "high"), ("trans_frac", "high"),
                       ("cis_20kb_of_cis", "low")):
        if col not in out.columns:
            continue
        v = out[col].to_numpy(float)
        med = np.nanmedian(v)
        mad = np.nanmedian(np.abs(v - med)) or np.nan
        score = (v - med) / (1.4826 * mad) if mad and np.isfinite(mad) else np.zeros_like(v)
        out[f"{col}_z"] = np.round(score, 2)
        out[f"{col}_flag"] = (score > z) if worse == "high" else (score < -z)
    flag_cols = [c for c in out.columns if c.endswith("_flag")]
    out["suspect"] = out[flag_cols].any(axis=1) if flag_cols else False
    return out


def self_test() -> int:
    import tempfile

    good = """total\t1000000
total_unmapped\t50000
total_mapped\t900000
total_dups\t90000
total_nodups\t810000
cis\t648000
trans\t162000
cis_1kb+\t600000
cis_10kb+\t420000
cis_20kb+\t324000
pair_types/UU\t900000
chrom_freq/chr1/chr1\t123
dist_freq/1000-2000/++\t45
"""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "a.stats")
        open(p, "w").write(good)
        s = parse_pairtools_stats(p)
        assert "chrom_freq/chr1/chr1" not in s and "dist_freq/1000-2000/++" not in s
        assert s["pair_types/UU"] == 900000
        d = derive(s)
        assert abs(d["dup_rate"] - 0.1) < 1e-9
        assert abs(d["cis_frac"] - 0.8) < 1e-9
        assert abs(d["cis_trans"] - 4.0) < 1e-9
        assert abs(d["cis_20kb_of_cis"] - 0.5) < 1e-9

        # a version without the cis_20kb+ counter must degrade to NaN, not crash
        p2 = os.path.join(td, "b.stats")
        open(p2, "w").write("total\t10\ntotal_mapped\t9\ncis\t5\ntrans\t1\n")
        d2 = derive(parse_pairtools_stats(p2))
        assert np.isnan(d2["cis_20kb_of_cis"]) and np.isnan(d2["dup_rate"])

        df = collect({"good": p, "sparse": p2})
        assert len(df) == 2 and list(df["sample"]) == ["good", "sparse"]

    # outlier flagging: one library with high trans and low long-range cis
    df = pd.DataFrame(dict(
        sample=[f"lib{i}" for i in range(5)],
        dup_rate=[.10, .11, .10, .12, .10],
        trans_frac=[.20, .21, .20, .22, .55],
        cis_20kb_of_cis=[.50, .49, .51, .50, .18]))
    f = flag_outliers(df)
    assert f.loc[4, "suspect"] and not f.loc[0:3, "suspect"].any(), "one bad library flagged"

    print("✅ compare_hic_stats self-test passed")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--label", action="append", default=[], metavar="NAME=PATH",
                   help="labelled stats file; repeat per sample")
    p.add_argument("--glob", action="append", default=[],
                   help="glob of stats files; the filename becomes the label")
    p.add_argument("--flag-outliers", action="store_true",
                   help="MAD-based flags for dup rate, trans fraction, long-range cis")
    p.add_argument("--out-tsv", default=None)
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()

    paths: Dict[str, str] = {}
    for spec in args.label:
        if "=" not in spec:
            print(f"❌ --label expects NAME=PATH, got {spec!r}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        paths[name.strip()] = path.strip()
    for g in args.glob:
        for path in sorted(globmod.glob(g)):
            paths[os.path.basename(path).split(".")[0]] = path
    if not paths:
        print("❌ give --label NAME=PATH or --glob PATTERN", file=sys.stderr)
        return 2

    df = collect(paths)
    if args.flag_outliers:
        df = flag_outliers(df)

    show = [c for c in df.columns if c not in ("path",) and not c.endswith("_z")]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[show].round(4).to_string(index=False))

    if args.flag_outliers and "suspect" in df.columns and df["suspect"].any():
        print("\n⚠️  flagged (>1.5 MAD from the median on dup rate, trans fraction, "
              "or long-range cis):")
        print(df.loc[df["suspect"], ["sample", "dup_rate", "trans_frac",
                                     "cis_20kb_of_cis"]].round(4).to_string(index=False))

    if args.out_tsv:
        df.to_csv(args.out_tsv, sep="\t", index=False)
        print(f"\n✅ {args.out_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
