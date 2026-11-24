#!/usr/bin/env python
"""
01_snv_matched_null.py  --  SNV42 empirical significance against matched background SNPs, for several matching sizes
(same matching and P-value definition as 05_variant_scoring/03_rank_select.py, which used --n_match 500)

Background : within-gene low-PP (< 0.01) fine-mapped microglia eQTL SNPs scored in the eQTL run (same models / settings),
             compare/eQTL/eQTL_scores.tsv (role == null), n = 18,980
Matching   : per SNV, the n nearest background SNPs in (log10 distance to the nearest TSS + 100 bp, GC fraction +/- 1 kb),
             each standardised by its SD in the background; ties broken by index (same as 03_rank_select)
P value    : (1 + #background >= score) / (n + 1); BH across the 42 SNVs, per head and metric (MSE, MaxAbs)
Also       : pooled P against all 18,980 background SNPs, with BH q (not computed in 03_rank_select)
Outputs    : compare/selection_matched/SNV42_matched.tsv (one row per SNV; columns per n_match, head, metric)
             compare/selection_matched/SNV42_matched_summary.txt (per n_match and head: #SNVs with q < 0.05 / 0.1 on both metrics,
             and the values for rs636317)
Usage      : python 01_snv_matched_null.py [--n_match 500 1000 2000 5000]
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--fa", default="ref_data/hg38.fa", help="hg38 FASTA (indexed)")
ap.add_argument("--n_match", type=int, nargs="+", default=[500, 1000, 2000, 5000])
a = ap.parse_args()
S = f"{a.base}/variant_scoring"; OUT = f"{S}/compare/selection_matched"; os.makedirs(OUT, exist_ok=True)
SUP = os.environ.get("SUPREMO_DIR", f"{S}/SuPreMo")   # patched SuPreMo clone (see README)
CELLS = ["NPC", "Mic", "Neuron"]; MET = ["mse", "maxabs"]

def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); r = p[o] * len(p) / np.arange(1, len(p) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; t = np.empty_like(r); t[o] = np.minimum(r, 1); return t
def emp(x, null):
    null = np.sort(np.asarray(null, float)); null = null[~np.isnan(null)]
    return (1 + len(null) - np.searchsorted(null, x, side="left")) / (len(null) + 1)

e = pd.read_csv(f"{S}/compare/eQTL/eQTL_scores.tsv", sep="\t"); e["role"] = e["role"].fillna("null")
N = e[(e.role == "null") & e["mse_Mic"].notna()].copy().reset_index(drop=True)
snv = pd.read_csv(f"{S}/compare/SNV42_new_scores.tsv", sep="\t")
ga = pd.read_csv(f"{SUP}/data/gene_annot_hg38"); ga["tss"] = np.where(ga.strand == "+", ga.start, ga.end)
TSS = {c: np.sort(x.tss.values) for c, x in ga.groupby("chr")}
def near(c, p):
    t = TSS.get(c); i = np.searchsorted(t, p); return int(min(abs(p - t[max(i - 1, 0)]), abs(t[min(i, len(t) - 1)] - p)))
import pysam; fa = pysam.FastaFile(a.fa)
def gc(c, p):
    s = fa.fetch(c, max(0, p - 1000), p + 1000).upper(); return (s.count("G") + s.count("C")) / max(len(s) - s.count("N"), 1)
for d in (snv, N):
    d["ltss"] = np.log10(np.array([near(c, p) for c, p in zip(d.chrom, d.pos)]) + 100)
    d["gc"] = [gc(c, p) for c, p in zip(d.chrom, d.pos)]
sd_l, sd_g = N.ltss.std(), N.gc.std()
order = [(((N.ltss - r.ltss) / sd_l) ** 2 + ((N.gc - r.gc) / sd_g) ** 2).sort_values(kind="stable").index for r in snv.itertuples()]

res = snv[["var_id", "gene", "chrom", "pos", "ref", "alt"] + [f"{m}_{c}" for c in CELLS for m in MET]].copy()
log = [f"background: {len(N)} low-PP eQTL SNPs; matching on log10 TSS distance + GC (+/- 1 kb); 42 SNVs; BH across SNVs"]
for c in CELLS:
    for m in MET:
        col = f"{m}_{c}"; res[f"p_pool_{col}"] = [emp(x, N[col].values) for x in snv[col]]; res[f"q_pool_{col}"] = bh(res[f"p_pool_{col}"])
for n in a.n_match:
    for c in CELLS:
        for m in MET:
            col = f"{m}_{c}"
            res[f"p_n{n}_{col}"] = [emp(x, N.loc[ix[:n], col].values) for x, ix in zip(snv[col], order)]
            res[f"q_n{n}_{col}"] = bh(res[f"p_n{n}_{col}"])
def summarise(tag):
    for c in CELLS:
        both05 = (res[f"q_{tag}_mse_{c}"] < 0.05) & (res[f"q_{tag}_maxabs_{c}"] < 0.05)
        both10 = (res[f"q_{tag}_mse_{c}"] < 0.1) & (res[f"q_{tag}_maxabs_{c}"] < 0.1)
        r = res.set_index("var_id").loc["rs636317"]
        log.append(f"  {tag:6s} {c:6s} q<0.05 both: {int(both05.sum())} {list(res.var_id[both05])} | q<0.1 both: {int(both10.sum())} | "
                   f"rs636317 P mse {r[f'p_{tag}_mse_{c}']:.2g} q {r[f'q_{tag}_mse_{c}']:.2g}; P maxabs {r[f'p_{tag}_maxabs_{c}']:.2g} q {r[f'q_{tag}_maxabs_{c}']:.2g}")
summarise("pool")
for n in a.n_match: log.append(f"\n== matched n = {n} (min P = {1 / (n + 1):.1e})"); summarise(f"n{n}")
res.to_csv(f"{OUT}/SNV42_matched.tsv", sep="\t", index=False)
open(f"{OUT}/SNV42_matched_summary.txt", "w").write("\n".join(log) + "\n"); print("\n".join(log))
