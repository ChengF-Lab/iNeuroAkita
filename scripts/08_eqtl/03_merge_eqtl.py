#!/usr/bin/env python
"""
03_merge_eqtl.py  --  merge eQTL chunk scores, empirical P vs matched null, enrichment

Per SNP and head (NPC / Mic / Neuron): MSE, MaxAbs, L1 = mean of the 4 augmentations (--aug 0: shift 0 only, sensitivity)
Empirical P (per target, per head): p = (1 + #null SNPs with MSE >= target MSE) / (N_null + 1), pooled within-gene null;
                                    BH-FDR across targets
Locus-matched effect: log10(MSE_target) - log10(median MSE of its own matched null SNPs)
Enrichment: Wilcoxon signed-rank of the locus-matched effect (all targets; and by PP bin 0.1-0.5 / 0.5-0.9 / >0.9)
Outputs: variant_scoring/compare/eQTL/{eQTL_scores.tsv, eQTL_targets_stats.tsv, summary.txt, eQTL_disruption_vs_P.pdf/png}
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--head", default="Mic", help="head used for the main ranking/figure")
ap.add_argument("--aug", default="all", choices=["0", "all"], help="all = mean of shift -1/0/+1 + revcomp (default); 0 = shift-0 only (sensitivity)")
a = ap.parse_args()
B = a.base; S = f"{B}/variant_scoring"; OUT = f"{S}/compare/eQTL"; os.makedirs(OUT, exist_ok=True)
HEAD = {"HFF": "NPC", "H1hESC": "Mic", "GM12878": "Neuron"}; AUG = r"(0)" if a.aug == "0" else r"(-1|0|0_revcomp|1)"; log = []

m = pd.read_csv(f"{S}/inputs/eQTL/eQTL_master.tsv", sep="\t")
m["role"] = m["role"].fillna("null")   # pandas reads the string "null" as NaN
nm = pd.read_csv(f"{S}/inputs/eQTL/eQTL_null_map.tsv", sep="\t")
def score_alleles(master, score_dir):
    """shift 0 (or mean of the 4 augmentations with --aug all), one row per (SNP, ALT) of a master table"""
    parts = []
    for f in glob.glob(f"{score_dir}/*/*_scores"):
        x = pd.read_csv(f, sep="\t"); x["chunk"] = os.path.basename(f).replace("_scores", ""); parts.append(x)
    if not parts: return None, 0
    raw = pd.concat(parts, ignore_index=True).set_index(["chunk", "var_index"])
    r = master.set_index(["chunk", "var_index"])[["var_id", "alt"]].copy()
    for h, cell in HEAD.items():
        for met in ["mse", "maxabs", "l1"]:
            cols = [c for c in raw.columns if re.fullmatch(rf"{met}_{h}_{AUG}", c)]
            r[f"{met}_{cell}"] = raw[cols].mean(axis=1).reindex(r.index)
    return r.reset_index(drop=True), len(parts)
L1, n1 = score_alleles(m, f"{S}/scores/eQTL")
alt_master = f"{S}/inputs/eQTL_alt/eQTL_alt_master.tsv"
L2, n2 = (None, 0)
if os.path.exists(alt_master):
    ma = pd.read_csv(alt_master, sep="\t"); L2, n2 = score_alleles(ma, f"{S}/scores/eQTL_alt")
L = pd.concat([L1] + ([L2] if L2 is not None else []), ignore_index=True)
L.to_csv(f"{OUT}/eQTL_scores_by_allele.tsv", sep="\t", index=False)
log.append(f"alleles scored: first ALT {L1.mse_Mic.notna().sum()} (chunks {n1}); extra ALTs {0 if L2 is None else int(L2.mse_Mic.notna().sum())} (chunks {n2})")
# worst-case allele per SNP and head; null SNPs treated identically
res = m.drop(columns=["alt"]).copy()
for cell in HEAD.values():
    ok = L[L[f"mse_{cell}"].notna()]
    best = ok.loc[ok.groupby("var_id")[f"mse_{cell}"].idxmax(), ["var_id", "alt", f"mse_{cell}", f"maxabs_{cell}", f"l1_{cell}"]]
    best = best.rename(columns={"alt": f"alt_max_{cell}"})
    res = res.merge(best, on="var_id", how="left")
res["n_alleles_scored"] = res.var_id.map(L[L.mse_Mic.notna()].groupby("var_id").size()).fillna(0).astype(int)
done = res[f"mse_{a.head}"].notna()
log.append(f"scored: {done.sum()}/{len(res)} SNPs  (targets {int((done & (res.role=='target')).sum())}/{int((res.role=='target').sum())}, "
           f"null {int((done & (res.role=='null')).sum())}/{int((res.role=='null').sum())})")
res.to_csv(f"{OUT}/eQTL_scores.tsv", sep="\t", index=False)

T = res[(res.role == "target") & done].copy(); N = res[(res.role == "null") & done].copy()
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
for cell in HEAD.values():
    nv = np.sort(N[f"mse_{cell}"].values)
    T[f"p_emp_{cell}"] = (1 + len(nv) - np.searchsorted(nv, T[f"mse_{cell}"].values, side="left")) / (len(nv) + 1)
    T[f"q_{cell}"] = bh(T[f"p_emp_{cell}"])
# locus-matched effect
med = (nm.merge(N[["var_id", f"mse_{a.head}"]], left_on="null_id", right_on="var_id")
         .groupby("target_id")[f"mse_{a.head}"].median().rename("null_median"))
T = T.merge(med, left_on="var_id", right_index=True, how="left")
T["log10_ratio_vs_locus_null"] = np.log10(T[f"mse_{a.head}"]) - np.log10(T.null_median)
T["PP_bin"] = pd.cut(T.PostProb_max, [0.1, 0.5, 0.9, 1.0001], labels=["0.1-0.5", "0.5-0.9", ">0.9"], right=False)

log.append(f"\n== empirical P ({a.head} head), N null = {len(N)}, minimum attainable P = {1/(len(N)+1):.1e}")
for thr in [0.05, 0.1]:
    log.append(f"  q < {thr}: " + ", ".join(f"{c} {int((T[f'q_{c}'] < thr).sum())}" for c in HEAD.values()))
log.append("\n== enrichment: target vs its own within-gene null (log10 MSE ratio; Wilcoxon signed-rank)")
for lab, x in [("all", T)] + [(str(b), T[T.PP_bin == b]) for b in T.PP_bin.cat.categories]:
    v = x.log10_ratio_vs_locus_null.dropna()
    if len(v) > 10:
        log.append(f"  PP {lab:8s} n={len(v):6d}  median log10 ratio = {v.median():+.3f}  "
                   f"frac target > null median = {(v > 0).mean():.2f}  P = {wilcoxon(v).pvalue:.1e}")
top = T.sort_values(f"mse_{a.head}", ascending=False)
cols = ["var_id", "genes", "PostProb_max", f"mse_{a.head}", "null_median", f"p_emp_{a.head}", f"q_{a.head}", "mse_NPC", "mse_Neuron", "fold", f"alt_max_{a.head}", "n_alleles_scored"]
log.append(f"\n== top 20 targets by MSE ({a.head})\n" + top[cols].head(20).to_string(index=False))
T["rank_new"] = T[f"mse_{a.head}"].rank(ascending=False)

T.to_csv(f"{OUT}/eQTL_targets_stats.tsv", sep="\t", index=False)

# ---- disruption vs empirical significance
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
x = np.log10(T[f"mse_{a.head}"]); y = -np.log10(T[f"p_emp_{a.head}"])
sc = ax[0].scatter(x, y, s=4 + 40 * T.PostProb_max, c=T.PostProb_max, cmap="viridis", alpha=.7, lw=0)
for r in top.head(8).itertuples():
    ax[0].annotate(f"{r.var_id} ({str(r.genes).split(';')[0]})", (np.log10(getattr(r, f"mse_{a.head}")), -np.log10(getattr(r, f"p_emp_{a.head}"))), fontsize=6)
qline = T.loc[T[f"q_{a.head}"] < 0.05, f"p_emp_{a.head}"]
if len(qline): ax[0].axhline(-np.log10(qline.max()), ls="--", c="grey", lw=.7)
ax[0].set_xlabel(f"log10 disruption (MSE, {a.head} head)"); ax[0].set_ylabel("-log10 empirical P"); plt.colorbar(sc, ax=ax[0], label="PostProb")
ax[1].hist(np.log10(N[f"mse_{a.head}"]), bins=80, color="#BDBDBD", density=True, label=f"within-gene null (n={len(N)})")
ax[1].hist(x, bins=80, color="#73D07A", alpha=.6, density=True, label=f"fine-mapped eQTL (n={len(T)})")
ax[1].set_xlabel(f"log10 MSE ({a.head})"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/eQTL_disruption_vs_P.pdf"); fig.savefig(f"{OUT}/eQTL_disruption_vs_P.png", dpi=200)
open(f"{OUT}/summary.txt", "w").write("\n".join(log) + "\n"); print("\n".join(log))
