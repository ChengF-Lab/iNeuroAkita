#!/usr/bin/env python
"""
04_ctcf_threshold_sensitivity.py  --  robustness of the 'bound vs unbound' result (Fig. 3c) to the ChIP thresholds and covariates

Per-site disruption = log10 of the max MSE over the site's SNVs (4-augmentation mean), motif-bearing sites only.
OLS: log10 max MSE ~ bound + GC + motif score + fold dummies (fold-change = 10^beta, normal-approx. P); one-sided Mann-Whitney too.
Definitions tested (ChIP = max NPC CTCF ChIP-seq signal within motif +/- 50 bp, as in 01_prep_ctcf_sat.py):
  default          bound = strong + weak (ChIP >= 5)           unbound = ChIP < 1
  strict_both      bound = ChIP >= 50                          unbound = ChIP < 0.5
  strong_only      bound = strong (ChIP >= 245.6)              unbound = ChIP < 1
  weak_only        bound = weak (5-161.8)                      unbound = ChIP < 1
  no_fold7         default, fold 7 removed (where unbound sites cluster)
  no_covariates    default, no GC / motif score / fold terms
  continuous       log10 max MSE ~ log10(ChIP + 1) + GC + motif score + fold, all motif sites (beta per 10-fold ChIP)
Output: compare/CTCF_sat/sensitivity/threshold_sensitivity.tsv
"""
import argparse, math
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--base", default=".", help="project directory (layout: see README)"); a = ap.parse_args()
S = f"{a.base}/variant_scoring"
sites = pd.read_csv(f"{S}/inputs/CTCF_sat/sites.tsv", sep="\t"); ss = pd.read_csv(f"{S}/compare/CTCF_sat/site_summary.tsv", sep="\t")
def P(t): return math.erfc(abs(t) / math.sqrt(2))
def mw(x, y):
    x, y = np.asarray(x), np.asarray(y); n1, n2 = len(x), len(y); r = pd.Series(np.r_[x, y]).rank().values; U = r[:n1].sum() - n1 * (n1 + 1) / 2
    _, c = np.unique(np.r_[x, y], return_counts=True); N = n1 + n2; sd = math.sqrt(n1 * n2 / 12 * ((N + 1) - (c ** 3 - c).sum() / (N * (N - 1))))
    return U / (n1 * n2), 0.5 * math.erfc(((U - n1 * n2 / 2) / sd) / math.sqrt(2))
def ols(d, x, cov=True):
    X = pd.DataFrame({x: d[x].values}, index=d.index)
    if cov: X = pd.concat([X, pd.get_dummies(d.fold, drop_first=True).astype(float), d[["gc", "motif_score"]]], axis=1)
    X["c"] = 1.0; A = X.values.astype(float); b = np.linalg.lstsq(A, d.l.values, rcond=None)[0]; res = d.l.values - A @ b
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * (res ** 2).sum() / (len(res) - A.shape[1])); return b[0], se[0]
rows = []
for H in ["NPC", "Mic", "Neuron"]:
    t = ss[ss["head"] == H].drop(columns=["chip", "motif_score"], errors="ignore").merge(sites[["site_id", "chip", "gc", "motif_score", "fold"]], on="site_id")
    t = t[(t.cls != "control") & t.max_mse.notna()].copy(); t["l"] = np.log10(t.max_mse)
    D = {"default": (t.cls.isin(["strong", "weak"]), t.chip < 1, t, True),
         "strict_both": (t.chip >= 50, t.chip < 0.5, t, True),
         "strong_only": (t.cls == "strong", t.chip < 1, t, True),
         "weak_only": (t.cls == "weak", t.chip < 1, t, True),
         "no_fold7": (t.cls.isin(["strong", "weak"]), t.chip < 1, t[t.fold != "fold7"], True),
         "no_covariates": (t.cls.isin(["strong", "weak"]), t.chip < 1, t, False)}
    for name, (b_, u_, base, cov) in D.items():
        d = base[(b_ | u_).reindex(base.index, fill_value=False)].copy(); d["bound"] = b_.reindex(d.index).astype(float)
        be, se = ols(d, "bound", cov); auc, pmw = mw(d[d.bound == 1].l, d[d.bound == 0].l)
        rows.append(dict(head=H, definition=name, n_bound=int(d.bound.sum()), n_unbound=int((d.bound == 0).sum()), fold_change=10 ** be,
                         CI_lo=10 ** (be - 1.96 * se), CI_hi=10 ** (be + 1.96 * se), P_adj=P(be / se), AUC=auc, P_MW=pmw))
    d = t.copy(); d["lchip"] = np.log10(d.chip + 1); be, se = ols(d, "lchip")
    rows.append(dict(head=H, definition="continuous (per 10-fold ChIP)", n_bound=len(d), n_unbound=np.nan, fold_change=10 ** be,
                     CI_lo=10 ** (be - 1.96 * se), CI_hi=10 ** (be + 1.96 * se), P_adj=P(be / se), AUC=np.nan, P_MW=np.nan))
R = pd.DataFrame(rows); R.to_csv(f"{S}/compare/CTCF_sat/sensitivity/threshold_sensitivity.tsv", sep="\t", index=False)
pd.set_option("display.width", 200); print(R.round(4).to_string(index=False))
