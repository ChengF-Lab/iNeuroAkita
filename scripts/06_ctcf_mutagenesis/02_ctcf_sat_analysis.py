#!/usr/bin/env python
"""
02_ctcf_sat_analysis.py  --  CTCF saturation mutagenesis (Fig. 3), predicted 3D disruption vs motif PWM

Input : inputs/CTCF_sat/{ctcf_sat_master.tsv, sites.tsv} (01_prep_ctcf_sat.py), scores/CTCF_sat/<chunk>/<chunk>_scores (SuPreMo)
Score : per SNV and head, MSE and MaxAbs = mean of the 4 augmentations (shift -1/0/+1 + revcomp), as for all other variants
Panels (per head; compare/CTCF_sat/):
  A  position profile: mean log10 MSE per motif position for strong / weak / unbound / control sites, with the motif
     information content (bits) on a second axis; Pearson r between the strong-site profile and IC (19 positions)
  B  dLLR (PWM log-odds ALT - REF) vs log10 MSE, strong sites (each dot = one base change); Spearman rho; LOESS-free binned means
  C  per-site disruption (max over the site's SNVs) by class; Mann-Whitney U strong vs control, bound vs unbound
  D  per-site: ChIP signal vs max disruption (bound sites)
Outputs: ctcf_sat_scores.tsv (one row per SNV), position_summary.tsv, site_summary.tsv, stats.txt, Fig3_sat_<head>.{pdf,svg,png}
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    from scipy.stats import mannwhitneyu, spearmanr, pearsonr
except ImportError:
    mannwhitneyu = spearmanr = pearsonr = None

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--aug", default="all", choices=["0", "all"])
ap.add_argument("--formats", default="pdf,svg,png")
a = ap.parse_args()
S = f"{a.base}/variant_scoring"; IN = f"{S}/inputs/CTCF_sat"; OUT = f"{S}/compare/CTCF_sat"; os.makedirs(OUT, exist_ok=True)
HEAD = {"HFF": "NPC", "H1hESC": "Mic", "GM12878": "Neuron"}; AUG = r"(0)" if a.aug == "0" else r"(-1|0|0_revcomp|1)"
CLS = [("strong", "#b2182b"), ("weak", "#ef8a62"), ("unbound", "#67a9cf"), ("control", "0.55")]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.6})

m = pd.read_csv(f"{IN}/ctcf_sat_master.tsv", sep="\t"); sites = pd.read_csv(f"{IN}/sites.tsv", sep="\t")
parts = []
for f in glob.glob(f"{S}/scores/CTCF_sat/*/*_scores"):
    x = pd.read_csv(f, sep="\t"); x["chunk"] = os.path.basename(f).replace("_scores", ""); parts.append(x)
raw = pd.concat(parts, ignore_index=True).set_index(["chunk", "var_index"])
m = m.set_index(["chunk", "var_index"])
for h, cell in HEAD.items():
    for met in ["mse", "maxabs"]:
        cols = [c for c in raw.columns if re.fullmatch(rf"{met}_{h}_{AUG}", c)]
        m[f"{met}_{cell}"] = raw[cols].mean(axis=1).reindex(m.index)
m = m.reset_index(); m.to_csv(f"{OUT}/ctcf_sat_scores.tsv", sep="\t", index=False)
log = [f"SNVs scored: {m.mse_Mic.notna().sum()}/{len(m)} (chunks {len(parts)}); aug = {a.aug}"]
def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if spearmanr: r, p = spearmanr(x[ok], y[ok]); return r, p
    return pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman"), np.nan
pos_rows, site_rows = [], []
for cell in HEAD.values():
    m[f"lmse_{cell}"] = np.log10(m[f"mse_{cell}"].clip(lower=1e-12))
    ps = m.groupby(["cls", "motif_pos"]).agg(lmse=(f"lmse_{cell}", "mean"), maxabs=(f"maxabs_{cell}", "mean"), ic=("ic", "first")).reset_index()
    ps["head"] = cell; pos_rows.append(ps)
    ss = m.groupby(["site_id", "cls"]).agg(max_mse=(f"mse_{cell}", "max"), mean_lmse=(f"lmse_{cell}", "mean"),
                                           chip=("chip", "first"), motif_score=("motif_score", "first")).reset_index()
    ss["head"] = cell; site_rows.append(ss)
    st = ps[ps.cls == "strong"].sort_values("motif_pos")
    r_ic = np.corrcoef(st.lmse, st.ic)[0, 1]
    s = m[m.cls == "strong"]; r_d, p_d = rho(s.dLLR.values, s[f"lmse_{cell}"].values)
    log.append(f"[{cell}] strong-site position profile vs IC: Pearson r = {r_ic:.2f} (n = {len(st)} positions); "
               f"dLLR vs log10 MSE (strong, n = {s[f'lmse_{cell}'].notna().sum()} SNVs): Spearman rho = {r_d:.2f}, P = {p_d:.1e}")
    for c1, c2 in [("strong", "control"), ("strong", "unbound"), ("weak", "control"), ("unbound", "control")]:
        x1, x2 = ss[ss.cls == c1].max_mse.dropna(), ss[ss.cls == c2].max_mse.dropna()
        p = mannwhitneyu(x1, x2, alternative="greater").pvalue if mannwhitneyu and len(x1) and len(x2) else np.nan
        log.append(f"[{cell}] per-site max MSE {c1} (median {x1.median():.2e}, n={len(x1)}) > {c2} (median {x2.median():.2e}, n={len(x2)}): "
                   f"one-sided Mann-Whitney P = {p:.1e}")
    # ---------------- figure
    fig, ax = plt.subplots(1, 4, figsize=(7.1, 1.9), gridspec_kw=dict(width_ratios=[1.5, 1.1, 0.9, 1.0], wspace=0.55))
    A = ax[0]; A2 = A.twinx()
    A2.bar(st.motif_pos, st.ic, color="0.88", width=0.8, zorder=0); A2.set_ylabel("motif IC (bits)", fontsize=6); A2.set_ylim(0, 2.2)
    A.set_zorder(A2.get_zorder() + 1); A.patch.set_visible(False)
    for c, col in CLS:
        q = ps[ps.cls == c].sort_values("motif_pos")
        if len(q): A.plot(q.motif_pos, q.lmse, "-o", c=col, ms=2, lw=0.8, label=c)
    A.set_xlabel("motif position"); A.set_ylabel("mean log10 MSE"); A.legend(frameon=False, fontsize=5.3, ncol=2, loc="upper left")
    A.set_title(f"r(strong, IC) = {r_ic:.2f}", fontsize=6.3)
    B = ax[1]
    B.scatter(s.dLLR, s[f"lmse_{cell}"], s=2, c="#b2182b", alpha=0.25, lw=0, rasterized=True)
    bins = np.linspace(s.dLLR.min(), s.dLLR.max(), 12); mid = (bins[1:] + bins[:-1]) / 2
    mb = s.groupby(pd.cut(s.dLLR, bins), observed=False)[f"lmse_{cell}"].mean().values
    B.plot(mid, mb, c="k", lw=1); B.set_xlabel("PWM ΔLLR (ALT − REF)"); B.set_ylabel("log10 MSE")
    B.set_title(f"strong sites, ρ = {r_d:.2f}", fontsize=6.3)
    Cx = ax[2]; data = [ss[ss.cls == c].max_mse.dropna().pipe(np.log10) for c, _ in CLS]
    bp = Cx.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=0.8))
    for b_, (_, col) in zip(bp["boxes"], CLS): b_.set_facecolor(col); b_.set_linewidth(0.5)
    Cx.set_xticks(range(1, 5)); Cx.set_xticklabels([c for c, _ in CLS], rotation=35, ha="right", fontsize=6)
    Cx.set_ylabel("per-site max log10 MSE")
    D = ax[3]; bb = ss[ss.cls.isin(["strong", "weak"])]
    D.scatter(np.log10(bb.chip.clip(lower=0.1)), np.log10(bb.max_mse), s=4, c=[dict(CLS)[c] for c in bb.cls], lw=0)
    r_c, _ = rho(np.log10(bb.chip.clip(lower=0.1)).values, np.log10(bb.max_mse).values)
    D.set_xlabel("log10 NPC CTCF ChIP"); D.set_ylabel("per-site max log10 MSE"); D.set_title(f"bound sites, ρ = {r_c:.2f}", fontsize=6.3)
    for x_ in ax:
        for s_ in ["top", "right"]: x_.spines[s_].set_visible(False)
    A2.spines["top"].set_visible(False)
    fig.suptitle(f"CTCF saturation mutagenesis, {cell} head", fontsize=7, x=0.01, ha="left", y=1.02)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.28, top=0.86)
    for f in a.formats.split(","): fig.savefig(f"{OUT}/Fig3_sat_{cell}.{f}", dpi=300, bbox_inches="tight")
    plt.close(fig)
pd.concat(pos_rows).to_csv(f"{OUT}/position_summary.tsv", sep="\t", index=False)
pd.concat(site_rows).to_csv(f"{OUT}/site_summary.tsv", sep="\t", index=False)
open(f"{OUT}/stats.txt", "w").write("\n".join(log) + "\n"); print("\n".join(log))
