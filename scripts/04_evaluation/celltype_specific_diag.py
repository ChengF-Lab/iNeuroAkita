#!/usr/bin/env python3
"""
celltype_specific_diag.py — step 0 of the genome-wide cell-type-specific contact analysis:
distributions used to CHOOSE the thresholds (nothing is called specific here).

Model and data (out-of-fold everywhere: every chromosome is predicted by the A4 fold that held it out)
  heldout      A4 <foldK>/predictOut_v3/hold_test/*_predictions_targets.npz, the 7 folds' test
               chromosomes = whole genome; targets NPC_P16 / Neuron_P16 / Mic_AD
  independent  A4 <foldK>/predictOut_indep/indep_{NPC,neuron,Mic}/*_predictions_targets.npz, only the
               windows on foldK's test chromosomes; targets NPC_ctrl / Neuron_P35 / Mic_ctrl.
               Each sample npz supplies its own head's prediction and its own target (the npz may hold
               only that head); windows are aligned by (chrom, start) across the three files.
Each map is sigma-smoothed exactly as add_smoothed_metrics.py (NaN-aware Gaussian, both maps).

Quantities
  window   r, MSE, SD of target / prediction per window x cell (full upper triangle, offset d) —
           the per-cell window filter of the next step is r(focal cell) > r_thr
  pixel    a random subsample (--frac of pixels with distance >= --min_dist) holding the three
           predictions, the three targets, distance and the three window r values:
             pred value per head         -> signal gate (head quantile or absolute)
             spec = pred_focal - max(pred_other two)        -> delta
             tspec = same on the targets -> concordance (observed difference in the same direction)

Outputs (--out, default model_trainOut/celltype_specific/diag)
  per_window_r.csv.gz         dataset, fold, chrom, start, end, cell, r, mse, target_sd, pred_sd
  T0a_window_r_summary.csv    quantiles of r and fraction of windows above 0.2 / 0.3 / 0.4
  T0b_pred_quantiles.csv      pixel prediction / target quantiles per head, all windows and r(cell) > r_thr,
                              overall and by distance band
  T0c_spec_quantiles.csv      quantiles of spec (and tspec) per focal cell, + pairwise differences
  T0d_candidate_grid.csv      for r_thr x signal gate x delta: estimated genome-wide pixel count,
                              % of pixels, and % with tspec > 0 (concordance) — the table to choose from
  pixel_subsample.npz         the subsample, for further exploration
  figures: D1 window r histograms, D2 r vs target SD, D3 prediction densities, D4 spec densities,
           D5 concordance vs delta  (.pdf .png)

Usage
  python celltype_specific_diag.py --base_dir $base_DIR                 # both datasets
  python celltype_specific_diag.py --base_dir $base_DIR --datasets heldout --frac 0.05
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_smoothed_metrics import vec_to_mat, nan_gauss

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8})

FOLD_TEST = {"fold1": ["10", "17"], "fold2": ["8", "9", "21"], "fold3": ["1", "11", "20"],
             "fold4": ["2", "3", "19"], "fold5": ["4", "5", "18", "22"], "fold6": ["6", "7", "12", "13"],
             "fold7": ["14", "15", "16", "X"]}
TRUNK = "model_trainOut/NPC_Mic_Neuron_out_trunk"
CELLS = ["NPC", "Neuron", "Mic"]
HEAD = {"NPC": "NPC_P16", "Neuron": "Neuron_P16", "Mic": "Mic_AD"}            # model heads
OBS = {"heldout": HEAD,
       "independent": {"NPC": "NPC_ctrl", "Neuron": "Neuron_P35", "Mic": "Mic_ctrl"}}
INDEP_KEY = {"NPC": "NPC", "Neuron": "neuron", "Mic": "Mic"}
COL = {"NPC": "#2a9d8f", "Neuron": "#e9a23b", "Mic": "#8e5bd0"}
BIN = 2048
QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99, 0.999]
DIST_BANDS = [(0, 100_000), (100_000, 300_000), (300_000, 1_000_000)]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def cnum(c):
    return str(c).replace("chr", "")


def first(pattern):
    f = sorted(glob.glob(pattern))
    return f[0] if f else None


def save(fig, prefix):
    for ext in ("pdf", "png"):
        fig.savefig(f"{prefix}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    log(f"✅ {prefix}.pdf/.png")


# ----------------------------------------------------------------------------- loading
def fold_inputs(base, dataset, fold):
    """-> (P [n,len,3 heads in CELLS order], {cell: T [n,len]}, meta DataFrame, L, d) for the fold's
    test chromosomes, or None."""
    keep_c = set(FOLD_TEST[fold])
    if dataset == "heldout":
        p = (first(f"{base}/{TRUNK}/{fold}/predictOut_v3/hold_test/trunk_{fold}_test_predictions_targets.npz")
             or first(f"{base}/{TRUNK}/{fold}/predictOut_v3/hold_test/*_predictions_targets.npz"))
        paths = {c: p for c in CELLS}
    else:
        paths = {c: first(f"{base}/{TRUNK}/{fold}/predictOut_indep/indep_{INDEP_KEY[c]}/*_predictions_targets.npz")
                 for c in CELLS}
    if any(v is None for v in paths.values()):
        log(f"⚠️  {dataset} {fold}: npz missing {paths}")
        return None
    # each cell's prediction (its own head) and target come from that cell's npz: the held-out npz holds
    # all three heads; an independent npz may hold only the head of its own sample.
    z0 = np.load(paths["NPC"], allow_pickle=False)
    chrom0 = np.array([str(c) for c in z0["chrom"]])
    rows = np.where(np.isin(np.array([cnum(c) for c in chrom0]), list(keep_c)))[0]
    if len(rows) == 0:
        log(f"⚠️  {dataset} {fold}: no windows on {sorted(keep_c)}")
        return None
    meta = pd.DataFrame(dict(chrom=chrom0[rows], start=z0["start"].astype(int)[rows],
                             end=z0["end"].astype(int)[rows]))
    L, d = int(z0["matrix_len"]), int(z0["diagonal_offset"])
    key0 = list(zip(meta.chrom, meta.start))
    cache, P, T = {paths["NPC"]: z0}, [], {}
    for c in CELLS:
        zc = cache.get(paths[c]) or cache.setdefault(paths[c], np.load(paths[c], allow_pickle=False))
        heads = [str(h) for h in zc["model_head_names"]]
        obs = [str(o) for o in zc["obs_target_names"]]
        if HEAD[c] not in heads:
            raise SystemExit(f"❌ {paths[c]}: head {HEAD[c]} not in {heads}")
        if OBS[dataset][c] not in obs:
            raise SystemExit(f"❌ {paths[c]}: target {OBS[dataset][c]} not in {obs}")
        pos = {k: i for i, k in enumerate(zip([str(x) for x in zc["chrom"]], zc["start"].astype(int)))}
        miss = [k for k in key0 if k not in pos]
        if miss:
            raise SystemExit(f"❌ {paths[c]}: {len(miss)} windows of {paths['NPC']} missing, e.g. {miss[:3]}")
        rc = np.array([pos[k] for k in key0])          # same windows, aligned by (chrom, start)
        P.append(zc["test_preds_subset"][rc][:, :, heads.index(HEAD[c])])
        T[c] = zc["test_targets"][rc][:, :, obs.index(OBS[dataset][c])]
    P = np.stack(P, axis=2)                              # [n, len, 3] in CELLS order
    del cache
    log(f"   {dataset} {fold}: {len(rows)} windows on chr{','.join(sorted(keep_c, key=lambda x: (len(x), x)))}")
    return P, T, meta, L, d


def r_mse(p, t):
    k = np.isfinite(p) & np.isfinite(t)
    if k.sum() < 3 or p[k].std() == 0 or t[k].std() == 0:
        return np.nan, np.nan
    return float(np.corrcoef(p[k], t[k])[0, 1]), float(np.mean((p[k] - t[k]) ** 2))


def collect(base, datasets, sigma, frac, min_dist, seed):
    rng = np.random.default_rng(seed)
    wins, pix = [], []
    for ds in datasets:
        for fold in FOLD_TEST:
            got = fold_inputs(base, ds, fold)
            if got is None:
                continue
            P, T, meta, L, d = got
            iu = np.triu_indices(L, d)
            dist = (iu[1] - iu[0]) * BIN
            far = dist >= min_dist
            for w in range(len(meta)):
                sp, st, rr = {}, {}, {}
                for k, c in enumerate(CELLS):
                    sp[c] = nan_gauss(vec_to_mat(P[w, :, k].astype(float), L, d), sigma)[iu]
                    st[c] = nan_gauss(vec_to_mat(T[c][w].astype(float), L, d), sigma)[iu]
                    r, m = r_mse(sp[c], st[c])
                    rr[c] = r
                    wins.append(dict(dataset=ds, fold=fold, chrom=meta.chrom[w], start=meta.start[w],
                                     end=meta.end[w], cell=c, r=r, mse=m,
                                     target_sd=float(np.nanstd(st[c])), pred_sd=float(np.nanstd(sp[c])),
                                     frac_target_nan=float(np.mean(~np.isfinite(st[c])))))
                sel = far & (rng.random(len(dist)) < frac)
                n = int(sel.sum())
                if n == 0:
                    continue
                block = {"dataset": np.full(n, ds), "dist": dist[sel].astype(np.int32)}
                for c in CELLS:
                    block[f"pred_{c}"] = sp[c][sel].astype(np.float32)
                    block[f"tgt_{c}"] = st[c][sel].astype(np.float32)
                    block[f"r_{c}"] = np.full(n, rr[c], np.float32)
                pix.append(block)
            del P, T
    if not wins:
        raise SystemExit("❌ no data found")
    W = pd.DataFrame(wins)
    X = pd.DataFrame({k: np.concatenate([b[k] for b in pix]) for k in pix[0]})
    for c in CELLS:
        others = [o for o in CELLS if o != c]
        X[f"spec_{c}"] = X[f"pred_{c}"] - X[[f"pred_{o}" for o in others]].max(axis=1)
        X[f"tspec_{c}"] = X[f"tgt_{c}"] - X[[f"tgt_{o}" for o in others]].max(axis=1)
    return W, X


# ----------------------------------------------------------------------------- tables
def qrow(v, **lab):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    row = dict(lab, n=len(v), mean=v.mean() if len(v) else np.nan, sd=v.std() if len(v) else np.nan)
    row.update({f"q{q * 100:g}": (np.quantile(v, q) if len(v) else np.nan) for q in QS})
    return row


def tables(W, X, r_thrs, r_main, deltas, frac, out):
    # T0a window r
    rows = []
    for (ds, c), g in W.groupby(["dataset", "cell"]):
        row = qrow(g.r, dataset=ds, cell=c)
        row.update({f"frac_r>{t:g}": float((g.r > t).mean()) for t in r_thrs})
        row["spearman_r_vs_target_sd"] = g[["r", "target_sd"]].corr("spearman").iloc[0, 1]
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{out}/T0a_window_r_summary.csv", index=False, float_format="%.4g")

    # T0b prediction / target quantiles per head
    rows = []
    for ds, g in X.groupby("dataset"):
        for c in CELLS:
            for sub, gg in [("all", g), (f"r_{c}>{r_main:g}", g[g[f"r_{c}"] > r_main])]:
                for band in [None] + DIST_BANDS:
                    h = gg if band is None else gg[(gg.dist >= band[0]) & (gg.dist < band[1])]
                    bl = "all" if band is None else f"{band[0] // 1000}-{band[1] // 1000}kb"
                    rows.append(qrow(h[f"pred_{c}"], dataset=ds, cell=c, value="pred", windows=sub, dist=bl))
                    rows.append(qrow(h[f"tgt_{c}"], dataset=ds, cell=c, value="target", windows=sub, dist=bl))
    pd.DataFrame(rows).to_csv(f"{out}/T0b_pred_quantiles.csv", index=False, float_format="%.4g")

    # T0c spec quantiles + pairwise
    rows = []
    for ds, g in X.groupby("dataset"):
        for c in CELLS:
            gg = g[g[f"r_{c}"] > r_main]
            rows.append(qrow(g[f"spec_{c}"], dataset=ds, quantity=f"spec_{c}", windows="all"))
            rows.append(qrow(gg[f"spec_{c}"], dataset=ds, quantity=f"spec_{c}", windows=f"r_{c}>{r_main:g}"))
            rows.append(qrow(g[f"tspec_{c}"], dataset=ds, quantity=f"tspec_{c}", windows="all"))
        for a, b in [("NPC", "Neuron"), ("NPC", "Mic"), ("Neuron", "Mic")]:
            rows.append(qrow(g[f"pred_{a}"] - g[f"pred_{b}"], dataset=ds, quantity=f"pred_{a}-{b}", windows="all"))
            rows.append(qrow(np.abs(g[f"pred_{a}"] - g[f"pred_{b}"]), dataset=ds,
                             quantity=f"|pred_{a}-{b}|", windows="all"))
    pd.DataFrame(rows).to_csv(f"{out}/T0c_spec_quantiles.csv", index=False, float_format="%.4g")

    # T0d candidate grid
    rows = []
    for ds, g in X.groupby("dataset"):
        n_all = len(g)
        for c in CELLS:
            qpred = {q: np.quantile(g[f"pred_{c}"], q) for q in (0.8, 0.9, 0.95)}
            qspec = {q: np.quantile(g[f"spec_{c}"], q) for q in (0.9, 0.95, 0.99)}
            gates = [("none", -np.inf)] + [(f"pred>q{q * 100:g} ({v:.3f})", v) for q, v in qpred.items()] \
                + [("pred>0.3 abs", 0.3)]
            dl = [(f"{x:g}", x) for x in deltas] + [(f"spec q{q * 100:g} ({v:.3f})", v) for q, v in qspec.items()]
            for rt in [None] + list(r_thrs):
                base_m = np.ones(n_all, bool) if rt is None else (g[f"r_{c}"] > rt).values
                for gname, gv in gates:
                    m1 = base_m & (g[f"pred_{c}"].values > gv)
                    for dname, dv in dl:
                        m = m1 & (g[f"spec_{c}"].values > dv)
                        ts = g[f"tspec_{c}"].values[m]
                        ts = ts[np.isfinite(ts)]
                        rows.append(dict(dataset=ds, focal=c, r_thr="none" if rt is None else rt,
                                         signal_gate=gname, delta=dname, n_sampled=int(m.sum()),
                                         est_pixels_genome=int(round(m.sum() / frac)),
                                         pct_of_pixels=100 * m.sum() / n_all,
                                         pct_tspec_pos=100 * np.mean(ts > 0) if len(ts) else np.nan,
                                         median_tspec=np.median(ts) if len(ts) else np.nan))
    pd.DataFrame(rows).to_csv(f"{out}/T0d_candidate_grid.csv", index=False, float_format="%.4g")


# ----------------------------------------------------------------------------- figures
def figures(W, X, r_main, out):
    dss = list(W.dataset.unique())
    # D1 window r histograms
    fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 3), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        for c in CELLS:
            v = W[(W.dataset == ds) & (W.cell == c)].r.dropna()
            ax.hist(v, bins=np.linspace(-0.2, 1, 61), histtype="step", lw=1.3, color=COL[c],
                    label=f"{c} (median {v.median():.2f}; {100 * (v > r_main).mean():.0f}% > {r_main:g})")
        ax.axvline(r_main, color="grey", ls="--", lw=0.8)
        ax.set(title=ds, xlabel="window Pearson r (σ-smoothed)", ylabel="windows")
        ax.legend(frameon=False, fontsize=7)
    save(fig, f"{out}/D1_window_r_hist")
    # D2 r vs target SD
    fig, axes = plt.subplots(len(dss), 3, figsize=(10, 3 * len(dss)), squeeze=False)
    for i, ds in enumerate(dss):
        for j, c in enumerate(CELLS):
            g = W[(W.dataset == ds) & (W.cell == c)].dropna(subset=["r", "target_sd"])
            ax = axes[i, j]
            ax.hexbin(g.target_sd, g.r, gridsize=40, mincnt=1, cmap="Greys")
            ax.axhline(r_main, color=COL[c], ls="--", lw=0.8)
            rho = g[["r", "target_sd"]].corr("spearman").iloc[0, 1]
            ax.set(title=f"{ds} {c}  (Spearman ρ = {rho:.2f})", xlabel="target SD in window (σ-smoothed)",
                   ylabel="window r")
    fig.tight_layout()
    save(fig, f"{out}/D2_window_r_vs_target_sd")
    # D3 prediction densities
    fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 3), squeeze=False)
    bins = np.linspace(-1.5, 2, 141)
    for ax, ds in zip(axes[0], dss):
        g = X[X.dataset == ds]
        for c in CELLS:
            ax.hist(g[f"pred_{c}"], bins=bins, density=True, histtype="step", lw=1.2, color=COL[c],
                    label=f"{c} all (q90 {np.quantile(g[f'pred_{c}'], .9):.2f})")
            gg = g[g[f"r_{c}"] > r_main]
            ax.hist(gg[f"pred_{c}"], bins=bins, density=True, histtype="step", lw=1, ls="--", color=COL[c],
                    label=f"{c} r>{r_main:g}")
        ax.axvline(0.3, color="grey", ls=":", lw=0.8)
        ax.set(title=f"{ds}: predicted value per pixel", xlabel="prediction (σ-smoothed)", ylabel="density")
        ax.legend(frameon=False, fontsize=6)
    save(fig, f"{out}/D3_pred_density")
    # D4 spec densities
    fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 3), squeeze=False)
    bins = np.linspace(-1.5, 1.5, 121)
    for ax, ds in zip(axes[0], dss):
        g = X[X.dataset == ds]
        for c in CELLS:
            v = g[f"spec_{c}"]
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.2, color=COL[c],
                    label=f"{c} (q95 {np.quantile(v, .95):.2f}, q99 {np.quantile(v, .99):.2f})")
        ax.set_yscale("log")
        ax.set(title=f"{ds}: pred(focal) − max(pred others)", xlabel="spec", ylabel="density (log)")
        ax.legend(frameon=False, fontsize=6)
    save(fig, f"{out}/D4_spec_density")
    # D5 concordance vs delta (focal r > r_main, no signal gate / pred > q90)
    deltas = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 3), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        g = X[X.dataset == ds]
        for c in CELLS:
            gg = g[g[f"r_{c}"] > r_main]
            for ls, gate in [("-", -np.inf), ("--", np.quantile(g[f"pred_{c}"], 0.9))]:
                h = gg[gg[f"pred_{c}"] > gate]
                y = []
                for dl in deltas:
                    t = h.loc[h[f"spec_{c}"] > dl, f"tspec_{c}"].dropna()
                    y.append(100 * (t > 0).mean() if len(t) >= 50 else np.nan)
                ax.plot(deltas, y, ls=ls, color=COL[c], lw=1.2,
                        label=f"{c}{'' if gate == -np.inf else ' + pred>q90'}")
        bg = np.nanmean([100 * (g[f"tspec_{c}"].dropna() > 0).mean() for c in CELLS])
        ax.axhline(bg, color="grey", lw=0.7, ls=":")
        ax.set(title=f"{ds}: observed same direction (r_focal>{r_main:g})", xlabel="δ (spec threshold)",
               ylabel="% pixels with target spec > 0", ylim=(0, 100))
        ax.legend(frameon=False, fontsize=6)
    save(fig, f"{out}/D5_concordance_vs_delta")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out", default=None, help="default <base>/model_trainOut/celltype_specific/diag")
    ap.add_argument("--datasets", nargs="+", default=["heldout", "independent"], choices=["heldout", "independent"])
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--frac", type=float, default=0.02, help="pixel subsample fraction")
    ap.add_argument("--min_dist", type=int, default=20_000, help="min contact distance (bp) for pixels")
    ap.add_argument("--r_thr", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    ap.add_argument("--r_main", type=float, default=0.3, help="r threshold used in the figures")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    base = a.base_dir.rstrip("/")
    out = a.out or f"{base}/model_trainOut/celltype_specific/diag"
    os.makedirs(out, exist_ok=True)

    W, X = collect(base, a.datasets, a.sigma, a.frac, a.min_dist, a.seed)
    W.to_csv(f"{out}/per_window_r.csv.gz", index=False, float_format="%.5g")
    np.savez_compressed(f"{out}/pixel_subsample.npz", **{k: (X[k].to_numpy() if pd.api.types.is_numeric_dtype(X[k]) else X[k].to_numpy(dtype="U"))
                           for k in X.columns},
                        frac=a.frac, sigma=a.sigma, min_dist=a.min_dist)
    log(f"   {len(W) // 3} windows, {len(X):,} sampled pixels")
    tables(W, X, a.r_thr, a.r_main, a.deltas, a.frac, out)
    figures(W, X, a.r_main, out)
    log(f"✅ diagnostics in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
