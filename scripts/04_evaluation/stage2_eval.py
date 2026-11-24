#!/usr/bin/env python3
"""
stage2_eval.py — evaluation beyond whole-map Pearson r (distance bands, insulation, baselines).

  1  distance-stratified r / MSE      bands 0-100, 100-250, 250-500, >500 kb (+ all)
  2  insulation correlation and boundary recall / precision
  3  baselines   dist     : per-diagonal mean of the SAME cell's targets on the fold's TRAINING chromosomes
                            (distance only, no sequence)
                 obs_<c'> : the observed map of another cell type used as the prediction
                            (held-out: the other training cells; independent: the other external samples)
                 trainobs_<c'> (independent only, optional): observed TRAINING-cell maps on the v2 windows
                            (--train_targets_v2), e.g. Mic_AD observed -> Mic_ctrl ("same cell type,
                            other sample") and the other two training cells

Models: A4 (merged Akita-init, the fold that held the chromosome out) and A0 (Akita zero-shot, head selected
on the same fold's validation chromosomes). Every predictor is scored on the same windows and pixels.
Maps are sigma-smoothed as add_smoothed_metrics.py (NaN-aware Gaussian, both maps) for r / MSE;
insulation is computed on the unsmoothed maps (it is itself an average).

Insulation: for bin i, mean of the w x w square M[i-w:i, i+1:i+w+1] (NaN-aware, integral images), per
--ins_window w (bins of 2,048 bp). Boundaries = local minima of the window-z-scored profile with
prominence >= --prominence and spacing >= w; a boundary is recovered if the other profile has one within
+/- --tol bins. Recall = recovered observed boundaries / observed; precision = predicted boundaries with an
observed one nearby / predicted (summed over windows).

Outputs (--out, default model_trainOut/stage2_eval)
  per_window_bands.csv.gz      dataset, fold, chrom, start, cell, predictor, band, r, mse, n
  per_window_insulation.csv.gz dataset, fold, chrom, start, cell, predictor, w, ins_r, boundary counts
  S1_distance_bands.csv        median / mean r and MSE per dataset x cell x predictor x band,
                               + paired Wilcoxon A4 vs A0 per band (windows) and fold-level means
  S2_insulation.csv            median insulation r; boundary recall, precision, F1 (micro)
  S3_baselines.csv             band "all": every predictor, median r, A4 - predictor, Wilcoxon
  figures E1-E4 (.pdf .png .svg)

Usage
  python stage2_eval.py --base_dir $base_DIR
  python stage2_eval.py --base_dir $base_DIR --datasets heldout --ins_window 10 25 --tol 2
  python stage2_eval.py --base_dir $base_DIR --train_targets_v2 <npz with NPC_P16/Neuron_P16/Mic_AD on v2 windows>
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
from scipy import stats
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_smoothed_metrics import vec_to_mat, nan_gauss
from celltype_specific_diag import (fold_inputs, FOLD_TEST, CELLS, HEAD, INDEP_KEY, COL, BIN, cnum, first, log)

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8})
FOLD_VALID = {"fold1": ["8", "9"], "fold2": ["11", "20"], "fold3": ["2", "19"], "fold4": ["4", "18"],
              "fold5": ["6", "16"], "fold6": ["14", "15"], "fold7": ["10", "17"]}
ZS = "model_trainOut/akita_zeroshot"
BANDS = [("0-100kb", 0, 100_000), ("100-250kb", 100_000, 250_000), ("250-500kb", 250_000, 500_000),
         (">500kb", 500_000, 10 ** 9), ("all", 0, 10 ** 9)]
PCOL = {"A4": "#e34948", "A0": "#2a78d6", "dist": "#9e9e9e"}


def save(fig, prefix):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{prefix}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    log(f"✅ {prefix}.pdf/.png/.svg")


def pcolor(p):
    if p in PCOL:
        return PCOL[p]
    c = p.split("_")[-1]
    return COL.get(c, "#777777")


# ----------------------------------------------------------------------------- A0
def selected_heads(base):
    s = pd.read_csv(f"{base}/{ZS}/summary/akita0_selected_heads.csv")
    return {(r.fold, r.obs_target): r.model_head for r in s.itertuples()}


class A0Source:
    """A0 predictions aligned to a list of windows; independent npz cached (one per sample)."""

    def __init__(self, base):
        self.base, self.cache = base, {}

    def _load(self, path):
        if path not in self.cache and "/test/" in path:       # held-out: keep only the current fold
            for k in [k for k in self.cache if "/test/" in k]:
                self.cache.pop(k)
        if path not in self.cache:
            z = np.load(path, allow_pickle=False)
            self.cache[path] = dict(P=z["test_preds_subset"], heads=[str(h) for h in z["model_head_names"]],
                                    key={k: i for i, k in enumerate(zip([str(c) for c in z["chrom"]],
                                                                        z["start"].astype(int)))})
            log(f"   A0 loaded {path}")
        return self.cache[path]

    def get(self, dataset, fold, cell, head, keys):
        if dataset == "heldout":
            p = first(f"{self.base}/{ZS}/{fold}/test/*_predictions_targets.npz")
        else:
            p = first(f"{self.base}/{ZS}/indep_v2_{INDEP_KEY[cell]}/*_predictions_targets.npz")
        if p is None:
            return None
        z = self._load(p)
        if head not in z["heads"]:
            log(f"⚠️  A0 head {head} not in {p}")
            return None
        idx = [z["key"].get(k) for k in keys]
        if any(i is None for i in idx):
            log(f"⚠️  A0 {p}: {sum(i is None for i in idx)} windows missing")
            return None
        return z["P"][np.array(idx)][:, :, z["heads"].index(head)]


def load_train_targets_v2(path):
    if not path:
        return None
    z = np.load(path, allow_pickle=False)
    obs = [str(o) for o in z["obs_target_names"]]
    log(f"   training-cell targets on v2 windows: {path} {obs}")
    return dict(T=z["test_targets"], obs=obs,
                key={k: i for i, k in enumerate(zip([str(c) for c in z["chrom"]], z["start"].astype(int)))})


# ----------------------------------------------------------------------------- distance baseline
def diag_means(base, dataset):
    """per chromosome x cell: sum and count of raw target values per diagonal (whole dataset)."""
    acc = {}
    for fold in FOLD_TEST:
        got = fold_inputs(base, dataset, fold)
        if got is None:
            continue
        _, T, meta, L, d = got
        iu = np.triu_indices(L, d)
        k = iu[1] - iu[0]
        for c in CELLS:
            for w in range(len(meta)):
                v = T[c][w].astype(float)
                ok = np.isfinite(v)
                s = np.bincount(k[ok], weights=v[ok], minlength=L)
                n = np.bincount(k[ok], minlength=L)
                key = (cnum(meta.chrom[w]), c)
                if key in acc:
                    acc[key][0] += s
                    acc[key][1] += n
                else:
                    acc[key] = [s, n]
    return acc, L, d


def dist_profile(acc, fold, cell, L):
    excl = set(FOLD_TEST[fold]) | set(FOLD_VALID[fold])
    s, n = np.zeros(L), np.zeros(L)
    for (c, cc), (ss, nn) in acc.items():
        if cc == cell and c not in excl:
            s += ss
            n += nn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan)


# ----------------------------------------------------------------------------- insulation
def insulation(M, w):
    """mean of M[i-w:i, i+1:i+w+1] for every i (NaN-aware, via integral images)."""
    L = M.shape[0]
    ok = np.isfinite(M)
    S = np.zeros((L + 1, L + 1))
    C = np.zeros((L + 1, L + 1))
    S[1:, 1:] = np.where(ok, M, 0).cumsum(0).cumsum(1)
    C[1:, 1:] = ok.cumsum(0).cumsum(1)
    i = np.arange(w, L - w)
    r0, r1, c0, c1 = i - w, i, i + 1, i + w + 1

    def box(A):
        return A[r1, c1] - A[r0, c1] - A[r1, c0] + A[r0, c0]
    s, n = box(S), box(C)
    out = np.full(L, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[i] = np.where(n >= 0.5 * w * w, s / n, np.nan)
    return out


def boundaries(ins, w, prom):
    v = ins.copy()
    ok = np.isfinite(v)
    if ok.sum() < 3 * w or np.nanstd(v) == 0:
        return np.array([], int)
    v = (v - np.nanmean(v)) / np.nanstd(v)
    v[~ok] = np.nanmax(v[ok])                   # never a minimum
    pk, _ = find_peaks(-v, prominence=prom, distance=max(1, w))
    return pk[ok[pk]]


def matched(a, b, tol):
    """number of positions in a with a position of b within tol."""
    if len(a) == 0 or len(b) == 0:
        return 0
    return int((np.abs(a[:, None] - b[None, :]).min(1) <= tol).sum())


def pearson(p, t):
    k = np.isfinite(p) & np.isfinite(t)
    if k.sum() < 10 or p[k].std() == 0 or t[k].std() == 0:
        return np.nan
    return float(np.corrcoef(p[k], t[k])[0, 1])


# ----------------------------------------------------------------------------- main loop
def evaluate(base, datasets, sigma, ins_windows, prom, tol, train_v2):
    heads0 = selected_heads(base)
    a0 = A0Source(base)
    band_rows, ins_rows = [], []
    for ds in datasets:
        acc, _, _ = diag_means(base, ds)
        log(f"▶ {ds}: distance profiles from {len(acc)} chromosome x cell sets")
        for fold in FOLD_TEST:
            got = fold_inputs(base, ds, fold)
            if got is None:
                continue
            P, T, meta, L, d = got
            keys = list(zip(meta.chrom, meta.start.astype(int)))
            iu = np.triu_indices(L, d)
            dist = (iu[1] - iu[0]) * BIN
            bmask = {b: (dist >= lo) & (dist < hi) for b, lo, hi in BANDS}
            A0 = {}
            for c in CELLS:
                h = heads0.get((fold, HEAD[c]))
                A0[c] = None if h is None else a0.get(ds, fold, c, h, keys)
            prof = {c: dist_profile(acc, fold, c, L) for c in CELLS}
            tv = None
            if ds == "independent" and train_v2 is not None:
                idx = [train_v2["key"].get(k) for k in keys]
                if any(i is None for i in idx):
                    log(f"⚠️  train_targets_v2: {sum(i is None for i in idx)} windows of {fold} missing; skipped")
                else:
                    tv = {c: train_v2["T"][np.array(idx)][:, :, train_v2["obs"].index(HEAD[c])]
                          for c in CELLS if HEAD[c] in train_v2["obs"]}
            for w in range(len(meta)):
                raw_t = {c: vec_to_mat(T[c][w].astype(float), L, d) for c in CELLS}
                sm_t = {c: nan_gauss(raw_t[c], sigma)[iu] for c in CELLS}
                for k, c in enumerate(CELLS):
                    preds = {"A4": P[w, :, k].astype(float), "dist": prof[c][iu[1] - iu[0]]}
                    if A0[c] is not None:
                        preds["A0"] = A0[c][w].astype(float)
                    for o in CELLS:
                        if o != c:
                            preds[f"obs_{o}"] = T[o][w].astype(float)
                    if tv is not None:
                        for o in tv:
                            preds[f"trainobs_{o}"] = tv[o][w].astype(float)
                    lab = dict(dataset=ds, fold=fold, chrom=meta.chrom[w], start=int(meta.start[w]), cell=c)
                    t_s = sm_t[c]
                    ins_t = {iw: insulation(raw_t[c], iw) for iw in ins_windows}
                    bnd_t = {iw: boundaries(ins_t[iw], iw, prom) for iw in ins_windows}
                    for name, v in preds.items():
                        mat = vec_to_mat(v, L, d)
                        p_s = nan_gauss(mat, sigma)[iu]
                        p_s[~np.isfinite(t_s)] = np.nan
                        for b, _, _ in BANDS:
                            m = bmask[b] & np.isfinite(p_s) & np.isfinite(t_s)
                            if m.sum() < 10:
                                continue
                            band_rows.append(dict(lab, predictor=name, band=b, r=pearson(p_s[m], t_s[m]),
                                                  mse=float(np.mean((p_s[m] - t_s[m]) ** 2)), n=int(m.sum())))
                        if name == "dist":
                            continue                  # Toeplitz map: no insulation structure
                        mat[~np.isfinite(raw_t[c])] = np.nan
                        for iw in ins_windows:
                            ins_p = insulation(mat, iw)
                            bp = boundaries(ins_p, iw, prom)
                            bt = bnd_t[iw]
                            ins_rows.append(dict(lab, predictor=name, w=iw, ins_r=pearson(ins_p, ins_t[iw]),
                                                 n_obs_boundaries=len(bt), n_pred_boundaries=len(bp),
                                                 obs_recovered=matched(bt, bp, tol),
                                                 pred_supported=matched(bp, bt, tol)))
            log(f"   {ds} {fold}: {len(meta)} windows done")
            del P, T, A0
    return pd.DataFrame(band_rows), pd.DataFrame(ins_rows)


# ----------------------------------------------------------------------------- summaries
def wilcox(a, b):
    k = a.notna() & b.notna()
    if k.sum() < 5:
        return np.nan
    try:
        return float(stats.wilcoxon(a[k], b[k]).pvalue)
    except ValueError:
        return np.nan


def summaries(B, I, out):
    key = ["dataset", "fold", "chrom", "start", "cell"]
    rows = []
    for (ds, c, b), g in B.groupby(["dataset", "cell", "band"]):
        wide = g.pivot_table(index=key, columns="predictor", values="r")
        widem = g.pivot_table(index=key, columns="predictor", values="mse")
        for p in wide.columns:
            row = dict(dataset=ds, cell=c, band=b, predictor=p, n_windows=int(wide[p].notna().sum()),
                       median_r=wide[p].median(), mean_r=wide[p].mean(), median_mse=widem[p].median())
            if "A4" in wide and p != "A4":
                row["median_delta_r_A4_minus"] = (wide["A4"] - wide[p]).median()
                row["wilcoxon_p_A4_vs"] = wilcox(wide["A4"], wide[p])
                fm = (wide["A4"] - wide[p]).groupby(level="fold").mean()
                row["folds_A4_better"] = f"{int((fm > 0).sum())}/{len(fm)}"
            rows.append(row)
    S = pd.DataFrame(rows)
    order = {b: i for i, (b, _, _) in enumerate(BANDS)}
    S = S.sort_values(["dataset", "cell", "band", "predictor"], key=lambda s: s.map(order) if s.name == "band" else s)
    S[S.predictor.isin(["A4", "A0"])].to_csv(f"{out}/S1_distance_bands.csv", index=False, float_format="%.4g")
    S[S.band == "all"].to_csv(f"{out}/S3_baselines.csv", index=False, float_format="%.4g")
    S.to_csv(f"{out}/S1b_distance_bands_all_predictors.csv", index=False, float_format="%.4g")

    rows = []
    for (ds, c, p, w), g in I.groupby(["dataset", "cell", "predictor", "w"]):
        rec = g.obs_recovered.sum() / g.n_obs_boundaries.sum() if g.n_obs_boundaries.sum() else np.nan
        pre = g.pred_supported.sum() / g.n_pred_boundaries.sum() if g.n_pred_boundaries.sum() else np.nan
        rows.append(dict(dataset=ds, cell=c, predictor=p, w=w, n_windows=len(g), median_ins_r=g.ins_r.median(),
                         mean_ins_r=g.ins_r.mean(), obs_boundaries=int(g.n_obs_boundaries.sum()),
                         pred_boundaries=int(g.n_pred_boundaries.sum()), recall=rec, precision=pre,
                         f1=2 * rec * pre / (rec + pre) if rec and pre else np.nan))
    S2 = pd.DataFrame(rows)
    for (ds, c, w), g in I.groupby(["dataset", "cell", "w"]):
        wide = g.pivot_table(index=key, columns="predictor", values="ins_r")
        if "A4" in wide and "A0" in wide:
            m = (S2.dataset == ds) & (S2.cell == c) & (S2.w == w) & (S2.predictor == "A0")
            S2.loc[m, "wilcoxon_p_A4_vs_A0_ins_r"] = wilcox(wide["A4"], wide["A0"])
    S2.to_csv(f"{out}/S2_insulation.csv", index=False, float_format="%.4g")
    return S, S2


# ----------------------------------------------------------------------------- figures
def figures(B, S, S2, out, ins_main):
    dss = list(B.dataset.unique())
    bands = [b for b, _, _ in BANDS if b != "all"]
    # E1 distance bands: A4 vs A0 boxes per cell
    fig, axes = plt.subplots(len(dss), 3, figsize=(11, 3.1 * len(dss)), squeeze=False, sharey=True)
    for i, ds in enumerate(dss):
        for j, c in enumerate(CELLS):
            ax = axes[i, j]
            g = B[(B.dataset == ds) & (B.cell == c)]
            for k, p in enumerate(["A4", "A0"]):
                data = [g[(g.band == b) & (g.predictor == p)].r.dropna().values for b in bands]
                bp = ax.boxplot(data, positions=np.arange(len(bands)) + (k - 0.5) * 0.36, widths=0.32,
                                patch_artist=True, showfliers=False, medianprops=dict(color="black"))
                for bx in bp["boxes"]:
                    bx.set_facecolor(matplotlib.colors.to_rgba(PCOL[p], 0.6))
            s = S[(S.dataset == ds) & (S.cell == c) & (S.predictor == "A0")].set_index("band")
            for x, b in enumerate(bands):
                if b in s.index:
                    pv = s.loc[b, "wilcoxon_p_A4_vs"]
                    ax.text(x, 1.0, "" if not np.isfinite(pv) else
                            ("****" if pv < 1e-4 else "***" if pv < 1e-3 else "**" if pv < 1e-2 else
                             "*" if pv < .05 else "ns"), ha="center", fontsize=7)
            ax.set_xticks(range(len(bands)))
            ax.set_xticklabels(bands)
            ax.set_ylim(-0.2, 1.08)
            ax.set_title(f"{ds} — {c}", fontsize=9)
            if j == 0:
                ax.set_ylabel("Pearson r (σ-smoothed)")
    axes[0, 0].legend([plt.Rectangle((0, 0), 1, 1, fc=matplotlib.colors.to_rgba(PCOL[p], .6)) for p in ("A4", "A0")],
                      ["iNeuroAkita (A4)", "Akita (A0)"], frameon=False, fontsize=7, loc="lower left")
    save(fig, f"{out}/E1_distance_bands")
    # E2 baselines, band all
    fig, axes = plt.subplots(1, len(dss), figsize=(5.5 * len(dss), 3.2), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        g = B[(B.dataset == ds) & (B.band == "all")]
        preds = ["A4", "A0", "dist"] + sorted(p for p in g.predictor.unique() if p.startswith("obs_")) + \
            sorted(p for p in g.predictor.unique() if p.startswith("trainobs_"))
        x0 = 0
        ticks, labs = [], []
        for c in CELLS:
            gc = g[g.cell == c]
            ps = [p for p in preds if p in gc.predictor.unique() and p != f"obs_{c}"]
            for k, p in enumerate(ps):
                v = gc[gc.predictor == p].r.dropna()
                bp = ax.boxplot([v], positions=[x0 + k], widths=0.7, patch_artist=True, showfliers=False,
                                medianprops=dict(color="black"))
                bp["boxes"][0].set_facecolor(matplotlib.colors.to_rgba(pcolor(p), 0.6 if p in PCOL else 0.35))
                ticks.append(x0 + k)
                labs.append(p)
            ax.text(x0 + (len(ps) - 1) / 2, 1.05, c, ha="center", fontsize=8, fontweight="bold")
            x0 += len(ps) + 1
        ax.set_xticks(ticks)
        ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=6)
        ax.set_ylim(-0.4, 1.12)
        ax.axhline(0, color="#cccccc", lw=0.6)
        ax.set(title=f"{ds}: models vs baselines (all distances)", ylabel="Pearson r (σ-smoothed)")
    save(fig, f"{out}/E2_baselines")
    # E3 insulation r and E4 boundary P/R
    if S2 is not None and len(S2):
        s2 = S2[S2.w == ins_main]
        fig, axes = plt.subplots(1, len(dss), figsize=(5 * len(dss), 3), squeeze=False)
        for ax, ds in zip(axes[0], dss):
            g = s2[s2.dataset == ds]
            preds = [p for p in ["A4", "A0"] + sorted(set(g.predictor) - {"A4", "A0"}) if p in set(g.predictor)]
            x = np.arange(len(CELLS))
            wd = 0.8 / max(1, len(preds))
            for k, p in enumerate(preds):
                v = g[g.predictor == p].set_index("cell").reindex(CELLS)
                ax.bar(x + (k - (len(preds) - 1) / 2) * wd, v.median_ins_r, wd,
                       color=matplotlib.colors.to_rgba(pcolor(p), 0.8 if p in PCOL else 0.4), label=p)
            ax.set_xticks(x)
            ax.set_xticklabels(CELLS)
            ax.set(title=f"{ds}: insulation r (w={ins_main} bins)", ylabel="median r")
            ax.legend(frameon=False, fontsize=6, ncol=2)
        save(fig, f"{out}/E3_insulation_r")
        fig, axes = plt.subplots(1, len(dss), figsize=(5 * len(dss), 3), squeeze=False)
        for ax, ds in zip(axes[0], dss):
            g = s2[(s2.dataset == ds) & s2.predictor.isin(["A4", "A0"])]
            x = np.arange(len(CELLS))
            for k, (p, met, hatch) in enumerate([("A4", "recall", ""), ("A4", "precision", "//"),
                                                 ("A0", "recall", ""), ("A0", "precision", "//")]):
                v = g[g.predictor == p].set_index("cell").reindex(CELLS)[met]
                ax.bar(x + (k - 1.5) * 0.2, v, 0.2, color=matplotlib.colors.to_rgba(PCOL[p], 0.7),
                       hatch=hatch, edgecolor="#333333", lw=0.4, label=f"{p} {met}")
            ax.set_xticks(x)
            ax.set_xticklabels(CELLS)
            ax.set(title=f"{ds}: boundary recall / precision (w={ins_main})", ylim=(0, 1))
            ax.legend(frameon=False, fontsize=6, ncol=2)
        save(fig, f"{out}/E4_boundaries")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--datasets", nargs="+", default=["heldout", "independent"], choices=["heldout", "independent"])
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--ins_window", type=int, nargs="+", default=[10, 25], help="insulation square, bins")
    ap.add_argument("--prominence", type=float, default=0.5, help="boundary prominence on the z-scored profile")
    ap.add_argument("--tol", type=int, default=2, help="boundary match tolerance, bins")
    ap.add_argument("--train_targets_v2", default=None,
                    help="npz with NPC_P16 / Neuron_P16 / Mic_AD targets on the v2 windows (optional)")
    a = ap.parse_args(argv)
    base = a.base_dir.rstrip("/")
    out = a.out or f"{base}/model_trainOut/stage2_eval"
    os.makedirs(out, exist_ok=True)
    B, I = evaluate(base, a.datasets, a.sigma, a.ins_window, a.prominence, a.tol,
                    load_train_targets_v2(a.train_targets_v2))
    B.to_csv(f"{out}/per_window_bands.csv.gz", index=False, float_format="%.5g")
    I.to_csv(f"{out}/per_window_insulation.csv.gz", index=False, float_format="%.5g")
    S, S2 = summaries(B, I, out)
    figures(B, S, S2, out, a.ins_window[-1])
    log(f"✅ stage-2 evaluation in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
