#!/usr/bin/env python3
"""
crosshead_matrix.py — full cross-head x cell-type matrix (task C2): every prediction head scored against
every observed cell type, on the same windows, out-of-fold.

Rows (predictors)  iNeuroAkita heads NPC_P16 / Neuron_P16 / Mic_AD (A4, the fold that held the chromosome
                   out) + the 5 original Akita heads (A0: HFF, H1hESC, GM12878, IMR90, HCT116) + the
                   OBSERVED maps of the other cell types (reference: how similar the cell types really are)
Columns (targets)  held-out NPC_P16 / Neuron_P16 / Mic_AD;  independent NPC_ctrl / Neuron_P35 / Mic_ctrl
Cell value         median per-window Pearson r (sigma-smoothed maps, as add_smoothed_metrics.py)

Specificity tests per target column
  own head vs each other iNeuroAkita head: paired Wilcoxon over windows, % windows where own head is best,
  specificity = r(own head) - mean r(other two heads)

Outputs (--out, default model_trainOut/crosshead_matrix)
  M1_matrix_median_r.csv   M2_specificity.csv   per_window_crosshead.csv.gz
  MF1_matrix_<dataset> heatmaps (numbers in cells)   MF2_specificity (.pdf .png .svg)

Usage
  python crosshead_matrix.py --base_dir $base_DIR
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_smoothed_metrics import vec_to_mat, nan_gauss
from celltype_specific_diag import fold_inputs, FOLD_TEST, CELLS, HEAD, OBS, COL, log
from stage2_eval import A0Source

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8})
AKITA = ["HFF", "H1hESC", "GM12878", "IMR90", "HCT116"]


def pr(x, y):
    k = np.isfinite(x) & np.isfinite(y)
    if k.sum() < 20 or x[k].std() == 0 or y[k].std() == 0:
        return np.nan
    return float(np.corrcoef(x[k], y[k])[0, 1])


def save(fig, pre):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{pre}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    log(f"✅ {pre}")


def run(a):
    out = a.out or f"{a.base_dir}/model_trainOut/crosshead_matrix"
    os.makedirs(out, exist_ok=True)
    a0 = A0Source(a.base_dir)
    rows = []
    for ds in a.datasets:
        for fold in FOLD_TEST:
            got = fold_inputs(a.base_dir, ds, fold)
            if got is None:
                continue
            P, T, meta, L, d = got
            iu = np.triu_indices(L, d)
            keys = list(zip(meta.chrom, meta.start.astype(int)))
            A0 = {}
            for c in CELLS:                  # all 5 Akita heads, aligned to these windows (per target file)
                A0[c] = {h: a0.get(ds, fold, c, h, keys) for h in AKITA}
            for w in range(len(meta)):
                sm = lambda v: nan_gauss(vec_to_mat(v.astype(float), L, d), a.sigma)[iu]
                tz = {c: sm(T[c][w]) for c in CELLS}
                pz = {f"A4:{HEAD[c]}": sm(P[w, :, k]) for k, c in enumerate(CELLS)}
                for c in CELLS:
                    tgt = OBS[ds][c]
                    lab = dict(dataset=ds, fold=fold, chrom=meta.chrom[w], start=int(meta.start[w]), target=tgt,
                               target_cell=c)
                    for name, p in pz.items():
                        rows.append(dict(lab, predictor=name, r=pr(p, tz[c])))
                    for h in AKITA:
                        if A0[c][h] is not None:
                            rows.append(dict(lab, predictor=f"A0:{h}", r=pr(sm(A0[c][h][w]), tz[c])))
                    for o in CELLS:
                        if o != c:
                            rows.append(dict(lab, predictor=f"obs:{OBS[ds][o]}", r=pr(tz[o], tz[c])))
            log(f"   {ds} {fold}: {len(meta)} windows")
            del P, T, A0
    R = pd.DataFrame(rows)
    R.to_csv(f"{out}/per_window_crosshead.csv.gz", index=False, float_format="%.4g")

    M = R.pivot_table(index=["dataset", "predictor"], columns="target_cell", values="r", aggfunc="median")[CELLS]
    M.to_csv(f"{out}/M1_matrix_median_r.csv", float_format="%.4g")
    key = ["dataset", "fold", "chrom", "start"]
    srows = []
    for (ds, c), g in R.groupby(["dataset", "target_cell"]):
        wide = g.pivot_table(index=key, columns="predictor", values="r")
        own = f"A4:{HEAD[c]}"
        others = [f"A4:{HEAD[o]}" for o in CELLS if o != c]
        best = wide[[own] + others].idxmax(axis=1)
        row = dict(dataset=ds, target=OBS[ds][c], own_head=own, median_r_own=wide[own].median(),
                   pct_windows_own_best=100 * float((best == own).mean()),
                   median_specificity=(wide[own] - wide[others].mean(axis=1)).median())
        for o in others:
            row[f"median_r_{o}"] = wide[o].median()
            row[f"wilcoxon_p_own_gt_{o}"] = stats.wilcoxon(wide[own], wide[o], alternative="greater",
                                                           nan_policy="omit").pvalue
        best0 = [p for p in wide.columns if p.startswith("A0:")]
        if best0:
            row["best_A0_head"] = wide[best0].median().idxmax()
            row["median_r_best_A0_head"] = wide[best0].median().max()
        srows.append(row)
    S = pd.DataFrame(srows)
    S.to_csv(f"{out}/M2_specificity.csv", index=False, float_format="%.4g")

    for ds in M.index.get_level_values(0).unique():
        m = M.loc[ds]
        order = [f"A4:{HEAD[c]}" for c in CELLS] + [f"A0:{h}" for h in AKITA] + \
            sorted(i for i in m.index if i.startswith("obs:"))
        m = m.reindex([o for o in order if o in m.index])
        fig, ax = plt.subplots(figsize=(3.8, 0.32 * len(m) + 1.2))
        im = ax.imshow(m.values, cmap="viridis", vmin=np.nanmin(m.values), vmax=np.nanmax(m.values), aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                if np.isfinite(v):
                    bold = m.index[i] == f"A4:{HEAD[CELLS[j]]}"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v < np.nanmean(m.values) else "black",
                            fontweight="bold" if bold else "normal")
        ax.set_xticks(range(len(CELLS)))
        ax.set_xticklabels([OBS[ds][c] for c in CELLS], rotation=30, ha="right")
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels(m.index, fontsize=7)
        for y in (2.5, 7.5):
            if y < len(m):
                ax.axhline(y, color="white", lw=2)
        ax.set_title(f"{ds}: predictor x observed target (median r)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        save(fig, f"{out}/MF1_matrix_{ds}")
    fig, axes = plt.subplots(1, len(a.datasets), figsize=(4 * len(a.datasets), 3), squeeze=False)
    for ax, ds in zip(axes[0], a.datasets):
        g = R[(R.dataset == ds) & R.predictor.str.startswith("A4:")]
        for j, c in enumerate(CELLS):
            gc = g[g.target_cell == c]
            for k, o in enumerate(CELLS):
                v = gc[gc.predictor == f"A4:{HEAD[o]}"].r.dropna()
                bp = ax.boxplot([v], positions=[j * 4 + k], widths=0.8, patch_artist=True, showfliers=False,
                                medianprops=dict(color="black"))
                bp["boxes"][0].set_facecolor(matplotlib.colors.to_rgba(COL[o], 0.9 if o == c else 0.35))
        ax.set_xticks([1, 5, 9])
        ax.set_xticklabels([OBS[ds][c] for c in CELLS])
        ax.set(title=f"{ds}: each target scored by the 3 heads (dark = own head)", ylabel="r per window")
    save(fig, f"{out}/MF2_specificity")
    log(f"✅ cross-head matrix in {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--datasets", nargs="+", default=["heldout", "independent"], choices=["heldout", "independent"])
    ap.add_argument("--sigma", type=float, default=2.0)
    run(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
