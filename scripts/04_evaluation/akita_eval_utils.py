#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
akita_eval_utils.py — shared helpers for akita_eval.py (map reconstruction, metrics, figures)

Implementation notes (each of these changes numbers or figure content if done differently):

1. `from_upper_triu` takes `diagonal_offset` (2 for Akita). Using
   np.triu_indices(L) at offset 0 infers L=446 from a 99,681-long vector
   instead of 448 and lays every value on the wrong diagonal.
2. The masked band near the diagonal is NaN, not 0. Zero-filling it would mean the
   Gaussian smoother bled zeros into real pixels, and `_masked_flat_offdiag`
   (which drops only the main diagonal) would feed ~2L perfectly-matched 0-vs-0 pairs
   into every correlation.
3. Metrics are computed on unsmoothed maps. Smoothing both maps before
   correlating suppresses exactly the high-frequency component the model
   predicts worst, so smoothed r runs high and is not comparable to published
   Akita numbers. `sigma` now only affects displayed maps.
4. Gene bands are placed against the CROPPED map extent
   (region_start + crop_bp .. + L*bin_size), not the full sequence span. Mapping against the
   full span shifts every gene by up to crop_bp/2 (~32 bins at Akita defaults).
5. Prediction channels and observed channels are indexed separately, so a joint
   multi-target model can be scored against a single-cell-type independent
   dataset without silently comparing the wrong head.

Public API
----------
    load_predictions(npz_path, regions=None, diagonal_offset=None)
    per_window_metrics(bundle, pairs, sigma=0.0)
    plot_joint_kde(df, ...)                # the MSE-vs-r figure, grouped
    plot_top_n_maps(...)
    plot_region_map_gene(...)
    colorize_gene_marks(marks, palette)
    OKABE_ITO, OKABE_ITO_LIGHT, OKABE_ITO_DARK
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
OKABE_ITO_LIGHT = ["#666666", "#F2C57C", "#9AD3F0", "#66C2A5",
                   "#F7EE9B", "#5FA8D3", "#EE9B6B", "#E4B3CB"]
OKABE_ITO_DARK = ["#000000", "#A66F00", "#2E7BA6", "#00664C",
                  "#A69C00", "#004E7A", "#8F3D00", "#8A4E72"]


def colorize_gene_marks(gene_marks: List[Dict], palette: Sequence[str]) -> List[Dict]:
    out = []
    for i, g in enumerate(gene_marks):
        g = dict(g)
        if not g.get("color"):
            g["color"] = palette[i % len(palette)]
        out.append(g)
    return out


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------


def infer_matrix_len(vec_len: int, diagonal_offset: int = 2) -> int:
    """vec_len = m(m+1)/2 with m = L - diagonal_offset; 99681, offset 2 -> 448."""
    m = (np.sqrt(8 * vec_len + 1) - 1) / 2
    m_int = int(round(m))
    if m_int * (m_int + 1) // 2 != vec_len:
        raise ValueError(f"vector length {vec_len} is not triangular")
    return m_int + diagonal_offset


def from_upper_triu(vec: np.ndarray, L: int, diagonal_offset: int = 2) -> np.ndarray:
    """Flattened upper triangle -> symmetric LxL matrix; masked band = NaN."""
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    mat = np.full((L, L), np.nan)
    iu = np.triu_indices(L, diagonal_offset)
    mat[iu] = vec
    mat[(iu[1], iu[0])] = vec
    return mat


def flat_triu(mat: np.ndarray, diagonal_offset: int = 2) -> np.ndarray:
    """Upper-triangle values only — each pixel counted once, masked band excluded."""
    return mat[np.triu_indices(mat.shape[0], diagonal_offset)]


def spatial_smoother(mat: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """NaN-aware Gaussian blur for DISPLAY ONLY. Never use before computing metrics."""
    if not sigma:
        return mat
    valid = np.isfinite(mat).astype(float)
    filled = np.where(np.isfinite(mat), mat, 0.0)
    num = gaussian_filter(filled, sigma=sigma)
    den = gaussian_filter(valid, sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    return np.where(np.isfinite(mat), out, np.nan)


def mirror_maps(pred_mat: np.ndarray, obs_mat: np.ndarray) -> np.ndarray:
    """Predicted above the diagonal, observed below — one panel, one colour scale."""
    L = pred_mat.shape[0]
    out = np.full((L, L), np.nan)
    iu = np.triu_indices(L, 1)
    il = np.tril_indices(L, -1)
    out[iu] = pred_mat[iu]
    out[il] = obs_mat[il]
    return out


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------


class Bundle(dict):
    """preds (N,vec,H), targets (N,vec,O), regions DataFrame, geometry, names."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def read_sequences_bed(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chrom", "start", "end", "split"], dtype={0: str})


def load_predictions(
    npz_path: str,
    data_dir: Optional[str] = None,
    splits: Optional[Sequence[str]] = None,
    diagonal_offset: Optional[int] = None,
    bin_size: Optional[int] = None,
) -> Bundle:
    """Load a prediction npz written by akita_predict.py (or the legacy
    two-key format, given --data-dir to recover regions)."""
    d = np.load(npz_path, allow_pickle=False)
    preds = d["test_preds_subset"].astype(np.float32)
    targs = d["test_targets"].astype(np.float32)

    off = int(d["diagonal_offset"]) if "diagonal_offset" in d.files else (
        diagonal_offset if diagonal_offset is not None else 2)
    L = int(d["matrix_len"]) if "matrix_len" in d.files else infer_matrix_len(targs.shape[1], off)
    bs = int(d["bin_size"]) if "bin_size" in d.files else (bin_size or 2048)

    if {"chrom", "start", "end"} <= set(d.files):
        regions = pd.DataFrame(dict(chrom=[str(c) for c in d["chrom"]],
                                    start=d["start"].astype(int),
                                    end=d["end"].astype(int)))
        if "split" in d.files:
            regions["split"] = [str(s) for s in d["split"]]
    elif data_dir:
        bed = read_sequences_bed(os.path.join(data_dir, "sequences.bed"))
        labels = list(splits) if splits else ["test"]
        regions = pd.concat([bed[bed["split"] == s] for s in labels], ignore_index=True)
    else:
        raise ValueError("npz has no region table; pass data_dir (and splits)")

    if len(regions) != targs.shape[0]:
        raise ValueError(
            f"{len(regions)} regions but {targs.shape[0]} windows in {os.path.basename(npz_path)}. "
            f"If this is a multi-split run, pass every split in the same order."
        )
    regions = regions.reset_index(drop=True)
    regions["seq_index"] = np.arange(len(regions))
    regions["region"] = (regions["chrom"] + ":" + regions["start"].astype(str)
                         + "-" + regions["end"].astype(str))

    head_names = ([str(x) for x in d["model_head_names"]] if "model_head_names" in d.files
                  else [f"head{i}" for i in range(preds.shape[-1])])
    obs_names = ([str(x) for x in d["obs_target_names"]] if "obs_target_names" in d.files
                 else [f"target{i}" for i in range(targs.shape[-1])])

    return Bundle(preds=preds, targets=targs, regions=regions, L=L,
                  diagonal_offset=off, bin_size=bs,
                  head_names=head_names, obs_names=obs_names, npz_path=npz_path)


def resolve_pairs(bundle: Bundle, pair_specs: Optional[Sequence[str]]) -> List[Tuple[int, int]]:
    """'0=2' or 'Neuron_P35=Neuron_P16' -> [(obs_index, head_index)]."""
    def idx(key, names, what):
        key = str(key).strip()
        if re.fullmatch(r"\d+", key):
            i = int(key)
        else:
            hits = [j for j, n in enumerate(names)
                    if re.sub(r"[^a-z0-9]", "", n.lower()) == re.sub(r"[^a-z0-9]", "", key.lower())]
            if len(hits) != 1:
                raise ValueError(f"{what} {key!r} not resolvable among {names}")
            i = hits[0]
        if not 0 <= i < len(names):
            raise ValueError(f"{what} index {i} out of range (have {len(names)})")
        return i

    if pair_specs:
        out = []
        for spec in pair_specs:
            o, h = spec.split("=")
            out.append((idx(o, bundle.obs_names, "observed target"),
                        idx(h, bundle.head_names, "model head")))
        return out
    if len(bundle.obs_names) == len(bundle.head_names):
        return [(i, i) for i in range(len(bundle.obs_names))]
    raise ValueError(
        f"{len(bundle.obs_names)} observed target(s) vs {len(bundle.head_names)} model head(s); "
        f"pass --pair OBS=HEAD (e.g. --pair Neuron_P35=Neuron_P16)"
    )


# ----------------------------------------------------------------------------
# metrics (unsmoothed)
# ----------------------------------------------------------------------------


def per_window_metrics(bundle: Bundle, pairs: Sequence[Tuple[int, int]]) -> pd.DataFrame:
    """Per-window MSE / Pearson / Spearman on the unsmoothed upper triangle."""
    from scipy.stats import pearsonr, spearmanr

    L, off = bundle.L, bundle.diagonal_offset
    iu = np.triu_indices(L, off)
    rows = []
    for oi, hi in pairs:
        for i in range(bundle.targets.shape[0]):
            obs = bundle.targets[i, :, oi].astype(np.float64)
            pred = bundle.preds[i, :, hi].astype(np.float64)
            m = np.isfinite(obs) & np.isfinite(pred)
            if m.sum() < 3:
                continue
            reg = bundle.regions.iloc[i]
            rows.append(dict(
                seq_index=int(reg["seq_index"]), chrom=reg["chrom"],
                start=int(reg["start"]), end=int(reg["end"]), region=reg["region"],
                split=reg.get("split", ""),
                obs_index=oi, obs_target=bundle.obs_names[oi],
                head_index=hi, model_head=bundle.head_names[hi],
                matched=(bundle.obs_names[oi] == bundle.head_names[hi]),
                n_pixels=int(m.sum()),
                mse=float(np.mean((pred[m] - obs[m]) ** 2)),
                pearson=float(pearsonr(pred[m], obs[m])[0]),
                spearman=float(spearmanr(pred[m], obs[m])[0]),
                obs_sd=float(np.std(obs[m])),
            ))
    return pd.DataFrame(rows)


def pooled_stats(bundle: Bundle, oi: int, hi: int) -> Dict[str, float]:
    """Pooled-pixel MSE and r across all windows (the 'global' numbers)."""
    n = sx = sy = sxx = syy = sxy = sse = 0.0
    for i in range(bundle.targets.shape[0]):
        x = bundle.preds[i, :, hi].astype(np.float64)
        y = bundle.targets[i, :, oi].astype(np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        n += x.size
        sx += x.sum(); sy += y.sum()
        sxx += (x * x).sum(); syy += (y * y).sum(); sxy += (x * y).sum()
        sse += ((x - y) ** 2).sum()
    if n < 3:
        return dict(mse=np.nan, corr=np.nan)
    cov = sxy - sx * sy / n
    vx, vy = sxx - sx**2 / n, syy - sy**2 / n
    return dict(mse=float(sse / n),
                corr=float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan)


# ----------------------------------------------------------------------------
# the MSE-vs-r joint figure
# ----------------------------------------------------------------------------


def _hist_kde_curve(ax, values: np.ndarray, n_bins: int, color: str,
                    horizontal: bool = False, alpha: float = 1.0) -> None:
    """Scaled KDE curve over a count histogram, as in the original figure."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3 or np.std(v) <= 0:
        return
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return
    grid = np.linspace(lo, hi, 256)
    dens = gaussian_kde(v)(grid) * v.size * ((hi - lo) / n_bins)
    if horizontal:
        ax.plot(dens, grid, color=color, lw=1.5, alpha=alpha)
    else:
        ax.plot(grid, dens, color=color, lw=1.5, alpha=alpha)


def plot_joint_kde(
    df: pd.DataFrame,
    out_pdf: str,
    title: str = "",
    group_by: Optional[str] = None,
    global_ref: Optional[Dict[str, float]] = None,
    y_min: Optional[float] = -0.4,
    y_max: Optional[float] = 1.0,
    color_mode: str = "kde_scatter",
    kde_grid: int = 150,
    point_size: int = 12,
    alpha_pts: float = 0.85,
    hist_color_mse: str = "steelblue",
    hist_color_r: str = "salmon",
    scatter_cmap: str = "viridis",
    palette: Sequence[str] = tuple(OKABE_ITO[1:]),
    pdf_handle: Optional[PdfPages] = None,
    style: str = "detailed",
    panel_label: Optional[str] = None,
    hist_bins: int = 40,
    hist_alpha: float = 1.0,
    hist_edgecolor: str = "k",
    hist_edgewidth: float = 0.4,
    kde_line_alpha: float = 1.0,
) -> None:
    """Per-window MSE vs Pearson r with marginal histograms.

    Differences from the original: points may be coloured by chromosome or split
    (`group_by`), the reference lines distinguish the pooled-pixel value from the
    median per-window value (they are different quantities and in the original
    figure only the pooled one was drawn, which is why the dashed line sat well
    below the mode of the cloud), and the annotation box reports n, median, IQR.
    """
    d = df.dropna(subset=["mse", "pearson"])
    if d.empty:
        return
    x = d["mse"].to_numpy(float)
    y = d["pearson"].to_numpy(float)
    classic = (style == "classic")
    grouped = bool(group_by) and group_by in d.columns and d[group_by].nunique() > 1 and not classic

    fig = plt.figure(figsize=(8.5, 8.5))
    gs = plt.GridSpec(4, 4, figure=fig, hspace=0.28, wspace=0.28)
    ax = fig.add_subplot(gs[1:4, 0:3])
    axx = fig.add_subplot(gs[0, 0:3], sharex=ax)
    axy = fig.add_subplot(gs[1:4, 3], sharey=ax)

    if grouped:
        groups = sorted(d[group_by].unique(), key=lambda v: (len(str(v)), str(v)))
        for gi, g in enumerate(groups):
            s_ = d[d[group_by] == g]
            c = palette[gi % len(palette)]
            ax.scatter(s_["mse"], s_["pearson"], s=point_size, alpha=alpha_pts, color=c,
                       label=f"{g} (n={len(s_)}, med r={s_['pearson'].median():.3f})",
                       edgecolors="none")
            g_alpha = hist_alpha if hist_alpha < 1.0 else 0.55  # overlapping groups
            axx.hist(s_["mse"], bins=hist_bins, alpha=g_alpha, color=c)
            axy.hist(s_["pearson"], bins=hist_bins, alpha=g_alpha, color=c,
                     orientation="horizontal")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    else:
        if x.size > 2 and color_mode in ("kde_scatter", "kde_contour_scatter"):
            z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
            order = np.argsort(z)
            if color_mode == "kde_contour_scatter":
                X, Y = np.mgrid[x.min():x.max():complex(kde_grid),
                                y.min():y.max():complex(kde_grid)]
                Z = gaussian_kde(np.vstack([x, y]))(
                    np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax.contourf(X, Y, Z, levels=12, cmap="Blues", alpha=0.55)
            sc = ax.scatter(x[order], y[order], c=z[order], s=point_size,
                            alpha=alpha_pts, cmap=scatter_cmap, edgecolors="none")
            cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            if not classic:
                cb.set_label("point density")
        else:
            ax.scatter(x, y, s=point_size, alpha=alpha_pts, edgecolors="none")
        axx.hist(x, bins=hist_bins, color=hist_color_mse, alpha=hist_alpha,
                 edgecolor=hist_edgecolor, linewidth=hist_edgewidth)
        axy.hist(y, bins=hist_bins, color=hist_color_r, orientation="horizontal",
                 alpha=hist_alpha, edgecolor=hist_edgecolor, linewidth=hist_edgewidth)
        _hist_kde_curve(axx, x, hist_bins, hist_color_mse, alpha=kde_line_alpha)
        _hist_kde_curve(axy, y, hist_bins, hist_color_r, horizontal=True,
                        alpha=kde_line_alpha)

    med_r, med_mse = float(np.median(y)), float(np.median(x))
    q25, q75 = float(np.percentile(y, 25)), float(np.percentile(y, 75))
    gm = gr = None
    if global_ref:
        gm, gr = global_ref.get("mse"), global_ref.get("corr")

    if classic:
        # original look: dashed pooled-pixel lines only, compact box, plain labels
        if gm is not None and np.isfinite(gm):
            ax.axvline(gm, color="k", ls="--", lw=1)
        if gr is not None and np.isfinite(gr):
            ax.axhline(gr, color="k", ls="--", lw=1)
        box = (f"MSE (global) = {gm:.4f}\nCorr (global) = {gr:.4f}"
               if (gm is not None and gr is not None)
               else f"MSE (median) = {med_mse:.4f}\nr (median) = {med_r:.4f}")
        ax.text(0.98, 0.02, box, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9))
        ax.set_xlabel("MSE", fontsize=13)
        ax.set_ylabel("Pearson corr", fontsize=13)
        if title:
            axx.set_title(title, fontsize=14)
    else:
        ax.axhline(med_r, color="0.25", ls="-", lw=1)
        ax.axvline(med_mse, color="0.25", ls="-", lw=1)
        lines = [f"n windows = {len(d)}",
                 f"median r = {med_r:.4f}  (IQR {q25:.3f}\u2013{q75:.3f})",
                 f"median MSE = {med_mse:.4f}"]
        if gm is not None and np.isfinite(gm):
            ax.axvline(gm, color="k", ls="--", lw=1)
            lines.append(f"pooled-pixel MSE = {gm:.4f}")
        if gr is not None and np.isfinite(gr):
            ax.axhline(gr, color="k", ls="--", lw=1)
            lines.append(f"pooled-pixel r = {gr:.4f}")
        if "pearson_null" in d.columns and d["pearson_null"].notna().any():
            lines.append(
                f"\u0394r vs distance null = {np.nanmean(d['pearson'] - d['pearson_null']):+.4f}")
        ax.text(0.98, 0.02, "\n".join([l for l in lines if l]), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
        ax.set_xlabel("MSE (per window, unsmoothed)")
        ax.set_ylabel("Pearson r (per window, unsmoothed)")
        fig.suptitle(title + "\nsolid = median per window, dashed = pooled across pixels",
                     fontsize=12)

    ax.grid(True, alpha=0.25)
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    axx.set_ylabel("Frequency", fontsize=12 if classic else 10)
    axx.set_xlabel("")
    plt.setp(axx.get_xticklabels(), visible=False)
    axy.set_xlabel("Frequency", fontsize=12 if classic else 10)
    axy.set_ylabel("")
    plt.setp(axy.get_yticklabels(), visible=False)
    axy.grid(False)
    if panel_label:
        fig.text(0.02, 0.97, panel_label, fontsize=20, fontweight="bold", va="top")

    if pdf_handle is not None:
        pdf_handle.savefig(fig, bbox_inches="tight")
    else:
        with PdfPages(out_pdf) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        print(f"\u2705 {out_pdf}")
    plt.close(fig)


# ----------------------------------------------------------------------------
# maps
# ----------------------------------------------------------------------------


def _map_extent(region_start: int, region_end: int, L: int, bin_size: int,
                crop_bp: Optional[int] = None) -> Tuple[int, int, int]:
    """Genomic span actually covered by the LxL map.

    The target map covers only the cropped centre of the sequence:
        covered = L * bin_size,  crop_bp = (region_len - covered) / 2
    Mapping gene coordinates against the full region (full-span mapping)
    shifts every band by crop_bp/2.
    """
    covered = L * bin_size
    if crop_bp is None:
        crop_bp = max(int(round(((region_end - region_start) - covered) / 2)), 0)
    map_start = region_start + crop_bp
    return map_start, map_start + covered, crop_bp


def _parse_region_str(region: str) -> Tuple[str, int, int]:
    m = re.match(r"^(\w+):(\d+)-(\d+)$", str(region).strip())
    if not m:
        raise ValueError(f"cannot parse region {region!r}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def genes_to_bin_spans(map_start: int, map_end: int, L: int, bin_size: int,
                       genes: List[Dict]) -> List[Dict]:
    """Clip gene intervals to the mapped extent and convert to bin coordinates."""
    spans = []
    for g in genes:
        s, e = int(g["start"]), int(g["end"])
        if e <= map_start or s >= map_end:
            continue
        b0 = (max(s, map_start) - map_start) / bin_size
        b1 = (min(e, map_end) - map_start) / bin_size
        spans.append(dict(name=g.get("name", ""), b0=b0, b1=max(b1, b0 + 0.5),
                          color=g.get("color")))
    return spans


def overlay_gene_spans(ax, spans: List[Dict], L: int, default_color: str = "#111111",
                       alpha: float = 0.22, label_color: str = "#111111",
                       label_outline: str = "white", label_y: float = -0.04,
                       draw_labels: bool = True) -> None:
    import matplotlib.patheffects as pe

    for sp in spans:
        c = sp.get("color") or default_color
        ax.axvspan(sp["b0"], sp["b1"], color=c, alpha=alpha, lw=0)
        ax.axhspan(sp["b0"], sp["b1"], color=c, alpha=alpha, lw=0)
        if draw_labels and sp.get("name"):
            t = ax.text((sp["b0"] + sp["b1"]) / 2, label_y * L if label_y < 0 else label_y,
                        sp["name"], ha="center",
                        va="top" if label_y < 0 else "bottom",
                        fontsize=8, color=label_color, clip_on=False)
            t.set_path_effects([pe.withStroke(linewidth=2, foreground=label_outline)])


def plot_map_pair(ax_pred, ax_obs, pred, obs, vmin, vmax, cmap="RdBu_r"):
    im = ax_pred.imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
    ax_obs.imshow(obs, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
    return im


def plot_top_n_maps(
    bundle: Bundle, metrics: pd.DataFrame, oi: int, hi: int, out_pdf: str,
    which: str = "best", top_n: int = 5, sigma: float = 0.0,
    vmin: float = -2.0, vmax: float = 2.0, cmap: str = "RdBu_r",
    mirrored: bool = True, pdf_handle: Optional[PdfPages] = None,
) -> None:
    """Best / median / worst windows for one (observed, head) pair."""
    d = metrics[(metrics["obs_index"] == oi) & (metrics["head_index"] == hi)]
    d = d.dropna(subset=["pearson"]).sort_values("pearson", ascending=False)
    if d.empty:
        return
    if which == "best":
        pick = d.head(top_n)
    elif which == "worst":
        pick = d.tail(top_n)
    else:
        mid = max(len(d) // 2 - top_n // 2, 0)
        pick = d.iloc[mid:mid + top_n]

    L, off = bundle.L, bundle.diagonal_offset
    own = pdf_handle is None
    pdf = PdfPages(out_pdf) if own else pdf_handle
    for _, row in pick.iterrows():
        si = int(row["seq_index"])
        pred = spatial_smoother(from_upper_triu(bundle.preds[si, :, hi], L, off), sigma)
        obs = spatial_smoother(from_upper_triu(bundle.targets[si, :, oi], L, off), sigma)
        if mirrored:
            fig, ax = plt.subplots(figsize=(5.2, 5.4))
            im = ax.imshow(mirror_maps(pred, obs), cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.03, 0.97, "predicted", transform=ax.transAxes, fontsize=8, va="top")
            ax.text(0.97, 0.03, "observed", transform=ax.transAxes, fontsize=8, ha="right")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
            im = plot_map_pair(axes[0], axes[1], pred, obs, vmin, vmax, cmap)
            axes[0].set_title("predicted", fontsize=9)
            axes[1].set_title("observed", fontsize=9)
            for a in axes:
                a.set_xticks([]); a.set_yticks([])
            fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
        smooth_note = f", display σ={sigma}" if sigma else ", unsmoothed"
        fig.suptitle(f"{row['region']}  [{row['chrom']}]\n"
                     f"{row['obs_target']} vs head {row['model_head']} | "
                     f"r={row['pearson']:.3f}  MSE={row['mse']:.4f}{smooth_note}", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    if own:
        pdf.close()
        print(f"✅ {out_pdf}")


def plot_region_map_gene(
    bundle: Bundle, metrics: pd.DataFrame, region: Union[int, str], oi: int, hi: int,
    out_pdf: str, sigma: float = 0.0, vmin: float = -2.0, vmax: float = 2.0,
    cmap: str = "RdBu_r", gene_marks: Optional[List[Dict]] = None,
    band_color: str = "#111111", band_alpha: float = 0.22,
    label_color: str = "#111111", label_outline: str = "white",
    label_pos: str = "bottom", label_offset: float = 0.04,
    bottom_adjust: float = 0.22, crop_bp: Optional[int] = None,
    mirrored: bool = False, pdf_handle: Optional[PdfPages] = None,
) -> None:
    """One region, predicted vs observed, with crop-aware gene bands.

    `region` may be a sequence index, an exact 'chr:start-end' label, or a
    'chr:start-end' locus, in which case every overlapping window is drawn.
    """
    regs = bundle.regions
    if isinstance(region, (int, np.integer)):
        idxs = [int(region)]
    else:
        exact = regs.index[regs["region"] == str(region)].tolist()
        if exact:
            idxs = exact
        else:
            chrom, s, e = _parse_region_str(region)
            idxs = regs.index[(regs["chrom"] == chrom) & (regs["start"] < e)
                              & (regs["end"] > s)].tolist()
            if not idxs:
                raise ValueError(f"no window overlaps {region}")

    L, off, bs = bundle.L, bundle.diagonal_offset, bundle.bin_size
    own = pdf_handle is None
    pdf = PdfPages(out_pdf) if own else pdf_handle
    for si in idxs:
        row = regs.iloc[si]
        m = metrics[(metrics["seq_index"] == int(row["seq_index"]))
                    & (metrics["obs_index"] == oi) & (metrics["head_index"] == hi)]
        r_val = float(m["pearson"].iloc[0]) if len(m) else np.nan
        mse_val = float(m["mse"].iloc[0]) if len(m) else np.nan

        pred = spatial_smoother(from_upper_triu(bundle.preds[si, :, hi], L, off), sigma)
        obs = spatial_smoother(from_upper_triu(bundle.targets[si, :, oi], L, off), sigma)
        map_start, map_end, used_crop = _map_extent(int(row["start"]), int(row["end"]),
                                                    L, bs, crop_bp)
        spans = genes_to_bin_spans(map_start, map_end, L, bs, gene_marks) if gene_marks else []
        label_y = -label_offset if label_pos == "bottom" else 1.0 + label_offset

        if mirrored:
            fig, axes_list = plt.subplots(1, 1, figsize=(6.0, 6.2))
            axes = [axes_list]
            im = axes[0].imshow(mirror_maps(pred, obs), cmap=cmap, vmin=vmin, vmax=vmax,
                                interpolation="none")
            axes[0].text(0.03, 0.97, "predicted", transform=axes[0].transAxes,
                         fontsize=8, va="top")
            axes[0].text(0.97, 0.03, "observed", transform=axes[0].transAxes,
                         fontsize=8, ha="right")
        else:
            fig, axes_arr = plt.subplots(1, 2, figsize=(9.2, 4.8))
            axes = list(axes_arr)
            im = plot_map_pair(axes[0], axes[1], pred, obs, vmin, vmax, cmap)
            axes[0].set_title("predicted", fontsize=9)
            axes[1].set_title("observed", fontsize=9)
        for a in axes:
            if spans:
                overlay_gene_spans(a, spans, L, band_color, band_alpha, label_color,
                                   label_outline, label_y)
            a.set_xlim(-0.5, L - 0.5)
            a.set_ylim(L - 0.5, -0.5)
            a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
        smooth_note = f", display σ={sigma}" if sigma else ", unsmoothed"
        fig.suptitle(
            f"{row['region']}   map covers {map_start:,}-{map_end:,} (crop {used_crop:,} bp)\n"
            f"{bundle.obs_names[oi]} vs head {bundle.head_names[hi]} | "
            f"r={r_val:.3f}  MSE={mse_val:.4f}{smooth_note}", fontsize=10)
        if label_pos == "bottom":
            fig.subplots_adjust(bottom=bottom_adjust)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    if own:
        pdf.close()
        print(f"✅ {out_pdf}")


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------


def self_test() -> int:
    assert infer_matrix_len(99681, 2) == 448

    rng = np.random.default_rng(0)
    L, off, bs = 20, 2, 2048
    vec = rng.normal(size=len(np.triu_indices(L, off)[0]))
    mat = from_upper_triu(vec, L, off)
    assert np.allclose(mat[np.triu_indices(L, off)], vec)
    assert np.isnan(np.diag(mat)).all() and np.isnan(np.diag(mat, 1)).all()
    assert flat_triu(mat, off).size == vec.size

    sm = spatial_smoother(mat, 1.0)
    assert np.isnan(sm[np.diag_indices(L)]).all(), "masked band stays masked"
    assert np.isfinite(sm[np.triu_indices(L, off)]).all(), "no NaN bleed into data"

    m = mirror_maps(np.ones((L, L)), np.full((L, L), 2.0))
    assert m[0, L - 1] == 1.0 and m[L - 1, 0] == 2.0

    # crop-aware gene placement: full-span mapping is measurably different
    seq_start, seq_len = 1_000_000, 1_048_576
    covered = L * bs
    ms, me, crop = _map_extent(seq_start, seq_start + seq_len, L, bs)
    assert me - ms == covered and crop == (seq_len - covered) // 2
    gene = [dict(name="G", start=ms + 4 * bs, end=ms + 6 * bs)]
    sp = genes_to_bin_spans(ms, me, L, bs, gene)[0]
    assert abs(sp["b0"] - 4) < 1e-9 and abs(sp["b1"] - 6) < 1e-9, "crop-aware bins"
    old_b0 = (gene[0]["start"] - seq_start) / (seq_len / L)  # full-span mapping
    assert abs(old_b0 - sp["b0"]) > 1.0, "full-span mapping is displaced by >1 bin"

    # genes outside the mapped extent are dropped, not clamped to the edge
    assert genes_to_bin_spans(ms, me, L, bs, [dict(name="X", start=ms - 10 * bs,
                                                   end=ms - 5 * bs)]) == []

    # pair resolution
    b = Bundle(obs_names=["Neuron_P35"], head_names=["NPC_P16", "Mic_AD", "Neuron_P16"])
    assert resolve_pairs(b, ["Neuron_P35=Neuron_P16"]) == [(0, 2)]
    assert resolve_pairs(b, ["0=2"]) == [(0, 2)]
    try:
        resolve_pairs(b, None)
    except ValueError as e:
        assert "--pair" in str(e)
    else:
        raise AssertionError("ambiguous pairing must raise")
    b2 = Bundle(obs_names=["a", "b"], head_names=["a", "b"])
    assert resolve_pairs(b2, None) == [(0, 0), (1, 1)]

    print("✅ akita_eval_utils self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
