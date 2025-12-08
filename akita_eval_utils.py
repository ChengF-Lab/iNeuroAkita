#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
akita_eval_utils.py

Reusable helpers to:
- load NPZ + metadata
- reconstruct/smooth matrices
- compute GLOBAL pooled MSE & Pearson r (streaming, off-diagonal, smoothed)
- build per-sequence Pearson for ranking
- plot: top-N maps, specific region map (with optional gene spans/labels)
- plot: KDE-colored 'smoothed evaluation' scatter+hist with global overlays & fixed y-range

All plotting colors are configurable from the calling code (no hard-coding required).
"""

from typing import Optional, Union, List, Dict, Tuple
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.patheffects as pe


# ----------------------------
# Palettes (for convenience)
# ----------------------------
OKABE_ITO = ["#000000","#E69F00","#56B4E9","#009E73",
             "#F0E442","#0072B2","#D55E00","#CC79A7"]

TABLEAU10 = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
             "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]

TAB20 = ["#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a",
         "#d62728","#ff9896","#9467bd","#c5b0d5","#8c564b","#c49c94",
         "#e377c2","#f7b6d2","#7f7f7f","#c7c7c7","#bcbd22","#dbdb8d",
         "#17becf","#9edae5"]

SET2 = ["#66C2A5","#FC8D62","#8DA0CB","#E78AC3","#A6D854",
        "#FFD92F","#E5C494","#B3B3B3"]


def colorize_gene_marks(gene_marks: List[Dict], palette: List[str]) -> List[Dict]:
    """Fill missing 'color' fields in gene_marks with cyclic palette colors."""
    out = []
    for i, g in enumerate(gene_marks):
        g2 = dict(g)
        if "color" not in g2 or not g2["color"]:
            g2["color"] = palette[i % len(palette)]
        out.append(g2)
    return out


# ----------------------------
# Loading helpers
# ----------------------------
def load_npz_and_metadata(
    npz_path: str,
    bed_file1: str,
    bed_file2: str,
    targets_file: str,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, List[str], int]:
    """
    Returns:
        test_preds_subset: np.ndarray (N, vec_len, T)
        test_targets:      np.ndarray (N, vec_len, T)
        region_labels:     pd.Series (length N_test)
        target_names:      List[str] (length T)
        L:                 int
    """
    data = np.load(npz_path)
    test_preds_subset = data["test_preds_subset"]
    test_targets = data["test_targets"]

    # load sequences
    sequences1 = pd.read_csv(bed_file1, sep="\t", names=["chr", "start", "end", "type"])
    sequences2 = pd.read_csv(bed_file2, sep="\t", names=["chr", "start", "end", "type"])
    sequences = pd.concat([sequences1, sequences2], axis=0).reset_index(drop=True)
    sequences_test = sequences[sequences["type"] == "test"].reset_index(drop=True)
    region_labels = sequences_test.apply(lambda r: f"{r['chr']}:{r['start']}-{r['end']}", axis=1)

    # targets
    targets_df = pd.read_csv(targets_file, sep="\t")
    target_names = targets_df["identifier"].tolist()

    # infer L
    _, vec_len, _ = test_targets.shape
    L = int((np.sqrt(8 * vec_len + 1) - 1) // 2)

    return test_preds_subset, test_targets, region_labels, target_names, L


# ----------------------------
# Matrix helpers
# ----------------------------
def from_upper_triu(vec: np.ndarray, L: int) -> np.ndarray:
    """Reconstruct full symmetric matrix (with original diagonal) from upper-triangular vector."""
    mat = np.zeros((L, L), dtype=np.float32)
    triu_idx = np.triu_indices(L)
    mat[triu_idx] = vec
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat


def spatial_smoother(mat: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian smoothing; keep the original diagonal."""
    sm = gaussian_filter(mat, sigma=sigma)
    np.fill_diagonal(sm, np.diag(mat))
    return sm


def _masked_flat_offdiag(mat: np.ndarray) -> np.ndarray:
    """Flatten matrix excluding the diagonal."""
    L = mat.shape[0]
    mask = ~np.eye(L, dtype=bool).reshape(-1)
    return mat.reshape(-1)[mask]


# ----------------------------
# Metrics
# ----------------------------
def compute_global_stats_streaming(
    test_preds_subset: np.ndarray,
    test_targets: np.ndarray,
    L: int,
    sigma: float = 2.0,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    GLOBAL pooled MSE & Pearson r per target across ALL sequences and off-diagonal pixels,
    after Gaussian smoothing. Streaming accumulation (memory-safe).

    Returns: {target_name: {"mse": float, "corr": float}}
    """
    N, _, T = test_targets.shape
    if target_names is None:
        target_names = [f"target{t}" for t in range(T)]

    stats: Dict[str, Dict[str, float]] = {}
    for t in range(T):
        n_tot = 0
        sx = sy = sxx = syy = sxy = sse = 0.0

        for i in range(N):
            x_vec = test_targets[i, :, t].astype(np.float32)
            y_vec = test_preds_subset[i, :, t].astype(np.float32)

            X = from_upper_triu(x_vec, L)
            Y = from_upper_triu(y_vec, L)

            Xs = spatial_smoother(X, sigma=sigma)
            Ys = spatial_smoother(Y, sigma=sigma)

            x = _masked_flat_offdiag(Xs)
            y = _masked_flat_offdiag(Ys)

            m = ~np.isnan(x) & ~np.isnan(y)
            if not np.any(m):
                continue

            x = x[m]; y = y[m]
            n = x.size
            n_tot += n

            sx  += float(x.sum())
            sy  += float(y.sum())
            sxx += float((x * x).sum())
            syy += float((y * y).sum())
            sxy += float((x * y).sum())
            sse += float(((y - x) ** 2).sum())

        if n_tot == 0:
            stats[target_names[t]] = {"mse": float("nan"), "corr": float("nan")}
        else:
            mx, my = sx / n_tot, sy / n_tot
            sxy_c = sxy - n_tot * mx * my
            sxx_c = sxx - n_tot * mx * mx
            syy_c = syy - n_tot * my * my
            denom = (max(sxx_c, 0.0) * max(syy_c, 0.0)) ** 0.5
            r = (sxy_c / denom) if denom > 0 else float("nan")
            mse = sse / n_tot
            stats[target_names[t]] = {"mse": mse, "corr": r}

    return stats


def build_per_seq_pearson(
    test_preds_subset: np.ndarray,
    test_targets: np.ndarray,
    L: int,
    target_names: List[str],
    sigma: float = 2.0,
    region_labels: Optional[Union[pd.Series, List[str]]] = None,
) -> pd.DataFrame:
    """
    Compute per-sequence Pearson r (smoothed, off-diagonal) for ranking.
    Returns a DataFrame with columns: Region (if provided) + Pearson_<target>.
    """
    N, _, T = test_targets.shape
    out: Dict[str, List[float]] = {}
    if region_labels is not None:
        out["Region"] = list(region_labels[:N])

    for t in range(T):
        r_list: List[float] = []
        for i in range(N):
            Xt = spatial_smoother(from_upper_triu(test_targets[i, :, t], L), sigma=sigma)
            Yt = spatial_smoother(from_upper_triu(test_preds_subset[i, :, t], L), sigma=sigma)
            m = ~np.eye(L, dtype=bool).reshape(-1)
            x = Xt.reshape(-1)[m]
            y = Yt.reshape(-1)[m]

            if x.size > 1 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                xm, ym = x.mean(), y.mean()
                xv = x - xm; yv = y - ym
                denom = np.sqrt((xv * xv).sum() * (yv * yv).sum())
                r = (xv * yv).sum() / denom if denom > 0 else np.nan
            else:
                r = np.nan
            r_list.append(r)
        out[f"Pearson_{target_names[t]}"] = r_list

    return pd.DataFrame(out)


# ----------------------------
# Plotters (maps)
# ----------------------------
def plot_top_n_maps(
    metrics_df_smooth: pd.DataFrame,
    region_labels: Union[pd.Series, List[str]],
    test_preds_subset: np.ndarray,
    test_targets: np.ndarray,
    L: int,
    target_names: List[str],
    target_index: int,
    out_pdf: str,
    top_n: int = 5,
    sigma: float = 2.0,
    vmin: float = -2.0,
    vmax: float = 2.0,
    cmap: str = "RdBu_r",
) -> None:
    """Plot top-N regions by per-sequence Pearson for a given target_index."""
    tname = target_names[target_index]
    pearson_col = f"Pearson_{tname}"
    if pearson_col not in metrics_df_smooth.columns:
        raise ValueError(f"{pearson_col} not found in metrics_df_smooth")

    top_df = metrics_df_smooth.sort_values(pearson_col, ascending=False).head(top_n)

    with PdfPages(out_pdf) as pdf:
        for seq_idx in top_df.index:
            region = region_labels[seq_idx]
            r_val = float(metrics_df_smooth.loc[seq_idx, pearson_col])

            pred = spatial_smoother(from_upper_triu(test_preds_subset[seq_idx, :, target_index], L), sigma=sigma)
            true = spatial_smoother(from_upper_triu(test_targets[seq_idx, :, target_index], L), sigma=sigma)

            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            im1 = plt.matshow(pred, fignum=False, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im1, fraction=0.04, pad=0.05)
            plt.title(f"Predicted — {tname}", y=1.15)
            plt.ylabel(region)

            plt.subplot(1, 2, 2)
            im2 = plt.matshow(true, fignum=False, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im2, fraction=0.04, pad=0.05)
            plt.title(f"Target — {tname}", y=1.15)

            plt.suptitle(f"{region} | r = {r_val:.3f}", fontsize=10)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"✅ Saved top-{top_n} map plots for '{tname}' to: {out_pdf}")


def _resolve_region_index(
    region_labels: Union[pd.Series, List[str]],
    region: Union[int, str]
) -> int:
    """Return sequence index given an integer index or a label 'chr:start-end'."""
    if isinstance(region, int):
        if region < 0 or region >= len(region_labels):
            raise IndexError(f"Region index {region} out of range [0, {len(region_labels)-1}]")
        return region
    labels = np.asarray(region_labels)
    idx = np.where(labels == region)[0]
    if idx.size == 0:
        raise ValueError(f"Region '{region}' not found")
    return int(idx[0])


def plot_region_map(
    region: Union[int, str],
    target_index: int,
    region_labels: Union[pd.Series, List[str]],
    test_preds_subset: np.ndarray,
    test_targets: np.ndarray,
    L: int,
    target_names: List[str],
    out_pdf: str,
    sigma: float = 2.0,
    vmin: float = -2.0,
    vmax: float = 2.0,
    cmap: str = "RdBu_r",
) -> None:
    """Plot a single region's Pred vs Target smoothed matrices for a given target."""
    seq_idx = _resolve_region_index(region_labels, region)
    tname = target_names[target_index]
    reg_str = region_labels[seq_idx]

    pred = spatial_smoother(from_upper_triu(test_preds_subset[seq_idx, :, target_index], L), sigma=sigma)
    true = spatial_smoother(from_upper_triu(test_targets[seq_idx, :, target_index], L), sigma=sigma)

    with PdfPages(out_pdf) as pdf:
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        im1 = plt.matshow(pred, fignum=False, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im1, fraction=0.04, pad=0.05)
        plt.title(f"Predicted — {tname}", y=1.15)
        plt.ylabel(reg_str)

        plt.subplot(1, 2, 2)
        im2 = plt.matshow(true, fignum=False, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im2, fraction=0.04, pad=0.05)
        plt.title(f"Target — {tname}", y=1.15)

        plt.suptitle(f"{reg_str} — {tname}", fontsize=10)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(f"✅ Saved region map for '{tname}' @ {reg_str} to: {out_pdf}")


# ---- Gene band overlay helpers (configurable colors) ----
def _clip_interval(a0: int, a1: int, b0: int, b1: int) -> Optional[Tuple[int,int]]:
    """Return overlap [max(a0,b0), min(a1,b1)] if any."""
    s = max(a0, b0)
    e = min(a1, b1)
    return (s, e) if s < e else None


def _parse_region_str(region: str) -> Tuple[str, int, int]:
    # "chr19:44345344-45393920" -> ("chr19", 44345344, 45393920)
    chrom, span = region.split(":")
    s, e = span.split("-")
    return chrom, int(s), int(e)


def _genes_to_bin_spans(region_start: int, region_end: int, L: int, genes: List[Dict]):
    """
    genes: [{'name':..., 'start':..., 'end':..., 'color': '#hex' (optional)}]
    -> [(name, i0, i1, gs, ge, color_or_None), ...]
    """
    import math
    out = []
    bp_len = region_end - region_start
    if bp_len <= 0:
        return out
    bin_bp = bp_len / float(L)
    for g in (genes or []):
        ov = _clip_interval(g["start"], g["end"], region_start, region_end)
        if ov is None:
            continue
        gs, ge = ov
        i0 = max(0, min(L, int(math.floor((gs - region_start) / bin_bp))))
        i1 = max(0, min(L, int(math.ceil ((ge - region_start) / bin_bp))))
        if i1 <= i0:
            continue
        out.append((g["name"], i0, i1, gs, ge, g.get("color")))
    return out


def _overlay_gene_spans(
    ax,
    spans,
    L: int,
    default_color: str = "#F0E442",
    alpha: float = 0.22,
    lw: float = 0.9,
    label_mode: str = "gene",       # "gene" (use per-gene color) or "fixed"
    label_color: str = "#000000",   # used if label_mode == "fixed"
    label_outline: str = "white",
    label_y: float = 1.02,
    **kwargs                         # accept alias color=...
):
    """
    Draw vertical & horizontal translucent bands for each span and place labels above axes.
    Accepts alias 'color=' at callsite; maps to default_color.
    """
    if "color" in kwargs and kwargs["color"] is not None:
        default_color = kwargs["color"]

    for name, i0, i1, gs, ge, col in spans:
        band_col = col or default_color
        # bands in bin coords
        ax.axvspan(i0, i1, color=band_col, alpha=alpha, linewidth=0, zorder=3)
        ax.axhspan(i0, i1, color=band_col, alpha=alpha, linewidth=0, zorder=3)

        # label centered above axes
        x_frac = 0.5 * (i0 + i1) / float(L)
        txt_color = (band_col if label_mode == "gene" else label_color)
        ax.text(
            x_frac, label_y, name, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color=txt_color, zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground=label_outline)]
        )


def plot_region_map_gene(
    region: Union[int, str],
    target_index: int,
    region_labels: Union[pd.Series, List[str]],
    test_preds_subset: np.ndarray,
    test_targets: np.ndarray,
    L: int,
    target_names: List[str],
    out_pdf: str,
    sigma: float = 2.0,
    vmin: float = -2.0,
    vmax: float = 2.0,
    cmap: str = "RdBu_r",
    gene_marks: Optional[List[Dict]] = None,
    band_color: str = "#F0E442",
    band_alpha: float = 0.22,
    label_mode: str = "gene",            # "gene" or "fixed"
    label_color: str = "#000000",
    label_outline: str = "white",
    label_pos: str = "bottom",           # "top" or "bottom"
    label_offset: float = 0.04,
    top_adjust: float = 0.88,
    bottom_adjust: float = 0.22
) -> None:
    """
    Plot Pred vs Target matrices for a single region.
    Title is a compact, two-line header:
        <chr:start-end>
        Target <target_name> | r=<...>  MSE=<...>
    The left panel has no title; the right panel has no panel title either (cleaner).
    """
    # --- resolve region & names
    seq_idx = _resolve_region_index(region_labels, region)
    tname   = target_names[target_index]
    reg_str = region_labels[seq_idx]

    # --- reconstruct & smooth
    pred = spatial_smoother(from_upper_triu(test_preds_subset[seq_idx, :, target_index], L), sigma=sigma)
    true = spatial_smoother(from_upper_triu(test_targets[seq_idx, :, target_index], L), sigma=sigma)

    # --- per-region stats on smoothed off-diagonal
    off = ~np.eye(L, dtype=bool).reshape(-1)
    x = true.reshape(-1)[off]
    y = pred.reshape(-1)[off]
    # numeric safety
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.any():
        xv = x[valid]; yv = y[valid]
        xm, ym = xv.mean(), yv.mean()
        xzc, yzc = xv - xm, yv - ym
        denom = np.sqrt((xzc * xzc).sum() * (yzc * yzc).sum())
        r_val = float((xzc * yzc).sum() / denom) if denom > 0 else float("nan")
        mse_val = float(((yv - xv) ** 2).mean())
    else:
        r_val = float("nan")
        mse_val = float("nan")

    # --- label placement bookkeeping
    use_bottom = (label_pos.lower() == "bottom")
    label_y = -label_offset if use_bottom else 1.0 + label_offset
    subplots_adjust_kwargs = dict(bottom=bottom_adjust) if use_bottom else dict(top=top_adjust)

    # --- draw
    with PdfPages(out_pdf) as pdf:
        plt.figure(figsize=(8.8, 4.6))

        # Left: Predicted (no panel title)
        ax1 = plt.subplot(1, 2, 1)
        im1 = plt.matshow(pred, fignum=False, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im1, fraction=0.04, pad=0.05)
        # No panel title, keep clean
        # plt.title(f"Predicted — {tname}", y=1.15)
        plt.ylabel(reg_str)

        if gene_marks:
            chrom, reg_start, reg_end = _parse_region_str(reg_str)
            spans = _genes_to_bin_spans(reg_start, reg_end, L, gene_marks)
            _overlay_gene_spans(
                ax1, spans, L,
                default_color=band_color, alpha=band_alpha,
                label_mode=label_mode, label_color=label_color, label_outline=label_outline,
                label_y=label_y
            )
            for txt in ax1.texts:
                txt.set_clip_on(False)

        # Right: Target (no panel title; title consolidated in suptitle)
        ax2 = plt.subplot(1, 2, 2)
        im2 = plt.matshow(true, fignum=False, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im2, fraction=0.04, pad=0.05)
        # No panel title
        # plt.title(f"Target — {tname}", y=1.15)

        if gene_marks:
            chrom, reg_start, reg_end = _parse_region_str(reg_str)
            spans = _genes_to_bin_spans(reg_start, reg_end, L, gene_marks)
            _overlay_gene_spans(
                ax2, spans, L,
                default_color=band_color, alpha=band_alpha,
                label_mode=label_mode, label_color=label_color, label_outline=label_outline,
                label_y=label_y
            )
            for txt in ax2.texts:
                txt.set_clip_on(False)

        for ax in (ax1, ax2):
            ax.set_xlim(0, L); ax.set_ylim(L, 0)
            ax.set_xlabel("bins")

        # --- compact 2-line title with stats
        # Example:
        # chr6:106461184-107509760
        # Target NPC_control_male | r=0.742  MSE=0.1123
        plt.suptitle(
            f"{reg_str}\nTarget {tname} | r={r_val:.3f}  MSE={mse_val:.4f}",
            fontsize=11
        )
        plt.tight_layout()
        plt.subplots_adjust(**subplots_adjust_kwargs)
        pdf.savefig()
        plt.close()

    print(f"✅ Saved region map (with stats) for '{tname}' @ {reg_str} to: {out_pdf}")



# ----------------------------
# Plotter: KDE scatter + hist with global overlays
# ----------------------------
def _kde_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xy = np.vstack([x, y])
    return gaussian_kde(xy)(xy)

def _kde_background(x: np.ndarray, y: np.ndarray, gridsize: int = 150):
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    X, Y = np.mgrid[xmin:xmax:complex(gridsize), ymin:ymax:complex(gridsize)]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(gaussian_kde(np.vstack([x, y]))(positions).T, X.shape)
    return X, Y, Z

def plot_smoothed_eval_kde(
    smoothed_metrics_csv: str,
    out_pdf: str,
    targets_file: Optional[str] = None,
    global_ref: Optional[Dict[str, Dict[str, float]]] = None,
    y_min: float = -0.4,
    y_max: float = 1.0,
    color_mode: str = "kde_scatter",      # "kde_scatter", "kde_contour_scatter", "plain_scatter"
    kde_grid: int = 150,
    point_size: int = 12,
    alpha_pts: float = 0.85,
    hist_color_mse: str = "steelblue",
    hist_color_r: str = "salmon",
    scatter_cmap: str = "viridis",
    contour_cmap: str = "Blues",
    vline_color: str = "black",
    hline_color: str = "black",
    box_face: str = "white",
    box_edge: str = "gray",
    box_alpha: float = 0.9,
):
    """
    Recreate 'smoothed evaluation' scatter+hist plots with:
    - KDE colored points (or plain)
    - Fixed y-range
    - Global MSE/r lines (no inline labels)
    - Bottom-right legend-like box with global (or mean) values
    All colors/colormaps are configurable via parameters.
    """
    df = pd.read_csv(smoothed_metrics_csv)

    # ensure numeric
    metric_cols = [c for c in df.columns if c.startswith(("MSE_", "Pearson_", "Spearman_", "R2_"))]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # derive target names from CSV
    targets_from_csv = sorted({re.sub(r"^(MSE_|Pearson_|Spearman_|R2_)", "", c) for c in metric_cols})
    if not targets_from_csv:
        if not targets_file:
            raise RuntimeError("Cannot derive targets; provide targets_file")
        targets_from_csv = pd.read_csv(targets_file, sep="\t")["identifier"].tolist()

    with PdfPages(out_pdf) as pdfh:
        for tname in targets_from_csv:
            mse_col = f"MSE_{tname}"
            pr_col  = f"Pearson_{tname}"
            if mse_col not in df.columns or pr_col not in df.columns:
                continue

            sub = df[[mse_col, pr_col]].dropna(how="any")
            x = sub[mse_col].values
            y = sub[pr_col].values

            mse_mean = float(np.nanmean(x)) if x.size else np.nan
            r_mean   = float(np.nanmean(y)) if y.size else np.nan

            g_mse = None
            g_r   = None
            if global_ref is not None and tname in global_ref:
                g_mse = global_ref[tname].get("mse", None)
                g_r   = global_ref[tname].get("corr", None)

            fig = plt.figure(figsize=(8.5, 8.5))
            gs = plt.GridSpec(4, 4, figure=fig, hspace=0.28, wspace=0.28)
            ax_scatter = fig.add_subplot(gs[1:4, 0:3])
            ax_histx   = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
            ax_histy   = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)

            # KDE background (optional)
            if color_mode == "kde_contour_scatter" and x.size > 2:
                Xg, Yg, Zg = _kde_background(x, y, gridsize=kde_grid)
                cf = ax_scatter.contourf(Xg, Yg, Zg, levels=12, cmap=contour_cmap, alpha=0.55)
                fig.colorbar(cf, ax=ax_scatter, fraction=0.04, pad=0.02).set_label("KDE density")

            # Points
            if x.size > 2 and color_mode in ("kde_scatter", "kde_contour_scatter"):
                z = _kde_density(x, y)
                idx = np.argsort(z)  # low density first
                sc = ax_scatter.scatter(x[idx], y[idx], c=z[idx], s=point_size, alpha=alpha_pts, cmap=scatter_cmap)
                fig.colorbar(sc, ax=ax_scatter, fraction=0.04, pad=0.02).set_label("Point density")
            else:
                ax_scatter.scatter(x, y, s=point_size, alpha=alpha_pts)

            ax_scatter.set_xlabel("MSE")
            ax_scatter.set_ylabel("Pearson r")
            ax_scatter.grid(True, alpha=0.25)
            ax_scatter.set_title(f"MSE vs Pearson Correlation — {tname}", fontsize=12)

            # fixed y-range
            ax_scatter.set_ylim(y_min, y_max)

            # global lines (no inline labels)
            if g_mse is not None and np.isfinite(g_mse):
                ax_scatter.axvline(g_mse, color=vline_color, linestyle="--", linewidth=1)
            if g_r is not None and np.isfinite(g_r):
                ax_scatter.axhline(g_r, color=hline_color, linestyle="--", linewidth=1)

            # Histograms
            sns.histplot(x, bins=40, kde=True, ax=ax_histx, color=hist_color_mse)
            ax_histx.set_ylabel("Frequency")
            ax_histx.set_xlabel("")
            plt.setp(ax_histx.get_xticklabels(), visible=False)

            sns.histplot(y=y, bins=40, kde=True, ax=ax_histy, color=hist_color_r)
            ax_histy.set_xlabel("Frequency")
            ax_histy.set_ylabel("")
            plt.setp(ax_histy.get_yticklabels(), visible=False)
            ax_histy.grid(False)
            ax_histy.set_ylim(y_min, y_max)

            # bottom-right legend-like box with global (or mean) values
            box_mse = g_mse if (g_mse is not None and np.isfinite(g_mse)) else mse_mean
            box_r   = g_r   if (g_r   is not None and np.isfinite(g_r))   else r_mean
            box_hdr = "global" if (g_mse is not None and g_r is not None) else "mean"

            ax_scatter.text(
                0.98, 0.02,
                f"MSE ({box_hdr}) = {box_mse:.4g}\n"
                f"r ({box_hdr})   = {box_r:.4g}",
                transform=ax_scatter.transAxes, ha="right", va="bottom", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc=box_face, ec=box_edge, alpha=box_alpha)
            )

            plt.tight_layout()
            pdfh.savefig(fig)
            plt.close(fig)

    print(f"✅ Saved smoothed evaluation KDE plots to: {out_pdf}")

