#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved ChIP-seq and Contact Map Plotting Tool

This is the main plotting library. It provides functions for creating publication-quality plots showing:
- ChIP-seq signal (top panel)
- Target, Predicted, Mutated contact maps
- Difference map (Mutated - Predicted or Mutated - Target)
- Gene annotations with labels
- Color bars for visualization scales
- Automatic alignment of x-axes across all panels

USAGE:
    Method 1: Basic usage with PlotConfig
    -------------------------------------
    from Plot_ChIP_Map_Improved_v1 import PlotConfig, load_data, plot_one_region
    
    config = PlotConfig(
        ctrl_npz_path="...",
        mut_npz_path="...",
        bw_path="...",
        output_dir="...",
        target_index=0,
        target_name="NPC_control_male",
        suptitle_y=0.975,  # Title closer to plots
        top_margin=0.96,   # Less top space
    )
    region_labels, targets, preds_ctrl, preds_mut, bw = load_data(config)
    plot_one_region(region="chr8:...", config=config, ...)
    
    Method 2: Quick plot (simplest)
    --------------------------------
    from Plot_ChIP_Map_Improved_v5_test import quick_plot
    
    quick_plot(
        region="chr8:80656384-81704960",
        ctrl_npz_path="...",
        mut_npz_path="...",
        bw_path="...",
        output_path="output.pdf",
        gene_marks=[...]
    )
    
    Method 3: JSON config file
    ---------------------------
    from Plot_ChIP_Map_Improved_v5_test import load_config_from_json, save_config_to_json
    
    config = load_config_from_json("plot_config.json")
    # Use config as before
    
    Method 4: Separate running script
    ----------------------------------
    Create run_plot_chip_map.py that imports and uses these functions.
    See run_plot_chip_map.py for complete examples.

For detailed documentation, see README_Plot_ChIP_Map.md
"""

import os
import re
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
import pyBigWig
import matplotlib.patheffects as pe


# =====================
# Configuration Class
# =====================
@dataclass
class PlotConfig:
    """Configuration class for plot parameters."""
    # File paths
    ctrl_npz_path: str
    mut_npz_path: str
    bw_path: str
    output_dir: str
    
    # Data parameters
    target_index: int = 0
    target_name: str = "NPC_control_male"
    
    # Plot appearance
    figsize: Tuple[float, float] = (6.2, 22)
    title_fontsize: int = 10
    panel_fontsize: int = 9
    label_fontsize: int = 8
    
    # Color scales
    vmin_main: float = -2.0
    vmax_main: float = 2.0
    vmin_diff: float = -1.0
    vmax_diff: float = 1.0
    cmap_main: str = "RdBu_r"
    cmap_diff: str = "RdBu_r"
    
    # Smoothing
    sigma: float = 2.0
    
    # Difference mode
    diff_mode: str = "pred"  # "pred" => Mutated - Predicted, "target" => Mutated - Target
    
    # Signal-weighted diff map (to de-emphasize low-signal regions)
    diff_signal_threshold: float = 0.5  # Threshold for signal weighting (0.0 = no weighting, higher = more aggressive)
    
    # Layout
    height_ratios: List[float] = field(default_factory=lambda: [1, 6, 6, 6, 6, 0.8])
    hspace: float = 0.32
    
    # Title spacing
    suptitle_y: float = 0.97  # Title vertical position (0.95-0.99, higher = closer to top)
    subtitle_max_len: int = 50
    save_pad_inches: float = 0.10  # Padding when saving (increased to prevent title overlap)
    top_margin: float = 0.80  # Top margin adjustment (0.75-0.85, lower = more space for title)
    
    # Gene annotation
    gene_band_alpha: float = 0.25
    gene_label_offset: float = 0.05
    gene_label_fontsize: int = 7
    gene_label_outline: str = "white"  # Outline color for labels (white for dark labels)
    gene_label_color: str = "#000000"  # Label text color (black for good contrast)
    
    # ChIP-seq
    chip_linewidth: float = 0.7
    chip_linecolor: str = "black"
    
    # Colorbar
    colorbar_fraction: float = 0.8
    colorbar_pad: float = 0.2
    colorbar_aspect: float = 25
    colorbar_wspace: float = 0.25
    
    # Show metrics in subtitle
    show_metrics: bool = True
    show_mse: bool = True
    
    # Panels to show
    show_target: bool = True
    show_predicted: bool = True
    show_mutated: bool = True
    show_diff: bool = True
    show_chip: bool = True
    show_colorbars: bool = True


# =====================
# Helper Functions
# =====================
def from_upper_triu(vec: np.ndarray, L: int) -> np.ndarray:
    """Reconstruct symmetric matrix from upper triangular vector."""
    mat = np.zeros((L, L))
    triu_idx = np.triu_indices(L)
    mat[triu_idx] = vec
    return mat + mat.T - np.diag(np.diag(mat))


def spatial_smoother(mat: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian smoothing while preserving diagonal."""
    smoothed = gaussian_filter(mat, sigma=sigma)
    return np.where(np.eye(mat.shape[0]), mat, smoothed)


def _offdiag(a: np.ndarray) -> np.ndarray:
    """Extract off-diagonal elements."""
    L = a.shape[0]
    m = ~np.eye(L, dtype=bool)
    return a[m]


def pearson_offdiag(A: np.ndarray, B: np.ndarray) -> float:
    """Calculate Pearson correlation of off-diagonal elements."""
    x = _offdiag(A).astype(np.float64)
    y = _offdiag(B).astype(np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    x, y = x[m], y[m]
    x -= x.mean()
    y -= y.mean()
    den = np.sqrt((x*x).sum() * (y*y).sum())
    return (x*y).sum() / den if den > 0 else np.nan


def mse_offdiag(A: np.ndarray, B: np.ndarray) -> float:
    """Calculate MSE of off-diagonal elements."""
    x = _offdiag(A).astype(np.float64)
    y = _offdiag(B).astype(np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return np.nan
    d = y[m] - x[m]
    return float(np.mean(d*d))


def metrics_subtitle(t_mat: np.ndarray, p_mat: np.ndarray, m_mat: np.ndarray, 
                     show_mse: bool = True) -> Tuple[str, str, str]:
    """Generate metrics strings for subtitle."""
    r_tp = pearson_offdiag(t_mat, p_mat)
    m_tp = mse_offdiag(t_mat, p_mat)
    r_tm = pearson_offdiag(t_mat, m_mat)
    m_tm = mse_offdiag(t_mat, m_mat)
    r_pm = pearson_offdiag(p_mat, m_mat)
    m_pm = mse_offdiag(p_mat, m_mat)
    
    if show_mse:
        s_tp = f"Target–Pred r={r_tp:.3f}, M={m_tp:.3f}"
        s_tm = f"Target–Mutated r={r_tm:.3f}, M={m_tm:.3f}"
        s_pm = f"Predicted–Mutated r={r_pm:.3f}, M={m_pm:.3f}"
    else:
        s_tp = f"Target–Pred r={r_tp:.3f}"
        s_tm = f"Target–Mutated r={r_tm:.3f}"
        s_pm = f"Predicted–Mutated r={r_pm:.3f}"
    
    return s_tp, s_tm, s_pm


def wrap_metrics_by_pipe(subtitle: str, max_len: int = 50) -> str:
    """Wrap subtitle by splitting on pipe characters."""
    parts = re.split(r'\s*\|\s*', subtitle.strip())
    lines = []
    cur = ""
    for p in parts:
        if not cur:
            cur = p
        elif len(cur) + 3 + len(p) <= max_len:  # +3 for " | "
            cur = f"{cur} | {p}"
        else:
            lines.append(cur)
            cur = p
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def add_gene_spans_and_labels(
    ax: plt.Axes,
    region_start: int,
    region_end: int,
    gene_marks: List[Dict],
    band_alpha: float = 0.25,
    label_offset_axes: float = 0.05,
    label_fontsize: int = 7,
    label_outline: str = "white",
    label_color_fallback: str = "#111111",
) -> None:
    """
    Overlay vertical spans in genomic coordinates and place labels below the axis.
    
    Args:
        ax: Map axis where imshow(extent=[start,end,start,end]) was used
        region_start: Start of genomic region
        region_end: End of genomic region
        gene_marks: List of dicts with keys 'name', 'start', 'end', 'color'
        band_alpha: Transparency of gene bands
        label_offset_axes: Offset below axis (axes fraction)
        label_fontsize: Font size for gene labels
        label_outline: Outline color for labels
        label_color_fallback: Default color for labels
    """
    width = region_end - region_start
    if width <= 0 or not gene_marks:
        return

    # 1) Translucent vertical bands (in data coordinates for exact alignment)
    # Use black color for all gene bands for consistency
    for g in gene_marks:
        gs = max(region_start, int(g["start"]))
        ge = min(region_end, int(g["end"]))
        if ge <= region_start or gs >= region_end:
            continue
        # Always use black color for gene bands (ignore gene's color)
        ax.axvspan(gs, ge, color="black", alpha=band_alpha, lw=0, zorder=2)

    # 2) Labels below the axis
    x0, x1 = region_start, region_end
    for g in gene_marks:
        gs = max(region_start, int(g["start"]))
        ge = min(region_end, int(g["end"]))
        if ge <= region_start or gs >= region_end:
            continue

        mid_bp = 0.5 * (gs + ge)
        x_frac = (mid_bp - x0) / (x1 - x0)
        name = str(g.get("name", ""))
        # Always use label_color_fallback for text (dark color) for good contrast
        # regardless of gene band color (band can be yellow, but label is always dark)
        lcol = label_color_fallback

        txt = ax.text(
            x_frac, -label_offset_axes, name,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=label_fontsize, color=lcol, clip_on=False, zorder=5,
            weight='bold'  # Make text bolder for better visibility
        )
        if label_outline:
            # Use thicker outline for better contrast
            txt.set_path_effects([
                pe.Stroke(linewidth=3.0, foreground=label_outline),
                pe.Normal()
            ])


def parse_region(region: str) -> Tuple[str, int, int]:
    """Parse region string into chromosome, start, end."""
    chrom, coords = region.split(":")
    start, end = map(int, coords.split("-"))
    return chrom, start, end


def load_config_from_json(json_path: str) -> PlotConfig:
    """Load configuration from JSON file."""
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    # Convert figsize list to tuple if needed
    if 'figsize' in config_dict and isinstance(config_dict['figsize'], list):
        config_dict['figsize'] = tuple(config_dict['figsize'])
    # Convert height_ratios list if needed
    if 'height_ratios' in config_dict and isinstance(config_dict['height_ratios'], list):
        config_dict['height_ratios'] = config_dict['height_ratios']
    return PlotConfig(**config_dict)


def save_config_to_json(config: PlotConfig, json_path: str) -> None:
    """Save configuration to JSON file."""
    config_dict = asdict(config)
    # Convert tuples to lists for JSON compatibility
    if 'figsize' in config_dict and isinstance(config_dict['figsize'], tuple):
        config_dict['figsize'] = list(config_dict['figsize'])
    with open(json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_data(config: PlotConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Load all required data from NPZ files and bigWig file.
    
    Returns:
        region_labels: Array of region names
        targets: Target contact maps (or None)
        preds_ctrl: Predicted contact maps
        preds_mut: Mutated contact maps
        bw: BigWig file object
    """
    # Load NPZ files
    ctrl_data = np.load(config.ctrl_npz_path)
    mut_data = np.load(config.mut_npz_path)
    
    region_labels = ctrl_data["region_names"]
    targets = ctrl_data.get("targets", None)
    preds_ctrl = ctrl_data["predictions"]
    preds_mut = mut_data["predictions_avg"]
    
    # Load bigWig
    if not os.path.exists(config.bw_path):
        raise FileNotFoundError(f"BigWig file not found: {config.bw_path}")
    bw = pyBigWig.open(config.bw_path)
    
    return region_labels, targets, preds_ctrl, preds_mut, bw


def extract_region_data(
    region: str,
    region_labels: np.ndarray,
    targets: Optional[np.ndarray],
    preds_ctrl: np.ndarray,
    preds_mut: np.ndarray,
    bw: Any,
    config: PlotConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Extract data for a specific region.
    
    Returns:
        target_mat: Target contact map
        pred_mat: Predicted contact map
        mut_mat: Mutated contact map
        diff_mat: Difference map
        chip_vals: ChIP-seq values
        positions: Genomic positions for ChIP-seq
    """
    chrom, start, end = parse_region(region)
    
    # Find region index
    try:
        seq_idx = np.where(region_labels == region)[0][0]
    except IndexError:
        raise ValueError(f"Region {region} not found in region_labels")
    
    # Extract vectors
    vec_target = targets[seq_idx, :, config.target_index] if targets is not None else None
    vec_pred = preds_ctrl[seq_idx, :, config.target_index]
    vec_mut = preds_mut[seq_idx, :, config.target_index]
    
    # Calculate matrix size
    L = int((np.sqrt(8 * len(vec_pred) + 1) - 1) // 2)
    
    # Reconstruct and smooth matrices
    target_mat = spatial_smoother(from_upper_triu(vec_target, L), sigma=config.sigma) if vec_target is not None else None
    pred_mat = spatial_smoother(from_upper_triu(vec_pred, L), sigma=config.sigma)
    mut_mat = spatial_smoother(from_upper_triu(vec_mut, L), sigma=config.sigma)
    
    # Calculate difference
    if config.diff_mode.lower().startswith("p"):
        base_for_diff = pred_mat
        diff_label = "Mutated − Predicted"
    else:
        base_for_diff = target_mat if target_mat is not None else pred_mat
        diff_label = "Mutated − Target"
    diff_mat = mut_mat - base_for_diff
    
    # Apply signal-weighted scaling to diff map (similar to 02predict_ref_alt_and_score.py)
    # This de-emphasizes regions with weak signal in both maps while keeping the map continuous
    if config.diff_signal_threshold > 0:
        # Calculate signal strength (absolute values) in base and mutated maps
        signal_base = np.abs(base_for_diff)
        signal_mut = np.abs(mut_mat)
        # Use the maximum of base and mutated signal as the combined signal strength
        signal_combined = np.maximum(signal_base, signal_mut)
        
        # Calculate signal strength percentiles for weighting
        signal_percentiles = np.percentile(signal_combined, [10, 50, 90])
        signal_min, signal_med, signal_max = signal_percentiles
        
        # Normalize signal strength to [0, 1] range
        signal_norm = (signal_combined - signal_min) / (signal_max - signal_min + 1e-10)
        signal_norm = np.clip(signal_norm, 0, 1)
        
        # Apply threshold: regions below threshold get reduced weight
        threshold_val = config.diff_signal_threshold
        # Create weight: 1.0 for high signal, smoothly decreasing for low signal
        # Use a power function for smooth transition
        signal_weight = np.power(signal_norm, 1.0 / (threshold_val + 0.1))
        signal_weight = np.clip(signal_weight, 0.1, 1.0)  # Minimum 0.1 to keep some visibility
        
        # Scale diff values by signal weight (reduces intensity of low-signal regions)
        # This keeps the map continuous while de-emphasizing low-signal regions
        diff_mat = diff_mat * signal_weight
    
    # Extract ChIP-seq signal
    try:
        chip_vals = bw.values(chrom, start, end)
        positions = np.arange(start, end)
    except Exception:
        chip_vals = None
        positions = None
    
    return target_mat, pred_mat, mut_mat, diff_mat, chip_vals, positions


def plot_one_region(
    region: str,
    config: PlotConfig,
    region_labels: np.ndarray,
    targets: Optional[np.ndarray],
    preds_ctrl: np.ndarray,
    preds_mut: np.ndarray,
    bw: Any,
    gene_marks: Optional[List[Dict]] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Plot a single region with ChIP-seq signal and contact maps.
    
    Args:
        region: Region string (e.g., "chr8:80656384-81704960")
        config: PlotConfig object with all parameters
        region_labels: Array of region names
        targets: Target contact maps (or None)
        preds_ctrl: Predicted contact maps
        preds_mut: Mutated contact maps
        bw: BigWig file object
        gene_marks: List of gene annotations (optional)
        output_path: Output PDF path (optional, auto-generated if None)
    
    Returns:
        output_path: Path to saved PDF
    """
    chrom, start, end = parse_region(region)
    extent = [start, end, start, end]
    
    # Extract data
    target_mat, pred_mat, mut_mat, diff_mat, chip_vals, positions = extract_region_data(
        region, region_labels, targets, preds_ctrl, preds_mut, bw, config
    )
    
    # Calculate metrics
    if config.show_metrics and target_mat is not None:
        s_tp, s_tm, s_pm = metrics_subtitle(target_mat, pred_mat, mut_mat, config.show_mse)
        subtitle = f"{s_tp}  |  {s_tm}  |  {s_pm}"
        subtitle_wrapped = wrap_metrics_by_pipe(subtitle, max_len=config.subtitle_max_len)
    else:
        subtitle_wrapped = ""
    
    # Generate output path
    if output_path is None:
        region_safe = region.replace(":", "_").replace("-", "_")
        output_path = os.path.join(config.output_dir, f"plot_{region_safe}.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create figure
    fig = plt.figure(figsize=config.figsize)
    
    # Determine number of panels
    n_panels = sum([config.show_chip, config.show_target, config.show_predicted, 
                    config.show_mutated, config.show_diff, config.show_colorbars])
    height_ratios = []
    panel_idx = 0
    
    if config.show_chip:
        height_ratios.append(1)
    if config.show_target:
        height_ratios.append(6)
    if config.show_predicted:
        height_ratios.append(6)
    if config.show_mutated:
        height_ratios.append(6)
    if config.show_diff:
        height_ratios.append(6)
    if config.show_colorbars:
        height_ratios.append(0.8)
    
    gs = GridSpec(len(height_ratios), 1, height_ratios=height_ratios, hspace=config.hspace)
    
    current_row = 0
    ax_chip = None
    ax_ref = None  # Reference axis for alignment (first contact map panel)
    contact_map_axes = []  # List of all contact map axes for alignment
    
    # 1) ChIP-seq panel
    if config.show_chip:
        ax_chip = fig.add_subplot(gs[current_row])
        if chip_vals is not None and positions is not None:
            ax_chip.plot(positions, chip_vals, color=config.chip_linecolor, 
                        linewidth=config.chip_linewidth)
            ax_chip.set_xlim(start, end)
        else:
            ax_chip.text(0.5, 0.5, "ChIP-seq data unavailable", 
                        ha="center", va="center", transform=ax_chip.transAxes, fontsize=8)
            ax_chip.set_xlim(start, end)
        ax_chip.set_ylabel("ChIP-seq", fontsize=config.label_fontsize)
        ax_chip.set_xticks([])
        ax_chip.set_title("NPC CTCF ChIP-seq", fontsize=config.panel_fontsize)
        current_row += 1
    
    # 2) Target panel
    if config.show_target:
        ax_target = fig.add_subplot(gs[current_row])
        if target_mat is not None:
            ax_target.imshow(target_mat, cmap=config.cmap_main, 
                           vmin=config.vmin_main, vmax=config.vmax_main,
                           origin='upper', aspect='equal', extent=extent)
            ax_target.set_xlim(start, end)  # Explicitly set xlim
            # Add gene annotations if provided
            if gene_marks:
                add_gene_spans_and_labels(
                        ax_target, start, end, gene_marks,
                        band_alpha=config.gene_band_alpha,
                        label_offset_axes=config.gene_label_offset,
                        label_fontsize=config.gene_label_fontsize,
                        label_outline=config.gene_label_outline,
                        label_color_fallback=config.gene_label_color
                        )
            
            ax_target.set_title(f"Target — {config.target_name}", fontsize=config.panel_fontsize)
            ax_target.set_xlabel("Genomic position", fontsize=config.label_fontsize)
            ax_target.set_ylabel("Genomic position", fontsize=config.label_fontsize)
            if ax_ref is None:
                ax_ref = ax_target  # Use first contact map as reference
            contact_map_axes.append(ax_target)
        else:
            ax_target.text(0.5, 0.5, "Target data unavailable",
                          ha="center", va="center", transform=ax_target.transAxes, fontsize=8)
            ax_target.set_xlim(start, end)
            if ax_ref is None:
                ax_ref = ax_target
            contact_map_axes.append(ax_target)
        current_row += 1
    
    # 3) Predicted panel
    if config.show_predicted:
        ax_pred = fig.add_subplot(gs[current_row])
        ax_pred.imshow(pred_mat, cmap=config.cmap_main,
                      vmin=config.vmin_main, vmax=config.vmax_main,
                      origin='upper', aspect='equal', extent=extent)
        ax_pred.set_xlim(start, end)  # Explicitly set xlim
        # Add gene annotations if provided
        if gene_marks:
            add_gene_spans_and_labels(
                    ax_pred, start, end, gene_marks,
                    band_alpha=config.gene_band_alpha,
                    label_offset_axes=config.gene_label_offset,
                    label_fontsize=config.gene_label_fontsize,
                    label_outline=config.gene_label_outline,
                    label_color_fallback=config.gene_label_color
                    )

        ax_pred.set_title(f"Predicted — {config.target_name}", fontsize=config.panel_fontsize)
        ax_pred.set_xlabel("Genomic position", fontsize=config.label_fontsize)
        ax_pred.set_ylabel("Genomic position", fontsize=config.label_fontsize)
        if ax_ref is None:
            ax_ref = ax_pred  # Use first contact map as reference
        contact_map_axes.append(ax_pred)
        current_row += 1
    
    # 4) Mutated panel (with gene annotations)
    if config.show_mutated:
        ax_mut = fig.add_subplot(gs[current_row])
        ax_mut.imshow(mut_mat, cmap=config.cmap_main,
                     vmin=config.vmin_main, vmax=config.vmax_main,
                     origin='upper', aspect='equal', extent=extent)
        ax_mut.set_xlim(start, end)  # Explicitly set xlim
        
        # Add gene annotations if provided
        if gene_marks:
            add_gene_spans_and_labels(
                ax_mut, start, end, gene_marks,
                band_alpha=config.gene_band_alpha,
                label_offset_axes=config.gene_label_offset,
                label_fontsize=config.gene_label_fontsize,
                label_outline=config.gene_label_outline,
                label_color_fallback=config.gene_label_color
            )
        
        ax_mut.set_title(f"Mutated — {config.target_name}", fontsize=config.panel_fontsize)
        ax_mut.set_xlabel("Genomic position", fontsize=config.label_fontsize)
        ax_mut.set_ylabel("Genomic position", fontsize=config.label_fontsize)
        if ax_ref is None:
            ax_ref = ax_mut  # Use first contact map as reference
        contact_map_axes.append(ax_mut)
        current_row += 1
    
    # 5) Difference panel
    if config.show_diff:
        ax_diff = fig.add_subplot(gs[current_row])
        # Use percentile-based vmax when signal weighting is applied (similar to improved version)
        if config.diff_signal_threshold > 0:
            # Calculate vmax from 99.5th percentile of absolute values
            vmax_diff = np.percentile(np.abs(diff_mat), 99.5)
            vmin_diff = -vmax_diff  # Symmetric around zero
        else:
            # Use fixed vmin/vmax from config
            vmin_diff = config.vmin_diff
            vmax_diff = config.vmax_diff
        ax_diff.imshow(diff_mat, cmap=config.cmap_diff,
                      vmin=vmin_diff, vmax=vmax_diff,
                   origin='upper', aspect='equal', extent=extent)
        ax_diff.set_xlim(start, end)  # Explicitly set xlim
        # Add gene annotations if provided
        if gene_marks:
            add_gene_spans_and_labels(
                    ax_diff, start, end, gene_marks,
                    band_alpha=config.gene_band_alpha,
                    label_offset_axes=config.gene_label_offset,
                    label_fontsize=config.gene_label_fontsize,
                    label_outline=config.gene_label_outline,
                    label_color_fallback=config.gene_label_color
                    )
        diff_label = "Mutated − Predicted" if config.diff_mode.lower().startswith("p") else "Mutated − Target"
        ax_diff.set_title(diff_label, fontsize=config.panel_fontsize)
        ax_diff.set_xlabel("Genomic position", fontsize=config.label_fontsize)
        ax_diff.set_ylabel("Genomic position", fontsize=config.label_fontsize)
        if ax_ref is None:
            ax_ref = ax_diff  # Use first contact map as reference
        contact_map_axes.append(ax_diff)
        current_row += 1
    
    # 6) Color bars
    if config.show_colorbars:
        ax_cbrow = fig.add_subplot(gs[current_row])
        ax_cbrow.axis("off")
        sub_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[current_row], 
                                         wspace=config.colorbar_wspace, width_ratios=[1, 1])
        
        # Main colorbar
        from matplotlib.cm import get_cmap
        sm_main = plt.cm.ScalarMappable(cmap=get_cmap(config.cmap_main), 
                                       norm=Normalize(config.vmin_main, config.vmax_main))
        sm_main.set_array([])
        ax_cb1 = fig.add_subplot(sub_gs[0])
        ax_cb1.axis("off")
        cbar1 = fig.colorbar(sm_main, ax=ax_cb1, orientation="horizontal",
                            fraction=config.colorbar_fraction, pad=config.colorbar_pad,
                            aspect=config.colorbar_aspect)
        cbar1.set_label("Main map scale", fontsize=config.label_fontsize)
        
        # Diff colorbar
        sm_diff = plt.cm.ScalarMappable(cmap=get_cmap(config.cmap_diff),
                                       norm=Normalize(config.vmin_diff, config.vmax_diff))
        sm_diff.set_array([])
        ax_cb2 = fig.add_subplot(sub_gs[1])
        ax_cb2.axis("off")
        cbar2 = fig.colorbar(sm_diff, ax=ax_cb2, orientation="horizontal",
                            fraction=config.colorbar_fraction, pad=config.colorbar_pad,
                            aspect=config.colorbar_aspect)
        cbar2.set_label("Diff map scale", fontsize=config.label_fontsize)
    
    # ---------- Align ChIP-seq panel width to contact map panels (like CLEANED.py) ----------
    # Align right after creating all axes, BEFORE subplots_adjust
    # Use first contact map panel as reference (not ChIP panel)
    if config.show_chip and ax_ref is not None:
        fig.canvas.draw()  # Ensure positions are calculated
        box_ref = ax_ref.get_position()
        # Align ChIP-seq panel to match reference contact map panel width
        ax_chip.set_position([box_ref.x0, ax_chip.get_position().y0,
                             box_ref.width, ax_chip.get_position().height])
        # Ensure all contact map axes have same xlim
        reference_xlim = ax_ref.get_xlim()
        for ax in contact_map_axes:
            ax.set_xlim(reference_xlim)
    
    # Adjust top margin BEFORE setting title (lower value = more space for title)
    plt.subplots_adjust(top=config.top_margin)
    
    # Re-align after subplots_adjust (it may have shifted positions slightly)
    if config.show_chip and ax_ref is not None:
        fig.canvas.draw()
        box_ref = ax_ref.get_position()
        ax_chip.set_position([box_ref.x0, ax_chip.get_position().y0,
                             box_ref.width, ax_chip.get_position().height])
        # Re-sync xlim after layout adjustments
        reference_xlim = ax_ref.get_xlim()
        ax_chip.set_xlim(reference_xlim)
        for ax in contact_map_axes:
            ax.set_xlim(reference_xlim)
    
    # Title and subtitle (positioned with sufficient space above ChIP-seq panel)
    # With top_margin=0.80, the plot area ends at 80% of figure height, leaving 20% for title
    # Position title at suptitle_y (default 0.97) which is safely above the plot area
    if subtitle_wrapped:
        fig.suptitle(f"{region}\n{subtitle_wrapped}", fontsize=config.title_fontsize, 
                    y=config.suptitle_y)
    else:
        fig.suptitle(region, fontsize=config.title_fontsize, y=config.suptitle_y)
    
    # Save with minimal padding to keep title close
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=config.save_pad_inches)
    plt.close(fig)
    
    return output_path


def plot_multi_region(
    regions: List[str],
    config: PlotConfig,
    region_labels: np.ndarray,
    targets: Optional[np.ndarray],
    preds_ctrl: np.ndarray,
    preds_mut: np.ndarray,
    bw: Any,
    gene_marks_by_region: Optional[Dict[str, List[Dict]]] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Plot multiple regions in a single PDF.
    
    Args:
        regions: List of region strings
        config: PlotConfig object
        region_labels: Array of region names
        targets: Target contact maps (or None)
        preds_ctrl: Predicted contact maps
        preds_mut: Mutated contact maps
        bw: BigWig file object
        gene_marks_by_region: Dictionary mapping region -> gene_marks list
        output_path: Output PDF path (optional)
    
    Returns:
        output_path: Path to saved PDF
    """
    if output_path is None:
        output_path = os.path.join(config.output_dir, "multi_region_plot.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        for region in regions:
            try:
                gene_marks = gene_marks_by_region.get(region) if gene_marks_by_region else None
                
                # Extract data for this region (don't create separate PDF)
                chrom, start, end = parse_region(region)
                extent = [start, end, start, end]
                
                target_mat, pred_mat, mut_mat, diff_mat, chip_vals, positions = extract_region_data(
                    region, region_labels, targets, preds_ctrl, preds_mut, bw, config
                )
                
                # Calculate metrics
                if config.show_metrics and target_mat is not None:
                    s_tp, s_tm, s_pm = metrics_subtitle(target_mat, pred_mat, mut_mat, config.show_mse)
                    subtitle = f"{s_tp}  |  {s_tm}  |  {s_pm}"
                    subtitle_wrapped = wrap_metrics_by_pipe(subtitle, max_len=config.subtitle_max_len)
                else:
                    subtitle_wrapped = ""
                
                # Create figure
                fig = plt.figure(figsize=config.figsize)
                height_ratios = []
                if config.show_chip:
                    height_ratios.append(1)
                if config.show_target:
                    height_ratios.append(6)
                if config.show_predicted:
                    height_ratios.append(6)
                if config.show_mutated:
                    height_ratios.append(6)
                if config.show_diff:
                    height_ratios.append(6)
                if config.show_colorbars:
                    height_ratios.append(0.8)
                
                gs = GridSpec(len(height_ratios), 1, height_ratios=height_ratios, hspace=config.hspace)
                current_row = 0
                ax_chip = None
                ax_ref = None  # Reference axis for alignment (first contact map panel)
                contact_map_axes = []  # List of all contact map axes for alignment
                
                # ChIP
                if config.show_chip:
                    ax_chip = fig.add_subplot(gs[current_row])
                    if chip_vals is not None and positions is not None:
                        ax_chip.plot(positions, chip_vals, color=config.chip_linecolor,
                                    linewidth=config.chip_linewidth)
                        ax_chip.set_xlim(start, end)
                    else:
                        ax_chip.text(0.5, 0.5, "ChIP-seq data unavailable",
                                    ha="center", va="center", transform=ax_chip.transAxes, fontsize=8)
                        ax_chip.set_xlim(start, end)
                    ax_chip.set_ylabel("ChIP-seq", fontsize=config.label_fontsize)
                    ax_chip.set_xticks([])
                    ax_chip.set_title("NPC CTCF ChIP-seq", fontsize=config.panel_fontsize)
                    current_row += 1
                
                # Target
                if config.show_target:
                    ax_target = fig.add_subplot(gs[current_row])
                    if target_mat is not None:
                        ax_target.imshow(target_mat, cmap=config.cmap_main,
                                       vmin=config.vmin_main, vmax=config.vmax_main,
                         origin='upper', aspect='equal', extent=extent)
                        ax_target.set_xlim(start, end)  # Explicitly set xlim
                        if gene_marks:
                             add_gene_spans_and_labels(ax_target, start, end, gene_marks,
                                     band_alpha=config.gene_band_alpha,
                                     label_offset_axes=config.gene_label_offset,
                                     label_fontsize=config.gene_label_fontsize,
                                     label_outline=config.gene_label_outline,
                                     label_color_fallback=config.gene_label_color)
                        ax_target.set_title(f"Target — {config.target_name}", fontsize=config.panel_fontsize)
                        ax_target.set_xlabel("Genomic position", fontsize=config.label_fontsize)
                        ax_target.set_ylabel("Genomic position", fontsize=config.label_fontsize)
                        if ax_ref is None:
                            ax_ref = ax_target  # Use first contact map as reference
                        contact_map_axes.append(ax_target)
                    else:
                        ax_target.text(0.5, 0.5, "Target data unavailable",
                                      ha="center", va="center", transform=ax_target.transAxes, fontsize=8)
                        ax_target.set_xlim(start, end)
                        if ax_ref is None:
                            ax_ref = ax_target
                        contact_map_axes.append(ax_target)
                    current_row += 1
                
                # Predicted
                if config.show_predicted:
                    ax_pred = fig.add_subplot(gs[current_row])
                    ax_pred.imshow(pred_mat, cmap=config.cmap_main,
                                  vmin=config.vmin_main, vmax=config.vmax_main,
                       origin='upper', aspect='equal', extent=extent)
                    ax_pred.set_xlim(start, end)  # Explicitly set xlim
                    if gene_marks:
                        add_gene_spans_and_labels(ax_pred, start, end, gene_marks,
                                band_alpha=config.gene_band_alpha,
                                label_offset_axes=config.gene_label_offset,
                                label_fontsize=config.gene_label_fontsize,
                                label_outline=config.gene_label_outline,
                                label_color_fallback=config.gene_label_color)
                    ax_pred.set_title(f"Predicted — {config.target_name}", fontsize=config.panel_fontsize)
                    ax_pred.set_xlabel("Genomic position", fontsize=config.label_fontsize)
                    ax_pred.set_ylabel("Genomic position", fontsize=config.label_fontsize)
                    if ax_ref is None:
                        ax_ref = ax_pred  # Use first contact map as reference
                    contact_map_axes.append(ax_pred)
                    current_row += 1
                
                # Mutated
                if config.show_mutated:
                    ax_mut = fig.add_subplot(gs[current_row])
                    ax_mut.imshow(mut_mat, cmap=config.cmap_main,
                                 vmin=config.vmin_main, vmax=config.vmax_main,
                      origin='upper', aspect='equal', extent=extent)
                    ax_mut.set_xlim(start, end)  # Explicitly set xlim
                    if gene_marks:
                        add_gene_spans_and_labels(ax_mut, start, end, gene_marks,
                                                 band_alpha=config.gene_band_alpha,
                                                 label_offset_axes=config.gene_label_offset,
                                                 label_fontsize=config.gene_label_fontsize,
                                                 label_outline=config.gene_label_outline,
                                                 label_color_fallback=config.gene_label_color)
                    ax_mut.set_title(f"Mutated — {config.target_name}", fontsize=config.panel_fontsize)
                    ax_mut.set_xlabel("Genomic position", fontsize=config.label_fontsize)
                    ax_mut.set_ylabel("Genomic position", fontsize=config.label_fontsize)
                    if ax_ref is None:
                        ax_ref = ax_mut  # Use first contact map as reference
                    contact_map_axes.append(ax_mut)
                    current_row += 1
                
                # Diff
                if config.show_diff:
                    ax_diff = fig.add_subplot(gs[current_row])
                    # Use percentile-based vmax when signal weighting is applied (similar to improved version)
                    if config.diff_signal_threshold > 0:
                        # Calculate vmax from 99.5th percentile of absolute values
                        vmax_diff = np.percentile(np.abs(diff_mat), 99.5)
                        vmin_diff = -vmax_diff  # Symmetric around zero
                    else:
                        # Use fixed vmin/vmax from config
                        vmin_diff = config.vmin_diff
                        vmax_diff = config.vmax_diff
                    ax_diff.imshow(diff_mat, cmap=config.cmap_diff,
                                  vmin=vmin_diff, vmax=vmax_diff,
                       origin='upper', aspect='equal', extent=extent)
                    ax_diff.set_xlim(start, end)  # Explicitly set xlim
                    if gene_marks:
                         add_gene_spans_and_labels(ax_diff, start, end, gene_marks,
                                 band_alpha=config.gene_band_alpha,
                                 label_offset_axes=config.gene_label_offset,
                                 label_fontsize=config.gene_label_fontsize,
                                 label_outline=config.gene_label_outline,
                                 label_color_fallback=config.gene_label_color)
                    diff_label = "Mutated − Predicted" if config.diff_mode.lower().startswith("p") else "Mutated − Target"
                    ax_diff.set_title(diff_label, fontsize=config.panel_fontsize)
                    ax_diff.set_xlabel("Genomic position", fontsize=config.label_fontsize)
                    ax_diff.set_ylabel("Genomic position", fontsize=config.label_fontsize)
                    if ax_ref is None:
                        ax_ref = ax_diff  # Use first contact map as reference
                    contact_map_axes.append(ax_diff)
                    current_row += 1
                
                # Color bars
                if config.show_colorbars:
                    ax_cbrow = fig.add_subplot(gs[current_row])
                    ax_cbrow.axis("off")
                    sub_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[current_row],
                                                    wspace=config.colorbar_wspace, width_ratios=[1, 1])
                    from matplotlib.cm import get_cmap
                    sm_main = plt.cm.ScalarMappable(cmap=get_cmap(config.cmap_main),
                                                   norm=Normalize(config.vmin_main, config.vmax_main))
                    sm_main.set_array([])
                    ax_cb1 = fig.add_subplot(sub_gs[0])
                    ax_cb1.axis("off")
                    cbar1 = fig.colorbar(sm_main, ax=ax_cb1, orientation="horizontal",
                                        fraction=config.colorbar_fraction, pad=config.colorbar_pad,
                                        aspect=config.colorbar_aspect)
                    cbar1.set_label("Main map scale", fontsize=config.label_fontsize)
                    sm_diff = plt.cm.ScalarMappable(cmap=get_cmap(config.cmap_diff),
                                                   norm=Normalize(config.vmin_diff, config.vmax_diff))
                    sm_diff.set_array([])
                    ax_cb2 = fig.add_subplot(sub_gs[1])
                    ax_cb2.axis("off")
                    cbar2 = fig.colorbar(sm_diff, ax=ax_cb2, orientation="horizontal",
                                        fraction=config.colorbar_fraction, pad=config.colorbar_pad,
                                        aspect=config.colorbar_aspect)
                    cbar2.set_label("Diff map scale", fontsize=config.label_fontsize)
                
                # ---------- Align ChIP-seq panel width to contact map panels (like CLEANED.py) ----------
                # Align right after creating all axes, BEFORE subplots_adjust
                if config.show_chip and ax_ref is not None:
                    fig.canvas.draw()  # Ensure positions are calculated
                    box_ref = ax_ref.get_position()
                    # Align ChIP-seq panel to match reference contact map panel width
                    ax_chip.set_position([box_ref.x0, ax_chip.get_position().y0,
                                         box_ref.width, ax_chip.get_position().height])
                    # Ensure all contact map axes have same xlim
                    reference_xlim = ax_ref.get_xlim()
                    for ax in contact_map_axes:
                        ax.set_xlim(reference_xlim)
                
                # Adjust top margin BEFORE setting title (lower value = more space for title)
                plt.subplots_adjust(top=config.top_margin)
                
                # Re-align after subplots_adjust (it may have shifted positions slightly)
                if config.show_chip and ax_ref is not None:
                    fig.canvas.draw()
                    box_ref = ax_ref.get_position()
                    ax_chip.set_position([box_ref.x0, ax_chip.get_position().y0,
                                         box_ref.width, ax_chip.get_position().height])
                    # Re-sync xlim after layout adjustments
                    reference_xlim = ax_ref.get_xlim()
                    ax_chip.set_xlim(reference_xlim)
                    for ax in contact_map_axes:
                        ax.set_xlim(reference_xlim)
                
                # Title (positioned with sufficient space above ChIP-seq panel)
                # With top_margin=0.80, the plot area ends at 80% of figure height, leaving 20% for title
                # Position title at suptitle_y (default 0.97) which is safely above the plot area
                if subtitle_wrapped:
                    fig.suptitle(f"{region}\n{subtitle_wrapped}", fontsize=config.title_fontsize,
                               y=config.suptitle_y)
                else:
                    fig.suptitle(region, fontsize=config.title_fontsize, y=config.suptitle_y)
                
                # Save to multi-region PDF
                pdf.savefig(fig, bbox_inches='tight', pad_inches=config.save_pad_inches)
                plt.close(fig)
                
            except Exception as e:
                print(f"⚠️ Error plotting region {region}: {e}")
                continue
    
    return output_path


# =====================
# Convenience Wrapper Functions
# =====================
def quick_plot(
    region: str,
    ctrl_npz_path: str,
    mut_npz_path: str,
    bw_path: str,
    output_path: str,
    target_index: int = 0,
    target_name: str = "NPC_control_male",
    gene_marks: Optional[List[Dict]] = None,
    **config_overrides
) -> str:
    """
    Quick plotting function with minimal parameters.
    
    Args:
        region: Region string (e.g., "chr8:80656384-81704960")
        ctrl_npz_path: Path to control NPZ file
        mut_npz_path: Path to mutated NPZ file
        bw_path: Path to bigWig file
        output_path: Output PDF path
        target_index: Target index (default: 0)
        target_name: Target name (default: "NPC_control_male")
        gene_marks: List of gene annotations (optional)
        **config_overrides: Additional config parameters to override
    
    Returns:
        output_path: Path to saved PDF
    
    Example:
        quick_plot(
            region="chr8:80656384-81704960",
            ctrl_npz_path="...",
            mut_npz_path="...",
            bw_path="...",
            output_path="output.pdf",
            gene_marks=[{"name": "FABP5", "start": 81280536, "end": 81284775, "color": "#e6b800"}]
        )
    """
    # Create config with defaults and overrides
    config = PlotConfig(
        ctrl_npz_path=ctrl_npz_path,
        mut_npz_path=mut_npz_path,
        bw_path=bw_path,
        output_dir=os.path.dirname(output_path),
        target_index=target_index,
        target_name=target_name,
        **config_overrides
    )
    
    # Load data
    region_labels, targets_data, preds_ctrl, preds_mut, bw = load_data(config)
    
    # Plot
    return plot_one_region(
        region=region,
        config=config,
        region_labels=region_labels,
        targets=targets_data,
        preds_ctrl=preds_ctrl,
        preds_mut=preds_mut,
        bw=bw,
        gene_marks=gene_marks,
        output_path=output_path
    )


# =====================
# Example Usage (Optional - can be moved to separate script)
# =====================
# Note: For production use, it's recommended to create a separate running script
# (e.g., run_plot_chip_map.py) that imports and uses these functions.
# See run_plot_chip_map.py for examples.

if __name__ == "__main__":
    # Create configuration
    config = PlotConfig(
        # File paths
        ctrl_npz_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/predictOut_NPC_P16_CTCF_fasta/akita_fasta_predictions_targets.npz",
        mut_npz_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/predictOut_NPC_P16_mutated_CTCF_fasta/mutated_predictions_avg.npz",
        bw_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/CTCF_Mutated_disruption/NPC_CTCF_ChIP/ENCFF327FLJ.bigWig",
        output_dir="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/CTCF_Mutated_disruption/Plot",
        
        # Data parameters
        target_index=0,
        target_name="NPC_control_male",
        
        # Title spacing (defaults provide good spacing between title and ChIP-seq panel)
        # Adjust these to control title positioning if needed:
        # suptitle_y=0.995,  # Title vertical position (0.95-0.998, higher = closer to top)
        # top_margin=0.92,   # Top margin (0.85-0.95, lower = more space for title)
        # save_pad_inches=0.05,  # Padding when saving (0.02-0.1)
        
        # Other parameters use defaults - see PlotConfig class for all options
        # Examples:
        # - vmin_main=-2.0, vmax_main=2.0 (color scale)
        # - sigma=2.0 (smoothing)
        # - diff_mode="pred" (difference calculation)
        # - show_metrics=True (show correlation metrics)
        # - gene_band_alpha=0.25 (gene annotation transparency)
    )
    
    # Load data
    region_labels, targets, preds_ctrl, preds_mut, bw = load_data(config)
    
    # Define gene marks for regions
    gene_marks = [
        {"name": "FABP5", "start": 81280536, "end": 81284775, "color": "#e6b800"},
    ]
    
    gene_marks_by_region = {
        "chr8:80656384-81704960": gene_marks,
    }
    
    # Plot single region
    region = "chr8:80656384-81704960"
    output_path = plot_one_region(
        region=region,
        config=config,
        region_labels=region_labels,
        targets=targets,
        preds_ctrl=preds_ctrl,
        preds_mut=preds_mut,
        bw=bw,
        gene_marks=gene_marks,
        output_path=os.path.join(config.output_dir, "region_chr8_ChIP_ContactMap_WITH_MUT_DIFF_metrics_CBbottom.pdf")
    )
    print(f"✅ Single region PDF saved to: {output_path}")
    
    # Plot multiple regions
    regions = [
        "chr8:80656384-81704960",
        "chr5:76333056-77381632",
        "chr1:30457856-31506432",
    ]
    output_path = plot_multi_region(
        regions=regions,
        config=config,
        region_labels=region_labels,
        targets=targets,
        preds_ctrl=preds_ctrl,
        preds_mut=preds_mut,
        bw=bw,
        gene_marks_by_region=gene_marks_by_region,
        output_path=os.path.join(config.output_dir, "MultiRegion_WITH_MUT_DIFF_metrics_CBbottom.pdf")
    )
    print(f"✅ Multi-region PDF saved to: {output_path}")
    
    # =====================
    # Alternative: Using quick_plot for simple usage
    # =====================
    # Quick plot with minimal parameters:
    # quick_plot(
    #     region="chr8:80656384-81704960",
    #     ctrl_npz_path=config.ctrl_npz_path,
    #     mut_npz_path=config.mut_npz_path,
    #     bw_path=config.bw_path,
    #     output_path=os.path.join(config.output_dir, "quick_plot_output.pdf"),
    #     gene_marks=gene_marks,
    #     suptitle_y=0.975,  # Override config parameter
    #     top_margin=0.96,   # Override config parameter
    # )
    
    # =====================
    # Alternative: Using JSON config file
    # =====================
    # 1. Save current config to JSON:
    # save_config_to_json(config, "plot_config.json")
    #
    # 2. Load config from JSON:
    # config = load_config_from_json("plot_config.json")
    #
    # 3. Use the config as before
