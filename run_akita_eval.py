#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_akita_eval.py

Comprehensive evaluation script for Akita predictions:
  1) Loads NPZ + metadata
  2) Computes GLOBAL MSE & Pearson r (smoothed, off-diagonal)
  3) Builds per-sequence Pearson (for ranking)
  4) Plots:
     - top-N maps (per target)
     - a specific region map
     - KDE 'smoothed evaluation' scatter+hist with global overlays & fixed y-range
"""

import os
import argparse
import pandas as pd
from akita_eval_utils import (
    load_npz_and_metadata,
    compute_global_stats_streaming,
    build_per_seq_pearson,
    plot_top_n_maps,
    plot_region_map,
    plot_region_map_gene,
    plot_smoothed_eval_kde,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of Akita predictions"
    )
    parser.add_argument(
        "--npz_path",
        required=True,
        help="NPZ file with predictions and targets"
    )
    parser.add_argument(
        "--bed_file1",
        required=True,
        help="First sequences.bed file"
    )
    parser.add_argument(
        "--bed_file2",
        default=None,
        help="Second sequences.bed file (optional, for concatenated data)"
    )
    parser.add_argument(
        "--targets_file",
        required=True,
        help="Targets file (targets.txt)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--file_prefix",
        required=True,
        help="Prefix for output files"
    )
    parser.add_argument(
        "--smoothed_metrics_csv",
        default=None,
        help="Path to smoothed metrics CSV (if None, will be inferred from file_prefix)"
    )
    
    # Plot parameters
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma (default: 2.0)"
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-2.0,
        help="Minimum value for colormap (default: -2.0)"
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=2.0,
        help="Maximum value for colormap (default: 2.0)"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top regions to plot (default: 5)"
    )
    parser.add_argument(
        "--specific_region",
        default=None,
        help="Specific region to plot (e.g., 'chr1:100000-200000' or integer index). If None, uses first test region"
    )
    parser.add_argument(
        "--target_index",
        type=int,
        default=0,
        help="Target index for specific region plot (default: 0)"
    )
    
    # KDE plot parameters
    parser.add_argument(
        "--y_min",
        type=float,
        default=-0.4,
        help="Fixed y-axis minimum for KDE plots (default: -0.4)"
    )
    parser.add_argument(
        "--y_max",
        type=float,
        default=1.0,
        help="Fixed y-axis maximum for KDE plots (default: 1.0)"
    )
    parser.add_argument(
        "--color_mode",
        default="kde_scatter",
        choices=["kde_scatter", "kde_contour_scatter", "plain_scatter"],
        help="Color mode for KDE plots (default: kde_scatter)"
    )
    parser.add_argument(
        "--kde_grid",
        type=int,
        default=150,
        help="KDE grid resolution (default: 150)"
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=12,
        help="Point size for scatter plots (default: 12)"
    )
    parser.add_argument(
        "--alpha_pts",
        type=float,
        default=0.85,
        help="Alpha transparency for points (default: 0.85)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.npz_path):
        raise FileNotFoundError(f"NPZ file not found: {args.npz_path}")
    if not os.path.exists(args.bed_file1):
        raise FileNotFoundError(f"BED file 1 not found: {args.bed_file1}")
    if args.bed_file2 and not os.path.exists(args.bed_file2):
        raise FileNotFoundError(f"BED file 2 not found: {args.bed_file2}")
    if not os.path.exists(args.targets_file):
        raise FileNotFoundError(f"Targets file not found: {args.targets_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up file paths
    bed_file2 = args.bed_file2 if args.bed_file2 else args.bed_file1
    
    # Set up output file paths
    global_metrics_csv = os.path.join(args.output_dir, f"{args.file_prefix}_global_metrics.csv")
    topN_pdf_template = os.path.join(args.output_dir, f"{args.file_prefix}_top{{N}}_maps_{{tname}}.pdf")
    region_pdf_template = os.path.join(args.output_dir, f"{args.file_prefix}_{{tname}}_{{safe_region}}.pdf")
    
    if args.smoothed_metrics_csv:
        smoothed_metrics_csv = args.smoothed_metrics_csv
    else:
        smoothed_metrics_csv = os.path.join(args.output_dir, f"{args.file_prefix}_smoothed_metrics.csv")
    
    smoothed_plots_pdf = os.path.join(args.output_dir, f"{args.file_prefix}_smoothed_evaluation_plots.pdf")
    
    # 1) Load
    print("Loading predictions and metadata...")
    (test_preds_subset,
     test_targets,
     region_labels,
     target_names,
     L) = load_npz_and_metadata(args.npz_path, args.bed_file1, bed_file2, args.targets_file)
    
    print(f"N={test_targets.shape[0]}, L={L}, Targets={len(target_names)}")
    print("Example target names:", target_names[:3])
    
    # 2) GLOBAL MSE & r (streaming)
    print("\n[Step 2] Computing global metrics (smoothed, off-diagonal)...")
    global_stats = compute_global_stats_streaming(
        test_preds_subset=test_preds_subset,
        test_targets=test_targets,
        L=L,
        sigma=args.sigma,
        target_names=target_names
    )
    pd.DataFrame.from_dict(global_stats, orient="index").reset_index().rename(
        columns={"index": "target"}
    ).to_csv(global_metrics_csv, index=False)
    print("Saved global metrics to:", global_metrics_csv)
    for tn, d in global_stats.items():
        print(f"  {tn}: GLOBAL MSE={d['mse']:.4f}, GLOBAL r={d['corr']:.4f}")
    
    # 3) Per-sequence Pearson (for ranking)
    print("\n[Step 3] Building per-sequence Pearson for ranking...")
    metrics_df_smooth = build_per_seq_pearson(
        test_preds_subset=test_preds_subset,
        test_targets=test_targets,
        L=L,
        target_names=target_names,
        sigma=args.sigma,
        region_labels=region_labels
    )
    print("  metrics_df_smooth columns:", list(metrics_df_smooth.columns)[:6], "...")
    
    # 4a) Top-N maps by Pearson for each target
    print(f"\n[Step 4a] Plotting top-{args.top_n} regions by Pearson for each target...")
    for t_idx, tname in enumerate(target_names):
        out_pdf = topN_pdf_template.format(N=args.top_n, tname=tname)
        plot_top_n_maps(
            metrics_df_smooth=metrics_df_smooth,
            region_labels=region_labels,
            test_preds_subset=test_preds_subset,
            test_targets=test_targets,
            L=L,
            target_names=target_names,
            target_index=t_idx,
            out_pdf=out_pdf,
            top_n=args.top_n,
            sigma=args.sigma,
            vmin=args.vmin,
            vmax=args.vmax,
        )
    
    # 4b) A specific region map (by label or index)
    print("\n[Step 4b] Plotting a specific region map...")
    if args.specific_region is None:
        specific_region = region_labels.iloc[0]  # default to the first test region
    else:
        # Try to parse as integer index
        try:
            idx = int(args.specific_region)
            specific_region = region_labels.iloc[idx]
        except ValueError:
            # Assume it's a region string
            specific_region = args.specific_region
    
    safe_reg = str(specific_region).replace(":", "_").replace("-", "_")
    region_pdf = region_pdf_template.format(
        tname=target_names[args.target_index],
        safe_region=safe_reg
    )
    
    plot_region_map(
        region=specific_region,
        target_index=args.target_index,
        region_labels=region_labels,
        test_preds_subset=test_preds_subset,
        test_targets=test_targets,
        L=L,
        target_names=target_names,
        out_pdf=region_pdf,
        sigma=args.sigma,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    
    # 4c) KDE 'smoothed evaluation' scatter+hist with global overlays
    if os.path.exists(smoothed_metrics_csv):
        print("\n[Step 4c] Plotting KDE smoothed evaluation with global overlays...")
        plot_smoothed_eval_kde(
            smoothed_metrics_csv=smoothed_metrics_csv,
            out_pdf=smoothed_plots_pdf,
            targets_file=args.targets_file,
            global_ref=global_stats,
            y_min=args.y_min,
            y_max=args.y_max,
            color_mode=args.color_mode,
            kde_grid=args.kde_grid,
            point_size=args.point_size,
            alpha_pts=args.alpha_pts,
            hist_color_mse="steelblue",
            hist_color_r="salmon",
        )
    else:
        print(f"\n⚠️ Skipping KDE plots: {smoothed_metrics_csv} not found")
    
    print("\n✅ Done.")


if __name__ == "__main__":
    main()

