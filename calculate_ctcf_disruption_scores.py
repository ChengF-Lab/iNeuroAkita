#!/usr/bin/env python3
"""
Calculate disruption scores comparing reference and mutated sequence predictions.

This script computes disruption scores and signal changes between reference and
mutated predictions, then outputs a sorted TSV file with the results.
"""

import os
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate disruption scores from reference and mutated predictions"
    )
    parser.add_argument(
        "--ref_npz",
        required=True,
        help="NPZ file with reference predictions (should contain 'predictions' and 'region_names' arrays)"
    )
    parser.add_argument(
        "--mut_npz",
        required=True,
        help="NPZ file with mutated predictions (should contain 'predictions_avg' or 'predictions' and 'region_names' arrays)"
    )
    parser.add_argument(
        "--targets_file",
        required=True,
        help="Targets file (targets.txt) for target names"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--file_prefix",
        default="mutated",
        help="Prefix for output files (default: 'mutated')"
    )
    parser.add_argument(
        "--mut_key",
        default="predictions_avg",
        help="Key name for mutated predictions in NPZ file (default: 'predictions_avg')"
    )
    parser.add_argument(
        "--ref_key",
        default="predictions",
        help="Key name for reference predictions in NPZ file (default: 'predictions')"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input files
    if not os.path.exists(args.ref_npz):
        raise FileNotFoundError(f"Reference NPZ file not found: {args.ref_npz}")
    if not os.path.exists(args.mut_npz):
        raise FileNotFoundError(f"Mutated NPZ file not found: {args.mut_npz}")
    if not os.path.exists(args.targets_file):
        raise FileNotFoundError(f"Targets file not found: {args.targets_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions and metadata
    print("Loading predictions...")
    mut_data = np.load(args.mut_npz)
    ref_data = np.load(args.ref_npz)
    
    # Get predictions (try different key names)
    if args.mut_key in mut_data:
        pred_mut = mut_data[args.mut_key]
    elif "predictions" in mut_data:
        pred_mut = mut_data["predictions"]
    else:
        available_keys = list(mut_data.keys())
        raise KeyError(f"Key '{args.mut_key}' not found in mutated NPZ. Available keys: {available_keys}")
    
    if args.ref_key in ref_data:
        pred_ref = ref_data[args.ref_key]
    elif "predictions" in ref_data:
        pred_ref = ref_data["predictions"]
    else:
        available_keys = list(ref_data.keys())
        raise KeyError(f"Key '{args.ref_key}' not found in reference NPZ. Available keys: {available_keys}")
    
    # Load targets if available
    targets = ref_data.get("targets", None)
    
    print(f"Mutated prediction shape: {pred_mut.shape}")
    print(f"Reference prediction shape: {pred_ref.shape}")
    if targets is not None:
        print(f"Target shape: {targets.shape}")
    
    # Load sequence region info
    region_labels1 = ref_data.get("region_names", None)
    region_labels2 = mut_data.get("region_names", None)
    
    if region_labels1 is None or region_labels2 is None:
        # Try alternative key names
        region_labels1 = ref_data.get("region_labels", None)
        region_labels2 = mut_data.get("region_labels", None)
    
    if region_labels1 is None or region_labels2 is None:
        raise ValueError("Could not find region_names in NPZ files. Expected keys: 'region_names' or 'region_labels'")
    
    # Check if the arrays are exactly the same
    if np.array_equal(region_labels1, region_labels2):
        print("✅ Region labels match.")
        region_labels = region_labels1  # Use either
    else:
        raise ValueError("❌ Region label mismatch between reference and mutated predictions!")
    
    # Load target names
    print("Loading target names...")
    targets_df = pd.read_csv(args.targets_file, sep="\t")
    target_names = targets_df["identifier"].tolist()
    
    # Compute Disruption Score and ΔSignal
    print("Computing disruption scores...")
    N, P, T = pred_ref.shape  # N regions, P pixels, T targets
    
    disruption_scores = np.mean((pred_ref - pred_mut) ** 2, axis=1)  # Shape (N, T)
    signal_changes = np.mean(pred_ref ** 2, axis=1) - np.mean(pred_mut ** 2, axis=1)  # Shape (N, T)
    
    # Compute mean across targets for each region
    disruption_mean = disruption_scores.mean(axis=1)  # Shape (N,)
    signal_change_mean = signal_changes.mean(axis=1)
    
    # Save Output as DataFrame
    df_out = pd.DataFrame({
        "Region": region_labels,
        "Disruption_Score_Mean": disruption_mean,
        "Delta_Signal_Mean": signal_change_mean
    })
    
    # Optional: Add per-target columns (commented out by default)
    # for t in range(T):
    #     df_out[f"Disruption_Score_{target_names[t]}"] = disruption_scores[:, t]
    #     df_out[f"Delta_Signal_{target_names[t]}"] = signal_changes[:, t]
    
    # Sort by Disruption Score (descending)
    df_out_sorted = df_out.sort_values("Disruption_Score_Mean", ascending=False)
    
    # Save TSV File
    tsv_out = os.path.join(args.output_dir, f"{args.file_prefix}_CTCF_disruption_scores_sorted.tsv")
    df_out_sorted.to_csv(tsv_out, sep="\t", index=False)
    
    print(f"✅ Sorted output saved to: {tsv_out}")
    print(f"   Total regions: {len(df_out_sorted)}")
    print(f"   Mean disruption score: {disruption_mean.mean():.6f}")
    print(f"   Max disruption score: {disruption_mean.max():.6f}")


if __name__ == "__main__":
    main()
