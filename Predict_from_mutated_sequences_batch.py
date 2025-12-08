#!/usr/bin/env python3
"""
Predict Hi-C contact maps from mutated FASTA sequences and average across variants.

This script is similar to Predict_from_Seq2_batch.py but handles mutated sequences
where multiple variants exist per region (e.g., chr10:20477952-21526528_rand0, _rand1, etc.)
and averages predictions across all variants for each base region.
"""

import os
import json
import argparse
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from basenji import dna_io, seqnn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict Hi-C contact maps from mutated FASTA sequences and average across variants"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing params.json and model_best.h5"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Data directory containing statistics.json"
    )
    parser.add_argument(
        "--fasta_file",
        required=True,
        help="Input FASTA file with mutated sequences (IDs should have format: region_randN)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--file_prefix",
        default="mutated",
        help="Prefix for output files (default: 'mutated')"
    )
    parser.add_argument(
        "--expected_variants",
        type=int,
        default=10,
        help="Expected number of variants per region (default: 10). Regions with different counts will be filtered out."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for prediction (default: 8)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not os.path.exists(args.fasta_file):
        raise FileNotFoundError(f"FASTA file not found: {args.fasta_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    params_file = os.path.join(args.model_dir, 'params.json')
    model_file = os.path.join(args.model_dir, 'model_best.h5')
    stats_file = os.path.join(args.data_dir, 'statistics.json')
    
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"params.json not found in {args.model_dir}")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"model_best.h5 not found in {args.model_dir}")
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"statistics.json not found in {args.data_dir}")
    
    with open(params_file) as f:
        params = json.load(f)
        params_model = params['model']
    
    stats = json.load(open(stats_file))
    
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file)
    print("✅ Model loaded.")
    
    seq_length = stats['seq_length']
    target_length = stats['target_length']
    target_crop = stats['crop_bp'] // stats['pool_width']
    
    # Parse FASTA and Group by Base Region
    mutated_seqs = {}  # key = base region (e.g., chr10:20477952-21526528), value = list of one-hot sequences
    
    print(f"Reading mutated sequences from {args.fasta_file}...")
    for record in SeqIO.parse(args.fasta_file, "fasta"):
        full_id = record.id  # e.g., chr10:20477952-21526528_rand0
        # Try to extract base region (handle different naming conventions)
        if "_rand" in full_id:
            base_region = full_id.split("_rand")[0]
        elif "_mut" in full_id:
            base_region = full_id.split("_mut")[0]
        else:
            # If no variant suffix, use the full ID as the base region
            base_region = full_id
        
        seq = str(record.seq).upper()
        
        if len(seq) != seq_length:
            print(f"⚠️ Skipping {full_id} due to invalid length ({len(seq)} ≠ {seq_length})")
            continue
        
        seq_1hot = dna_io.dna_1hot(seq)
        mutated_seqs.setdefault(base_region, []).append(seq_1hot)
    
    # Filter for regions with expected number of mutated sequences
    original_count = len(mutated_seqs)
    mutated_seqs = {k: v for k, v in mutated_seqs.items() if len(v) == args.expected_variants}
    filtered_count = len(mutated_seqs)
    
    if filtered_count < original_count:
        print(f"⚠️ Filtered out {original_count - filtered_count} regions with != {args.expected_variants} variants")
    
    print(f"✅ {filtered_count} regions with {args.expected_variants} mutated variants loaded.")
    
    # Predict and Average
    region_names = []
    predictions_avg = []
    
    print("Predicting mutated sequences...")
    for region, onehot_seqs in tqdm(mutated_seqs.items(), desc="Predicting"):
        seq_batch = np.stack(onehot_seqs)  # shape: [N_variants, 1048576, 4]
        pred_batch = seqnn_model.model.predict(seq_batch, verbose=0, batch_size=args.batch_size)  # shape: [N_variants, N_pixels, N_targets]
        
        pred_avg = np.mean(pred_batch, axis=0)  # [N_pixels, N_targets]
        
        region_names.append(region)
        predictions_avg.append(pred_avg)
    
    predictions_avg = np.stack(predictions_avg)  # [N_regions, N_pixels, N_targets]
    
    # Save
    npz_file = os.path.join(args.output_dir, f"{args.file_prefix}_predictions_avg.npz")
    np.savez_compressed(npz_file,
                        region_names=np.array(region_names),
                        predictions_avg=predictions_avg)
    
    # Save metadata
    json_file = os.path.join(args.output_dir, f"{args.file_prefix}_metadata.json")
    with open(json_file, "w") as f:
        json.dump({
            "model_dir": args.model_dir,
            "data_dir": args.data_dir,
            "fasta_file": args.fasta_file,
            "num_regions": len(region_names),
            "expected_variants": args.expected_variants,
            "prediction_shape": predictions_avg.shape,
            "seq_length": seq_length,
            "target_length": target_length,
            "target_crop": target_crop
        }, f, indent=4)
    
    print(f"✅ Saved averaged predictions to {npz_file}")
    print(f"✅ Saved metadata to {json_file}")
    print(f"   Shape: {predictions_avg.shape}")


if __name__ == "__main__":
    main()

