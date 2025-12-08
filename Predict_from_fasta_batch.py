#!/usr/bin/env python3
"""
Full Prediction Pipeline - All Regions

Predict Hi-C contact maps from FASTA sequences using a trained Akita model.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
from basenji import dataset, dna_io, seqnn
from tqdm import tqdm
from scipy.stats import pearsonr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict Hi-C contact maps from FASTA sequences"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing params.json and model_best.h5"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Data directory containing sequences.bed and statistics.json"
    )
    parser.add_argument(
        "--fasta_file",
        required=True,
        help="Input FASTA file with sequences to predict"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--file_prefix",
        default="akita_fasta",
        help="Prefix for output files (default: 'akita_fasta')"
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
    target_length1 = seq_length // stats['pool_width']
    target_length1_cropped = target_length1 - 2 * target_crop
    hic_diags = stats['diagonal_offset']
    
    # Load test regions and targets
    print("Loading test regions and targets...")
    sequences = pd.read_csv(os.path.join(args.data_dir, 'sequences.bed'), sep='\t', names=['chr', 'start', 'stop', 'type'])
    sequences_test = sequences[sequences['type'] == 'test'].reset_index(drop=True)
    sequences_test['region_name'] = sequences_test['chr'] + ':' + sequences_test['start'].astype(str) + '-' + sequences_test['stop'].astype(str)
    
    # Load target .tfr
    test_data = dataset.SeqDataset(args.data_dir, 'test', batch_size=args.batch_size)
    _, test_targets = test_data.numpy(return_inputs=True, return_outputs=True)
    print(f"✅ Loaded test targets: {test_targets.shape}")
    
    # Load sequences from FASTA
    sequence_dict = {}
    print(f"Loading sequences from {args.fasta_file}...")
    for record in SeqIO.parse(args.fasta_file, "fasta"):
        region = record.id
        seq = str(record.seq).upper()
        if len(seq) != seq_length:
            raise ValueError(f"{region} has length {len(seq)} ≠ {seq_length}")
        sequence_dict[region] = dna_io.dna_1hot(seq)
    
    sequence_names = list(sequence_dict.keys())
    print(f"✅ Loaded {len(sequence_names)} sequences.")
    
    # Predict for each region
    pred_list = []
    target_list = []
    valid_region_names = []
    
    print("Predicting contact maps...")
    for region_name in tqdm(sequence_names, desc="Predicting"):
        if region_name not in sequences_test['region_name'].values:
            print(f"⚠️ Skipping region not in test set: {region_name}")
            continue
        
        region_idx = sequences_test.index[sequences_test['region_name'] == region_name].tolist()[0]
        input_seq = np.expand_dims(sequence_dict[region_name], axis=0)  # [1, 1048576, 4]
        prediction = seqnn_model.model.predict(input_seq, verbose=0)[0]  # [99681, N_targets]
        
        pred_list.append(prediction)
        target_list.append(test_targets[region_idx])  # [99681, N_targets]
        valid_region_names.append(region_name)
    
    # Save predictions and targets
    pred_array = np.stack(pred_list)
    target_array = np.stack(target_list)
    
    # Save predictions and targets
    npz_file = os.path.join(args.output_dir, f"{args.file_prefix}_predictions_targets.npz")
    np.savez_compressed(npz_file,
                        region_names=np.array(valid_region_names),
                        predictions=pred_array,
                        targets=target_array)
    
    # Save metadata
    json_file = os.path.join(args.output_dir, f"{args.file_prefix}_metadata.json")
    with open(json_file, "w") as f:
        json.dump({
            "model_dir": args.model_dir,
            "data_dir": args.data_dir,
            "fasta_file": args.fasta_file,
            "num_regions": len(valid_region_names),
            "prediction_shape": pred_array.shape,
            "target_shape": target_array.shape,
            "seq_length": seq_length,
            "target_length": target_length,
            "target_crop": target_crop,
            "diagonal_offset": hic_diags
        }, f, indent=4)
    
    print(f"✅ Saved predictions to {npz_file}")
    print(f"✅ Saved metadata to {json_file}")
    
    # Evaluation
    print("\nEvaluating predictions...")
    # Shape: [N_regions, N_pixels, N_targets]
    mse_global = np.mean((pred_array - target_array) ** 2)
    mse_per_target = np.mean((pred_array - target_array) ** 2, axis=(0, 1))
    print(f"📉 Global MSE: {mse_global:.6f}")
    print(f"📉 MSE per target: {mse_per_target}")
    
    for t in range(pred_array.shape[-1]):
        pred_t = pred_array[..., t]
        targ_t = target_array[..., t]
        mask = ~np.isnan(pred_t) & ~np.isnan(targ_t)
        
        if mask.sum() > 0:
            corr, _ = pearsonr(pred_t[mask].flatten(), targ_t[mask].flatten())
            print(f"📈 Pearson Correlation (target {t}): {corr:.4f}")
        else:
            print(f"⚠️ No valid data for target {t}")


if __name__ == "__main__":
    main()

