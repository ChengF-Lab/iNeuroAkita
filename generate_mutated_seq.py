#!/usr/bin/env python3
"""
Generate mutated sequences by replacing CTCF motifs with randomized sequences.

This script reads reference sequences from a FASTA file and a BED file containing
CTCF motif coordinates, then generates mutated sequences where motifs are replaced
with random sequences that preserve case.
"""

from Bio import SeqIO
from collections import defaultdict
import random
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mutated sequences with CTCF motifs replaced"
    )
    parser.add_argument(
        "--fasta_file",
        required=True,
        help="Input FASTA file with reference sequences"
    )
    parser.add_argument(
        "--motif_bed",
        required=True,
        help="BED file with CTCF motif coordinates (9 columns: motif_chr, motif_start, motif_end, motif_id, qval, strand, test_chr, test_start, test_end)"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output FASTA file for mutated sequences"
    )
    parser.add_argument(
        "--n_random",
        type=int,
        default=10,
        help="Number of random mutations per sequence (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Validate input files
    if not os.path.exists(args.fasta_file):
        raise FileNotFoundError(f"FASTA file not found: {args.fasta_file}")
    if not os.path.exists(args.motif_bed):
        raise FileNotFoundError(f"Motif BED file not found: {args.motif_bed}")
    
    # Load reference sequences with original case
    print(f"Loading sequences from {args.fasta_file}...")
    seq_dict = {rec.id: str(rec.seq) for rec in SeqIO.parse(args.fasta_file, "fasta")}
    print(f"✅ Loaded {len(seq_dict)} sequences")
    
    # Parse motif BED and adjust coordinates relative to test window
    motif_map = defaultdict(set)
    
    print(f"Loading motifs from {args.motif_bed}...")
    with open(args.motif_bed) as f:
    for line in f:
        motif_chr, motif_start, motif_end, motif_id, qval, strand, test_chr, test_start, test_end = line.strip().split()
        test_id = f"{test_chr}:{test_start}-{test_end}"
        motif_start, motif_end = int(motif_start), int(motif_end)
        test_start, test_end = int(test_start), int(test_end)
        rel_start = motif_start - test_start
        rel_end = motif_end - test_start
        if rel_start < 0 or rel_end > (test_end - test_start):
            print(f"⚠️ Motif out of bounds for {test_id}: {rel_start}-{rel_end}")
            continue
        motif_map[test_id].add((rel_start, rel_end, strand))

# Helper: mutate each base to a *different* one, preserving case
def mutate_preserve_case(seq_slice):
    mutated = []
    for base in seq_slice:
        base_upper = base.upper()
        if base_upper not in "ACGT":
            mutated.append(base)  # Non-DNA base, keep unchanged
            continue
        options = [b for b in "ACGT" if b != base_upper]
        new_base = random.choice(options)
        mutated.append(new_base if base.isupper() else new_base.lower())
    return ''.join(mutated)

# Generate mutated sequences
print(f"Generating {args.n_random} mutated sequences per region...")
rand_seqs = []

for seq_name, seq in seq_dict.items():
    motif_coords = motif_map.get(seq_name, [])
    for i in range(args.n_random):
        seq_list = list(seq)  # make mutable
        for start, end, strand in motif_coords:
            original = seq[start:end]
            rand_sub = mutate_preserve_case(original)
            seq_list[start:end] = rand_sub
        mutated_seq = ''.join(seq_list)
        assert len(mutated_seq) == len(seq), f"❌ Length mismatch in {seq_name}_rand{i}"
        rand_seqs.append((f"{seq_name}_rand{i}", mutated_seq))

# Create output directory if needed
os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)

# Write to FASTA
print(f"Writing {len(rand_seqs)} mutated sequences to {args.output_file}...")
with open(args.output_file, "w") as f:
    for name, seq in rand_seqs:
        f.write(f">{name}\n{seq}\n")

print(f"✅ Wrote {len(rand_seqs)} mutated sequences to {args.output_file}")

# Check lengths (optional validation)
expected_length = len(list(seq_dict.values())[0]) if seq_dict else None
if expected_length:
    bad_count = 0
    for record in SeqIO.parse(args.output_file, "fasta"):
        if len(record.seq) != expected_length:
            print(f"⚠️ {record.id} has length {len(record.seq)} ≠ {expected_length}")
            bad_count += 1
    if bad_count > 0:
        print(f"⚠️ Total FASTA records with unexpected length: {bad_count}")



total_motifs = 0
total_regions = len(motif_map)

for seq_name, coords in motif_map.items():
    print(f"🧬 {seq_name}: {len(coords)} motifs")
    for start, end, strand in coords:
        print(f"  Motif {start}-{end} ({end - start} bp)")
    total_motifs += len(coords)

print(f"\n🔢 Total regions: {total_regions}")
print(f"🧬 Total motifs mutated: {total_motifs}")
print("\n✅ Done.")


if __name__ == "__main__":
    main()

