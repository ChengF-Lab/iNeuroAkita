# Dissecting Alzheimer’s Disease Genetic Variants with Fine-Tuned 3D DNA Foundation Models

A comprehensive pipeline for training, predicting, and analyzing 3D genome structures by fine-tuning Akita model and evaluating the impact of sequence mutations on chromatin architecture.

image.png

## Overview

This repository provides tools for:
- **Training Akita models** to predict Hi-C contact maps from DNA sequences
- **Making predictions** on genomic sequences
- **Evaluating predictions** with various metrics
- **Analyzing CTCF motif mutations** and their effects on 3D genome structure
- **Analyzing SNP mutations** and their impact on chromatin architecture
- **Analyzing structural variants (SVs)** using SuPreMo integration
- **Computing disruption scores** to quantify the impact of sequence changes

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Evaluation](#evaluation)
- [CTCF Mutagenesis Analysis](#ctcf-mutagenesis-analysis)
- [SNP Mutagenesis Analysis](#snp-mutagenesis-analysis)
- [SV Mutagenesis Analysis](#sv-mutagenesis-analysis)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8
- Mamba or Conda
- CUDA-capable GPU (recommended for training and prediction)

### Environment Setup

```bash
# Option A: Create environment from YAML (recommended)
mamba env create -f environment.yml -n basenji

# Option B: If solver issues occur, use exact pinned packages
mamba create -n basenji --file environment.lock.txt

# Option C: Install pip extras if needed
mamba run -n basenji pip install -r requirements-pip.txt

# Activate the environment
conda activate basenji
```

## Quick Start

1. **Prepare your data** (see [Data Preparation](#data-preparation))
2. **Train or load a model** (see [Model Training](#model-training))
3. **Make predictions** (see [Prediction](#prediction))
4. **Evaluate results** (see [Evaluation](#evaluation))

## Data Preparation

### 1. Process Hi-C Data

Process Hi-C fastq data into `.cool` format using the [distiller-nf pipeline](https://github.com/open2c/distiller-nf).

### 2. Prepare Target Cooler File

Create a `target_cool.txt` file with the following format:

```
index	identifier	file	clip	sum_stat	description
0	NPC	./coolers/all_libraries.hg38.mapq_30.2048.cool	2	sum	NPC
```

### 3. Generate Input Data for Akita

Use `akita_data.py` to prepare training data:

```bash
akita_data.py \
  -g /path/to/ref_data/gaps_binsize2048_numconseq10.bed \
  -l 1048576 \
  --crop 65536 \
  --local \
  -o model_input/NPC_data \
  -b /path/to/ref_data/blacklist.bed \
  --as_obsexp \
  -p 8 \
  -k 1 \
  -t 0.1 \
  -v 0.1 \
  -w 2048 \
  --snap 2048 \
  --stride_train 262144 \
  --stride_test 524288 \
  --seed 44 \
  /path/to/ref_genome/hg38.fa \
model_input/target_cools.txt
```

**Parameters:**
- `-g`: Gaps BED file
- `-l`: Sequence length (default: 1048576)
- `--crop`: Crop size in bp
- `-o`: Output directory
- `-b`: Blacklist BED file
- `-p`: Number of parallel processes
- `-k`: Number of folds
- `-t`: Training fraction
- `-v`: Validation fraction
- `--stride_train`: Training stride
- `--stride_test`: Test stride

## Model Training

Train an Akita model using your prepared data:

```bash
akita_train.py \
  -k \
  -o model_input/trainOut_NPC \
               my_params.json \
               NPC_data
```

**Parameters:**
- `-k`: Keep existing model checkpoints
- `-o`: Output directory for trained model
- `my_params.json`: Model configuration file
- `NPC_data`: Input data directory

The trained model will be saved with `params.json` and `model_best.h5` files.

## Prediction

### Using Pre-trained Models

You can download pre-trained Akita models or use your own trained models.

### Basic Prediction Pipeline

The `run_akita_predict.py` script provides a complete prediction pipeline:

```bash
python run_akita_predict.py \
--model_dir /path/to/model \
--data_dirs /path/to/dataOut1 /path/to/dataOut2 \
--output_dir /path/to/predict_out \
--file_prefix akita_NPC \
--selected_targets all \
--batch_size 2 \
--make_smoothed_metrics \
--sigma 2.0 \
--add_region
```

**Parameters:**
- `--model_dir`: Directory containing `params.json` and `model_best.h5`
- `--data_dirs`: One or more test data directories (each should contain `sequences.bed`, `targets.txt`, and test TFRecords)
- `--output_dir`: Output directory for predictions
- `--file_prefix`: Base name for output files
- `--selected_targets`: Target indices (comma-separated, e.g., "0,1") or "all"
- `--batch_size`: Batch size for prediction
- `--make_smoothed_metrics`: Generate smoothed metrics CSV
- `--sigma`: Gaussian smoothing sigma
- `--add_region`: Include region column from sequences.bed

**Outputs:**
- `{file_prefix}_predictions_targets.npz`: Predictions and targets
- `{file_prefix}_metadata.json`: Metadata about the prediction run
- `{file_prefix}_smoothed_metrics.csv`: (Optional) Per-sequence smoothed metrics

### Prediction from FASTA Sequences

For custom FASTA sequences:

```bash
python predict_from_fasta_batch.py \
  --model_dir /path/to/model \
  --data_dir /path/to/data \
  --fasta_file /path/to/sequences.fa \
  --output_dir /path/to/output
```

## Evaluation

### Comprehensive Evaluation

Use `run_akita_eval.py` for detailed evaluation:

```bash
python run_akita_eval.py \
  --npz_path /path/to/predictions_targets.npz \
  --bed_file1 /path/to/sequences1.bed \
  --bed_file2 /path/to/sequences2.bed \
  --targets_file /path/to/targets.txt \
  --output_dir /path/to/eval_output \
  --file_prefix akita_NPC_eval \
  --sigma 2.0 \
  --top_n 5 \
  --vmin -2.0 \
  --vmax 2.0
```

**Parameters:**
- `--bed_file2`: Optional, only needed if data was concatenated from multiple sources
- `--smoothed_metrics_csv`: Optional, path to smoothed metrics CSV (if not provided, will be inferred)
- `--specific_region`: Optional, specific region to plot (e.g., "chr1:100000-200000" or integer index)
- `--target_index`: Target index for specific region plot (default: 0)

**Outputs:**
- Global metrics CSV (MSE and Pearson correlation)
- Top-N contact maps (PDF)
- Specific region maps (PDF)
- Smoothed evaluation plots (PDF)

### Evaluation Utilities

The `akita_eval_utils.py` module provides reusable functions for:
- Loading NPZ files and metadata
- Reconstructing and smoothing contact matrices
- Computing global pooled MSE & Pearson correlation
- Building per-sequence Pearson rankings
- Generating various visualization plots

## CTCF Mutagenesis Analysis

### 1. Generate Mutated Sequences

Generate randomized sequences with CTCF motifs replaced:

```bash
python generate_mutated_seq.py \
  --fasta_file CTCF_seq/test_sequences.fa \
  --motif_bed IntersectOut/intersect_CTCF.bed \
  --output_file CTCF_seq/CTCF_sequences_mutated.fa \
  --n_random 10
```

**Parameters:**
- `--fasta_file`: Input FASTA file with reference sequences
- `--motif_bed`: BED file with CTCF motif coordinates
- `--output_file`: Output FASTA file for mutated sequences
- `--n_random`: Number of random mutations per sequence

### 2. Predict on Mutated Sequences

```bash
# Predict on reference sequences
python predict_from_fasta_batch.py \
  --model_dir /path/to/model \
  --data_dir /path/to/data \
  --fasta_file CTCF_seq/test_sequences.fa \
  --output_dir predictOut_ref \
  --file_prefix akita_ref

# Predict on mutated sequences (averages across variants)
python predict_from_mutated_sequences_batch.py \
  --model_dir /path/to/model \
  --data_dir /path/to/data \
  --fasta_file CTCF_seq/CTCF_sequences_mutated.fa \
  --output_dir predictOut_mut \
  --file_prefix mutated \
  --expected_variants 10
```

**Note:** `predict_from_mutated_sequences_batch.py` expects mutated sequences with IDs like `region_rand0`, `region_rand1`, etc., and will average predictions across all variants for each base region.

### 3. Calculate Disruption Scores

Compute disruption scores comparing reference and mutated predictions:

```bash
python calculate_ctcf_disruption_scores.py \
  --ref_npz /path/to/ref_predictions.npz \
  --mut_npz /path/to/mut_predictions.npz \
  --targets_file /path/to/targets.txt \
  --output_dir /path/to/output \
  --file_prefix mutated
```

**Outputs:**
- `{file_prefix}_CTCF_disruption_scores_sorted.tsv`: Disruption scores sorted by magnitude

### 4. Visualize Results

Generate visualization plots:

```bash
python run_plot_chip_map.py \
  --disruption_scores /path/to/disruption_scores.tsv \
  --output_dir /path/to/plots \
  --file_prefix CTCF_mut
```

## SNP Mutagenesis Analysis

This section describes how to analyze the impact of single nucleotide polymorphisms (SNPs) on 3D genome structure.

### 1. Build REF/ALT Sequences

Generate 1 Mb reference and alternate sequences centered at each SNP:

```bash
python build_ref_alt_sequences_1mb.py \
  -i varInput/raw_frVar_with_hg38.txt \
  -f /path/to/reference_genome/hg38.fa \
  --out-ref Seq_out/frVar_REF.fa \
  --out-alt Seq_out/frVar_ALT.fa \
  --mismatch-tsv Seq_out/frVar_ref_mismatches.tsv
```

**Input Format:**
The input TSV file should contain columns for:
- Chromosome: `Chr`, `chr`, `CHROM`, `Chrom`, `chromosome`, or `Chromosome`
- Position: `Pos_hg38`, `Pos`, `POS`, `position`, `Position`, `Start`, or `snp_start`
- Reference allele: `REF`, `Ref`, or `ref`
- Alternate allele: `ALT`, `Alt`, `alt`, or `ALT_selected`
- Optional: `snpID`, `ID`, `rsID`, or `rsid` for variant identifiers
- Optional: `Gene`, `Pair_Symbol`, `gene`, or `Symbol` for gene annotations

**Parameters:**
- `-i`, `--input`: Input TSV file with SNP coordinates and alleles
- `-f`, `--fasta`: Reference genome FASTA file (must match build, e.g., hg38)
- `--out-ref`: Output FASTA file for reference sequences
- `--out-alt`: Output FASTA file for alternate sequences (one per ALT allele)
- `--mismatch-tsv`: Output TSV file logging sites where REF doesn't match FASTA

**Outputs:**
- REF FASTA: 1 Mb sequences centered at each SNP with reference allele
- ALT FASTA: 1 Mb sequences with alternate allele(s) substituted
- Mismatch log: Sites where REF allele doesn't match the reference genome

### 2. Predict and Score Disruption

Predict Hi-C contact maps for REF and ALT sequences and compute disruption scores:

```bash
python predict_ref_alt_and_score.py \
  --model_dir /path/to/model \
  --data_dir /path/to/data \
  --ref_fa Seq_out/frVar_REF.fa \
  --alt_fa Seq_out/frVar_ALT.fa \
  --out_scores Disrup_out/All_frVar_disruption_scores.tsv \
  --plots_dir Disrup_out/All_frVar_plots_ref_alt_diff \
  --plot_format both \
  --genes_bed GeneInput/myGene.bed \
  --diff_signal_threshold 0.5 \
  --batch 2 \
  --reduce mean
```

**Parameters:**
- `--model_dir`: Directory containing `params.json` and `model_best.h5`
- `--data_dir`: (Optional) Data directory containing `statistics.json`. If not provided, will search in model_dir
- `--ref_fa`: Input FASTA file with reference sequences
- `--alt_fa`: Input FASTA file with alternate sequences
- `--out_scores`: Output TSV file with disruption scores
- `--plots_dir`: Directory to save visualization plots (use `''` to skip plots)
- `--plot_format`: Plot format - `square`, `triangle`, or `both` (default: `both`)
- `--genes_bed`: (Optional) BED file with gene annotations for visualization
- `--diff_signal_threshold`: Signal strength weighting for diff maps (0-1, default: 0.3)
- `--batch`: Batch size for prediction (default: 2)
- `--reduce`: Reduction method over targets - `mean` or target index (default: `mean`)
- `--vlim`: Color scale limit for plots (default: 2.0)
- `--promoter_kb`: Promoter window size in kb for gene annotations (default: 2.0)

**Outputs:**
- `{out_scores}`: TSV file with disruption metrics (L2, MSE, L1, MaxAbs) for each REF/ALT pair
- `{plots_dir}/*.pdf`: Visualization plots showing REF, ALT, and DIFF contact maps
  - Square format: Full contact matrices
  - Triangle format: Upper triangular view
  - Gene tracks: Annotated gene positions when `--genes_bed` is provided

**Disruption Metrics:**
- **L2 (Frobenius norm)**: Overall magnitude of difference
- **MSE (Mean Squared Error)**: Average squared difference
- **L1 (Mean Absolute Error)**: Average absolute difference
- **MaxAbs**: Maximum absolute difference

## SV Mutagenesis Analysis

This section describes how to analyze the impact of structural variants (SVs) on 3D genome structure using the SuPreMo framework.

### 1. Run SuPreMo Analysis

SuPreMo (Structural Variant Prediction Model) analyzes structural variants and computes Akita-based disruption scores:
- https://github.com/ketringjoni/SuPreMo

```bash
python /path/to/SuPreMo/scripts/SuPreMo.py \
  /path/to/variants/SV_variants.txt \
  --fa /path/to/reference_genome/hg38.fa \
  --shift_by -1 0 1 \
  --revcomp add_revcomp \
  --file D_weighted \
  --dir /path/to/output_dir \
  --get_Akita_scores \
  --get_maps \
  --get_tracks \
  --get_seq \
  --roi genes \
  --roi_scales 10
```

**Parameters:**
- Input file: TSV file with structural variant coordinates
- `--fa`: Reference genome FASTA file
- `--shift_by`: Position shifts to test (e.g., `-1 0 1`)
- `--revcomp`: Whether to add reverse complement sequences (`add_revcomp` or `no_revcomp`)
- `--file`: File prefix for outputs
- `--dir`: Output directory for results
- `--get_Akita_scores`: Compute Akita disruption scores
- `--get_maps`: Generate contact map visualizations
- `--get_tracks`: Generate track visualizations
- `--get_seq`: Extract sequences
- `--roi`: Region of interest type (e.g., `genes`)
- `--roi_scales`: Scaling factor for ROI weighting

**Outputs:**
- `{file}_scores`: TSV file with disruption scores
- `{file}_maps.npy`: Contact maps (NumPy format)
- `{file}_tracks.npy`: Disruption tracks (NumPy format)
- Additional files depending on options

### 2. Visualize Results

Generate visualization plots from SuPreMo outputs:

```bash
python plot_structural_variant_analysis.py
```

**Note:** This script requires customization based on your specific SuPreMo output structure. Update the paths and parameters in the script to match your data:

- Input file paths
- Output directory paths
- Analysis parameters (shift, revcomp suffix, scoring method, etc.)
- ROI configuration

**Typical Workflow:**
1. Configure paths in `plot_structural_variant_analysis.py`
2. Run the script to generate PDF visualizations
3. Review disruption scores and contact maps for each variant

**SuPreMo Integration:**
This pipeline integrates with the [SuPreMo framework](https://github.com/your-repo/SuPreMo) for structural variant analysis. Ensure SuPreMo is properly installed and configured before running these analyses.

## Project Structure

```
.
├── README.md                          # This file
├── environment.yml                    # Conda environment specification
├── environment.lock.txt              # Pinned package versions
├── requirements-pip.txt              # Additional pip packages
│
├── 01_model_prediction/              # Model training and prediction scripts
│   ├── run_akita_predict.py          # Main prediction pipeline
│   ├── run_akita_eval.py             # Evaluation script
│   ├── run_akita_eval_v2.py         # Enhanced evaluation script
│   ├── akita_eval_utils.py           # Evaluation utilities
│   └── Specific_region_Map.py        # Region-specific visualization
│
├── 02_CTCF_mut/                      # CTCF mutagenesis analysis
│   ├── generate_mutated_seq.py       # Generate mutated sequences
│   ├── predict_from_fasta_batch.py   # Predict from reference sequences
│   ├── predict_from_mutated_sequences_batch.py  # Predict from mutated sequences
│   ├── calculate_ctcf_disruption_scores.py   # Calculate disruption scores
│   └── run_plot_chip_map.py          # Visualization
│
├── 03_SNP_mut/                       # SNP mutation analysis
│   ├── build_ref_alt_sequences_1mb.py        # Build REF/ALT sequences from SNPs
│   └── predict_ref_alt_and_score.py  # Predict and score disruption
│
└── 04_SV_mut/                        # Structural variant analysis
    └── plot_structural_variant_analysis.py        # Visualization for SV analysis
```

## Requirements

### Software Dependencies

- Python 3.8
- TensorFlow 2.13.1
- Basenji (for Akita model)
- NumPy, Pandas, Matplotlib, Seaborn
- BioPython
- Cooltools
- PyBedTools

See `environment.yml` for the complete list of dependencies.

### Data Requirements

- Reference genome FASTA file
- Hi-C contact maps in `.cool` format
- BED files for gaps, blacklists, and regions of interest
- Model parameters file (`params.json`) for training



## Acknowledgments

- [Akita model](https://github.com/calico/basenji/tree/master/manuscripts/akita) by Fudenberg G. et al.
  - 'Fudenberg G, Kelley DR, Pollard KS. Predicting 3D genome folding from DNA sequence with Akita. Nat Methods. 2020 Nov;17(11):1111-1117. doi: 10.1038/s41592-020-0958-x. Epub 2020 Oct 12. PMID: 33046897; PMCID: PMC8211359.'
- [Basenji](https://github.com/calico/basenji) framework
- [Distiller-nf](https://github.com/open2c/distiller-nf) for Hi-C processing
- [SuPreMo framework](https://github.com/your-repo/SuPreMo) for SVs mutagenesis analysis 


