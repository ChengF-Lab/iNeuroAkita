# Required Scripts for README.md

This document lists all Python scripts mentioned in the README.md that are required for the pipeline.

## Core Scripts (Root Directory)

### Model Prediction & Evaluation
- **`run_akita_predict.py`** - Main prediction pipeline for Akita models
- **`run_akita_eval.py`** - Comprehensive evaluation script
- **`akita_eval_utils.py`** - Evaluation utilities module (used by run_akita_eval.py)
- **`Predict_from_Seq2_batch.py`** - Predict from FASTA sequences

### CTCF Mutagenesis Analysis
- **`generate_mutated_seq.py`** - Generate mutated sequences with CTCF motifs replaced
- **`Predict_from_mutatedSeq_batch.py`** - Predict from mutated sequences (averages across variants)
- **`CTCF_mutated_disruptionScore.py`** - Calculate disruption scores comparing reference and mutated predictions
- **`run_plot_chip_map.py`** - Visualization for CTCF mutagenesis results

### SNP Mutagenesis Analysis
- **`01build_ref_alt_1Mb.py`** - Build REF/ALT sequences from SNPs (1 Mb windows)
- **`02predict_ref_alt_and_score_improved.py`** - Predict and score disruption for REF/ALT SNP pairs

### SV Mutagenesis Analysis
- **`plot_SIM_Lee_update.py`** - Visualization for SV analysis results

## External Dependencies (Not in this repository)

These scripts are mentioned in the README but are part of external packages:

- **`akita_data.py`** - Part of Basenji/Akita package (data preparation)
- **`akita_train.py`** - Part of Basenji/Akita package (model training)
- **`SuPreMo.py`** - Part of SuPreMo framework (SV analysis)

## Script Organization by Directory

### Root Directory Scripts
```
.
├── run_akita_predict.py
├── run_akita_eval.py
├── akita_eval_utils.py
├── Predict_from_Seq2_batch.py
├── generate_mutated_seq.py
├── Predict_from_mutatedSeq_batch.py
├── CTCF_mutated_disruptionScore.py
├── run_plot_chip_map.py
├── 01build_ref_alt_1Mb.py
├── 02predict_ref_alt_and_score_improved.py
└── plot_SIM_Lee_update.py
```

### Directory Structure (as documented in README)

```
├── 01_model_prediction/
│   ├── run_akita_predict.py
│   ├── run_akita_eval.py
│   ├── run_akita_eval_v2.py
│   ├── akita_eval_utils.py
│   └── Specific_region_Map.py
│
├── 02_CTCF_mut/
│   ├── generate_mutated_seq.py
│   ├── Predict_from_Seq2_batch.py
│   ├── Predict_from_mutatedSeq_batch.py
│   ├── CTCF_mutated_disruptionScore.py
│   └── run_plot_chip_map.py
│
├── 03_SNP_mut/
│   ├── 01build_ref_alt_1Mb.py
│   └── 02predict_ref_alt_and_score_improved.py
│
└── 04_SV_mut/
    └── plot_SIM_Lee_update.py
```

## Summary

**Total Required Scripts in Repository: 11**

1. `run_akita_predict.py`
2. `run_akita_eval.py`
3. `akita_eval_utils.py`
4. `Predict_from_Seq2_batch.py`
5. `generate_mutated_seq.py`
6. `Predict_from_mutatedSeq_batch.py`
7. `CTCF_mutated_disruptionScore.py`
8. `run_plot_chip_map.py`
9. `01build_ref_alt_1Mb.py`
10. `02predict_ref_alt_and_score_improved.py`
11. `plot_SIM_Lee_update.py`

**Note:** Some scripts appear in both the root directory and subdirectories (e.g., `02_CTCF_mut/`). The README references scripts that should be accessible from the root directory or their respective subdirectories.

