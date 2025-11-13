# DMS Analysis Pipeline with ESM2

This pipeline computes ESM2 log-likelihoods for a deep mutational scanning (DMS) library.

## Overview

This is a simplified version of the ESM2_Validation_Pipeline that:
- Takes a DMS library FASTA as input (all single mutations)
- Computes log-likelihoods using base ESM2 models
- Computes log-likelihoods using fine-tuned ESM2 models
- Removes all Nextstrain tree processing steps

## Directory Structure

```
DMS_Analysis/
├── Snakefile                          # Main pipeline definition
├── params.csv                          # Model parameters to test
├── dms_library_all_mutations.fasta    # Input DMS library (10,755 sequences)
├── estimated_sequence.fasta            # Wild-type sequence
├── summary_avgprefs.csv               # DMS experimental data
├── scripts/
│   ├── calc_ll_esm.py                 # ESM log-likelihood computation
│   ├── fasta_to_csv.py                # FASTA to CSV converter
│   ├── filter_by_date.py              # Date filtering script
│   └── make_esm_fine_tune_models.py   # Fine-tuning script
├── input/
│   └── Fine_Tuning_Datasets/
│       ├── H3N2_Tree/
│       │   └── fine_tune_fasta_ha.fasta  # HA sequences only (485)
│       └── H3N2_HA_2009_Cutoff.fasta     # Generated: filtered by date
└── results/
    ├── dms_library.csv                # Converted input
    ├── log_likelihoods/               # Output log-likelihoods
    │   └── {model_params}/
    │       ├── base/                  # Base model results
    │       └── H3N2_HA_2009_Cutoff/   # Fine-tuned results
    └── fine_tune_models/              # Fine-tuned models (or symlinks)
```

## Input

**DMS Library:** `dms_library_all_mutations.fasta`
- 10,755 sequences (1 wild-type + 10,754 single mutants)
- Generated from estimated_sequence.fasta
- Every possible single amino acid mutation (19 per position × 566 positions)

## Configuration

**params.csv** - Define models and training parameters:
```csv
epochs,learning_rate,model
1,5e-05,esm2_t33_650M_UR50D
1,5e-05,esm2_t36_3B_UR50D
```

**Snakefile variables:**
- `DATASET` - Dataset name: `H3N2_HA_2009_Cutoff`
- `CUTOFF_DATE` - Date cutoff: `2009.2643835616439` (approximately April 2009)
- `SEGMENT` - Segment used: `ha` (HA only)
- Fine-tuning uses only H3N2 HA sequences with dates ≤ cutoff

Models available:
- `esm2_t33_650M_UR50D` - 650M parameters (fastest)
- `esm2_t36_3B_UR50D` - 3B parameters (medium)
- `esm2_t48_15B_UR50D` - 15B parameters (slowest, highest memory)

## Usage

### Dry Run
```bash
snakemake --dry-run
```

### Execute Pipeline
```bash
# Run all jobs
snakemake --cores 8

# Run specific target
snakemake results/log_likelihoods/epochs~1/learning_rate~5e-05/model~esm2_t33_650M_UR50D/base/dms_library_LL.csv

# With GPU
snakemake --cores 8 --resources gpu=1
```

### Visualize DAG
```bash
snakemake --dag | dot -Tpng > dag.png
```

## Pipeline Steps

1. **Convert FASTA to CSV** (`fasta_to_csv`)
   - Input: dms_library_all_mutations.fasta
   - Output: results/dms_library.csv

2. **Base Model Log-Likelihoods** (`calc_ll_esm_base`)
   - For each model in params.csv
   - Output: results/log_likelihoods/{params}/base/dms_library_LL.csv

3. **Filter HA Sequences by Date** (`filter_ha_by_date`)
   - Filters HA segment only by date cutoff
   - Uses Nextstrain tree JSON to determine sequence dates
   - Keeps only sequences with num_date ≤ 2009.2643835616439
   - Output: input/Fine_Tuning_Datasets/H3N2_HA_2009_Cutoff.fasta

4. **Link or Train Fine-Tuned Models** (`link_or_generate_fine_tune_model`)
   - First tries to symlink existing models from ESM2_Validation_Pipeline
   - If not found, trains new model on HA date-filtered dataset
   - Output: results/fine_tune_models/{params}/H3N2_HA_2009_Cutoff/Fine_Tune_Model.bin

5. **Fine-Tuned Model Log-Likelihoods** (`calc_ll_esm_fine_tune`)
   - For each fine-tuned model
   - Output: results/log_likelihoods/{params}/H3N2_HA_2009_Cutoff/dms_library_LL_Fine_Tune.csv

## Output Format

Log-likelihood CSV files contain:
- `node` - Sequence ID (e.g., "Estimated_sequence_from_DMS_D1A")
- `sequence` - Amino acid sequence
- `log_likelihood` - Sum of log-probabilities for all tokens
- `runtime` - Computation time in seconds
- `segment` - Set to "DMS"
- `model_requested` - Model name
- `base_model_loaded` - Base model used
- `ft_kind` - Fine-tuning type ("none", "lora", or "state_dict")
- `adapters_path` - Path to LoRA adapters (if used)
- `repr_layer_used` - Representation layer

## Differences from ESM2_Validation_Pipeline

| Feature | ESM2_Validation | DMS_Analysis |
|---------|----------------|--------------|
| Input | Nextstrain JSON trees | Single FASTA file (DMS library) |
| Preprocessing | 3 tree-processing rules | Direct FASTA → CSV |
| Segmentation | 8 segments separately | Single unified analysis |
| Fine-tuning dataset | All 8 segments concatenated | HA only, filtered by date |
| Date filtering | Time wildcard from params | Fixed cutoff: 2009.264 |
| params.csv | next_tree, time, epochs, lr, model | epochs, lr, model |
| Output | Per-segment CSV files | Single CSV per model |
| Wildcards | ps, next_tree, segment, dataset | ps, dataset |

## Notes

- The pipeline automatically reuses existing fine-tuned models from `../../ESM2_Validation_Pipeline/results/fine_tune_models/` if available
- Fine-tuning is computationally expensive (especially for 3B and 15B models)
- Base model inference on 10K sequences takes ~30-60 minutes per model
- Ensure sufficient GPU memory (650M needs ~6GB, 3B needs ~16GB, 15B needs ~40GB)

## Generated Files

To regenerate the DMS library:
```bash
python generate_dms_library.py --input estimated_sequence.fasta --output dms_library_all_mutations.fasta
```
