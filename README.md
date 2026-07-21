# BandLLM

BandLLM predicts fixed near-Fermi electronic band structures from crystal CIF representations. The current public-ready version is a band-only pipeline: DOS arrays may exist in legacy data files, but DOS is not used as a training target, loss term, metric, or visualization target.

The repository is organized around a two-stage model:

1. Stage I trains an electronic band autoencoder that compresses a `128 x 16` band tensor into a compact latent representation and reconstructs the band structure.
2. Stage II uses a CIF Transformer/CrystaLLM-style encoder plus high-symmetry path information to predict the Stage-I latent, then decodes it back to the full band tensor.

The Python package name is currently kept as `crystallm` for compatibility with the original model code and internal imports. The project/repository name is `BandLLM`.

## Reproducibility Requirements

This GitHub repository contains the code needed to reproduce the training, prediction, and basic evaluation workflow. It is not a self-contained data/checkpoint archive.

To reproduce the reported pipeline, prepare the following files outside git:

```text
data/mp_all_bs3072/
pretrained/crystallm_v1_small/ckpt.pt
```

The expected dataset folder is the processed Materials Project band dataset used by the scripts. The pretrained checkpoint is the CrystaLLM-style CIF encoder checkpoint used by Stage II. Large datasets, model checkpoints, prediction arrays, and generated figures are intentionally excluded by `.gitignore`.

## Repository Contents

```text
BandLLM/
├── bin/
│   ├── download_mp_electronic.py
│   ├── prepare_electronic_dataset.py
│   ├── filter_electronic_manifest.py
│   ├── make_electronic_splits.py
│   ├── materialize_band16_targets.py
│   ├── compute_electronic_target_stats.py
│   ├── inspect_electronic_dataset.py
│   ├── train_electronic_autoencoder.py
│   ├── predict_electronic_autoencoder.py
│   ├── train_structure_to_electronic_latent.py
│   ├── predict_structure_to_electronic_latent.py
│   ├── analyze_electronic_predictions.py
│   ├── plot_electronic_prediction.py
│   ├── plot_electronic_sample_fitting.py
│   ├── plot_fitting_curves.py
│   ├── plot_error_analysis_suite.py
│   └── plot_ranked_electronic_predictions.py
├── crystallm/
│   ├── _tokenizer.py
│   ├── _model.py
│   ├── _electronic.py
│   ├── _electronic_model.py
│   └── spacegroups.txt
├── EXPERIMENT_L512_PHYS.md
├── requirements.txt
├── LICENSE
└── README.md
```

Large files are intentionally not included:

- `data/`
- `out/`
- `pretrained/`
- checkpoint files such as `*.pt`, `*.pth`
- prediction arrays such as `*.npz`
- generated figures and logs

## Environment

Create a Python environment with the dependencies in `requirements.txt`. The original experiments used a CUDA-enabled PyTorch environment.

```bash
pip install -r requirements.txt
```

Run commands from the repository root:

```bash
cd /path/to/BandLLM
```

All scripts add the repository root to `sys.path`, so editable installation is not required.

## Data Layout

The expected processed dataset layout is:

```text
data/mp_all_bs3072/
├── manifest.csv
├── manifest_full_split.csv
├── target_stats_band16_only.npz
├── cifs/
└── samples/
```

Each sample `.npz` should contain the CIF token inputs and the band-only target:

- `input_ids`
- `attention_mask`
- `band16`, shape `(128, 16)`
- `band16_mask`
- `target`, flattened band target of length `2048`
- `target_mask`
- `band_path`

Band energies are referenced to the Fermi level and use eV units.

## Data Preparation

Download Materials Project electronic data:

```bash
export MP_API_KEY=your_materials_project_api_key

python bin/download_mp_electronic.py --mp-all --label-mode full --out-dir data/mp_all_bs3072 --band-kpoints 128 --band-count 64 --band-axis distance --band-selection fermi --band-emin -10 --band-emax 10 --dos-points 1024 --dos-emin -10 --dos-emax 10 --block-size 2048 --resume --skip-errors
```

Materialize the final 16-band target:

```bash
python bin/materialize_band16_targets.py data/mp_all_bs3072/manifest.csv --overwrite --band-count 16
```

Create train/validation/test splits:

```bash
python bin/make_electronic_splits.py data/mp_all_bs3072/manifest.csv --out data/mp_all_bs3072/manifest_full_split.csv --train-fraction 0.90 --val-fraction 0.05 --test-fraction 0.05 --seed 1337 --group-by id
```

Compute target normalization statistics:

```bash
python bin/compute_electronic_target_stats.py data/mp_all_bs3072/manifest_full_split.csv --out data/mp_all_bs3072/target_stats_band16_only.npz
```

Inspect the dataset:

```bash
python bin/inspect_electronic_dataset.py data/mp_all_bs3072/manifest_full_split.csv
```

## Stage I: Band Autoencoder

Train the band autoencoder:

```bash
python bin/train_electronic_autoencoder.py data/mp_all_bs3072/manifest_full_split.csv --out-dir out/ae_band_l512_phys --target-stats data/mp_all_bs3072/target_stats_band16_only.npz --epochs 200 --batch-size 32 --latent-dim 512 --band-weight 1.0 --band-slope-weight 0.20 --band-curvature-weight 0.05 --band-fermi-window 1.0 --band-fermi-weight 0.25 --band-gap-weight 0.02 --device cuda
```

Predict/reconstruct the test split:

```bash
python bin/predict_electronic_autoencoder.py data/mp_all_bs3072/manifest_full_split.csv --ckpt out/ae_band_l512_phys/electronic_autoencoder.pt --out-dir out/ae_band_l512_phys/pred_test --split test --batch-size 32 --device cuda
```

## Stage II: CIF-to-Band-Latent Model

The Stage-II model uses a pretrained CrystaLLM-style checkpoint as the CIF encoder. Place the checkpoint at:

```text
pretrained/crystallm_v1_small/ckpt.pt
```

Train the structure-to-latent model:

```bash
python bin/train_structure_to_electronic_latent.py data/mp_all_bs3072/manifest_full_split.csv --autoencoder-ckpt out/ae_band_l512_phys/electronic_autoencoder.pt --out-dir out/s2_band_l512_phys --pretrained-ckpt pretrained/crystallm_v1_small/ckpt.pt --epochs 30 --batch-size 2 --grad-accum-steps 8 --lr 2e-5 --decoder-lr-mult 0.15 --structure-pooling attn --band-path-emb-dim 128 --head-hidden-dim 2048 --head-layers 2 --head-dropout 0.05 --latent-loss-type smooth_l1 --latent-huber-beta 0.5 --latent-loss-weight 1.0 --recon-loss-weight 0.5 --recon-start-epoch 3 --recon-warmup-epochs 12 --band-weight 1.0 --band-slope-weight 0.25 --band-curvature-weight 0.10 --band-fermi-window 1.0 --band-fermi-weight 0.5 --band-gap-weight 0.05 --band-gap-tol 1e-3 --max-grad-norm 1.0 --latent-stats-batch-size 64 --device cuda
```

Predict the test split:

```bash
python bin/predict_structure_to_electronic_latent.py data/mp_all_bs3072/manifest_full_split.csv --ckpt out/s2_band_l512_phys/structure_to_electronic_latent.pt --out-dir out/s2_band_l512_phys/pred_test --split test --batch-size 8 --device cuda
```

## Evaluation and Visualization

Compute ranked sample diagnostics:

```bash
python bin/analyze_electronic_predictions.py data/mp_all_bs3072/manifest_full_split.csv --pred-dir out/s2_band_l512_phys/pred_test --out-dir out/s2_band_l512_phys/analysis_test --split test --max-plots-per-group 10 --sort-metric band_mae --python-exe python
```

Global fitting plot:

```bash
python bin/plot_fitting_curves.py data/mp_all_bs3072/manifest_full_split.csv --pred-dir out/s2_band_l512_phys/pred_test --out-dir out/s2_band_l512_phys/fig_fitting --split test --max-points 120000 --axis-percentile 99.5
```

Error distribution and band-index/k-point heatmap:

```bash
python bin/plot_error_analysis_suite.py data/mp_all_bs3072/manifest_full_split.csv --pred-dir out/s2_band_l512_phys/pred_test --out-dir out/s2_band_l512_phys/fig_suite --split test --axis-percentile 99.5 --heatmap-percentile 99.5
```

Best/middle/worst sample visualizations:

```bash
python bin/plot_ranked_electronic_predictions.py data/mp_all_bs3072/manifest_full_split.csv --pred-dir out/s2_band_l512_phys/pred_test --out-dir out/s2_band_l512_phys/fig_test_ranked_samples --split test --sort-metric band_mae --count-per-tier 20 --python-exe python
```

Single-sample band overlay:

```bash
python bin/plot_electronic_prediction.py data/mp_all_bs3072/manifest_full_split.csv --pred-dir out/s2_band_l512_phys/pred_test --id mp-id-here --out out/sample_prediction.png
```

Single-sample fitting plot:

```bash
python bin/plot_electronic_sample_fitting.py data/mp_all_bs3072/manifest_full_split.csv --pred-dir out/s2_band_l512_phys/pred_test --id mp-id-here --out out/sample_fitting.png
```

## Current Reference Result

The `EXPERIMENT_L512_PHYS.md` file records the main internal run used to fix this version:

- Stage-I AE sample-mean MAE: `0.1917 eV`
- Stage-II sample-mean MAE: `0.4340 eV`
- Stage-II global MAE: `0.4248 eV`
- Stage-II global R2: `0.9326`

These numbers require the same processed dataset, split, pretrained encoder checkpoint, and training settings.

## Notes for GitHub Release

Before pushing:

1. Keep `data/`, `out/`, and `pretrained/` outside git.
2. Do not commit Materials Project raw data unless the license allows redistribution.
3. Do not commit private API keys, logs, or local machine paths.
4. Use a release asset or external storage for checkpoints if needed.
