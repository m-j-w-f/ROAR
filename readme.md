# ROAR -- Road Audio Recognition

Classification of road surface types from tyre-road noise recordings of electric vehicles. Measurements were collected on two test tracks using four electric vehicles, six tyre models, and seven microphone channels per recording. The goal is to distinguish track 150 from track 211 based solely on audio features, using cross-validation schemes that test generalization across unseen vehicles and measurement conditions.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Notebooks](#notebooks)
- [Models](#models)
- [Results](#results)
- [Setup](#setup)

## Project Overview

The dataset consists of HDF5 recordings from four electric vehicles (VW ID.4, Audi Q8 e-tron, Porsche Taycan, VW E-Golf) driving on two asphalt test tracks. Each recording contains multi-channel audio (SIS trailing/leading edge microphones, NAWS sensor, ISO microphone, 2m microphone) at ~48--51 kHz, plus vehicle CAN bus signals (speed, acceleration via IMU). Six tyre models are distributed across the vehicles.

The classification task is: given a single recording, predict which of the two tracks was driven on. The difficulty lies in generalization -- the model must work for vehicles and measurement conditions it was never trained on.

## Repository Structure

```
ROAR/
  pyproject.toml                        Project configuration and dependencies (pixi)
  readme.md                             This file
  data_cleaned/                         HDF5 recordings organized by track/vehicle/tyre
    track150/
    track211/
  data_extras/                          Metadata tables
    all_measurement_channels_name.csv   Channel reference with synonyms and relevance
    speed.csv                           Measurement type definitions (constant speed / acceleration)
    tracks.csv                          Track descriptions and GPS coordinates
    tyres.csv                           Tyre specifications (manufacturer, dimensions, assigned vehicle)
  plots/                                Generated figures (PDF)
  results/                              Model evaluation results (CSV)
  src/roar/
    __init__.py                         Project-wide constants and mappings
    preprocessing/
      __init__.py                       Package exports
      load_data.py                      HDF5 file discovery, filename parsing, channel loading
      features.py                       Feature extraction (time-domain, spectral, MFCCs, vehicle-invariant)
      fix_channel_names.py              Channel name harmonization across vehicles
    notenbooks/
      00file_naming.ipynb               File renaming and dataset construction
      01explore.ipynb                   Exploratory data analysis
      02analyze.ipynb                   Signal analysis and data cleaning
      03train.ipynb                     Model training with LOGO cross-validation
      04autoencoder.ipynb               Convolutional autoencoder for latent feature learning (experimental)
      05plot_scores.ipynb               Results visualization and model comparison
      06feature_importances.ipynb       SHAP-based feature importance analysis
```

## Data

The expected data structure is

```
data_cleaned/
├── track150
│   ├── 01_ID.4
│   │   └── tyre1
│   │       ├── Messung Numbering Table.xlsx
│   │       ├── track150_ID.4_tyre1_meas1_2p5_1_2025-08-07_11-21-28.h5
│   │       ├── ...
│   │       └── track150_ID.4_tyre1_meas3_2p5_1_2025-08-07_11-28-54.h5
│   ├── 02_Q8 e-tron
│   │   ├── tyre12
│   │   │   ├── Messung Numbering Table.xlsx
│   │   │   ├── track150_Q8 e-tron_tyre12_meas0_2p5_1_2025-08-15_13-06-24.h5
│   │   │   ├── ...
│   │   │   └── track150_Q8 e-tron_tyre12_meas2_2p5_1_2025-08-15_13-15-55.h5
│   │   └── tyre6
│   │       ├── Messung Numbering Table.xlsx
│   │       ├── track150_Q8 e-tron_tyre6_meas1_2p5_1_2025-09-29_17-21-55.h5
│   │       ├── ...
│   │       └── track150_Q8 e-tron_tyre6_meas3_2p5_1_2025-09-29_17-28-02.h5
│   ├── 03_Taycan
│   │   └── tyre10
│   │       ├── Messung Numbering Table.xlsx
│   │       ├── track150_Taycan_tyre10_meas1_2p5_1_2025-09-23_17-21-41.h5
│   │       ├── ...
│   │       └── track150_Taycan_tyre10_meas3_2p5_1_2025-09-23_17-27-29.h5
│   └── 04_E-Golf
│       └── tyre13
│           ├── Messung Numbering Table.xlsx
│           ├── track150_E-Golf_tyre13_meas1_2p5_1_2025-09-26_15-20-47.h5
│           ├── ...
│           └── track150_E-Golf_tyre13_meas3_2p5_1_2025-09-26_15-30-07.h5
└── track211
    ├── 01_ID.4
    │   ├── tyre1
    │   │   ├── Messung Numbering Table1.xlsx
    │   │   ├── track211_ID.4_tyre1_meas1_2p5_1_2025-08-07_10-27-31.h5
    │   │   ├── ...
    │   │   └── track211_ID.4_tyre1_meas6_2p5_1_2025-08-07_10-53-53.h5
    │   └── tyre3
    │       ├── Messung Numbering Table.xlsx
    │       ├── track211_ID.4_tyre3_2pt6_vr100_2025-07-11_10-32-06.h5
    │       ├── ...
    │       └── track211_ID.4_tyre3_3pt1_vr80_2025-07-11_10-13-39.h5
    ├── 02_Q8 e-tron
    │   ├── tyre12
    │   │   ├── Messung Numbering Table.xlsx
    │   │   ├── track211_Q8 e-tron_tyre12_meas1_2p5_1_2025-08-15_11-46-48.h5
    │   │   ├── ...
    │   │   └── track211_Q8 e-tron_tyre12_meas6_2p5_1_2025-08-15_12-34-39.h5
    │   └── tyre6
    │       ├── Messung Numbering Table.xlsx
    │       ├── track211_Q8 e-tron_tyre6_meas1_2p5_1_2025-09-29_16-57-57.h5
    │       ├── ...
    │       └── track211_Q8 e-tron_tyre6_meas6_2p5_1_2025-09-29_17-17-53.h5
    ├── 03_Taycan
    │   └── tyre10
    │       ├── Messung Numbering Table.xlsx
    │       ├── track211_Taycan_tyre10_meas1_2p5_1_2025-09-23_17-00-35.h5
    │       ├── ...
    │       └── track211_Taycan_tyre10_meas6_2p5_1_2025-09-23_17-15-37.h5
    └── 04_E-Golf
        └── tyre13
            ├── Messung Numbering Table.xlsx
            ├── track211_E-Golf_tyre13_meas1_2p5_1_2025-09-26_14-48-57.h5
            ├── ...
            └── track211_E-Golf_tyre13_meas6_2p5_1_2025-09-26_15-15-26.h5
```


### Vehicles

| Vehicle        | Tyres                                        |
|----------------|----------------------------------------------|
| VW ID.4        | EcoContact 6 Q, RainSport 5                  |
| Audi Q8 e-tron | PremiumContact 6 AO, Ventus S1 evo 3 ev      |
| Porsche Taycan | P-Zero R                                     |
| VW E-Golf      | Summer SRTT                                  |

### Tracks

| Track | Description                                         |
|-------|-----------------------------------------------------|
| 150   | Test track                                          |
| 211   | ika Teststrecke (50.791 N, 6.049 E)                 |

### Measurement Types

| Type  | Condition                        |
|-------|----------------------------------|
| meas1 | Constant speed ~45 km/h          |
| meas2 | Constant speed ~80 km/h          |
| meas3 | Constant speed ~100 km/h         |
| meas4 | Acceleration at 1 m/s^2          |
| meas5 | Acceleration at 2 m/s^2          |
| meas6 | Acceleration at 3--4 m/s^2       |

### Microphone Channels

Seven channels per recording: `Ch_1_labV12` through `Ch_4_labV12` (SIS trailing/leading edge), `NAWSSound`, `mic_iso`, and `mic_2m`. Cleaned variants (click-removed) are stored as `{channel}_cleaned` in the same HDF5 files.

## Preprocessing

The `src/roar/preprocessing/` package handles all data loading and feature engineering so notebooks are not cluttered.

**load_data.py** -- Discovers all HDF5 files recursively, parses structured metadata (track, vehicle, tyre, measurement type, date) from filenames via regex, and returns a Polars DataFrame. Also provides a function to load individual channels from HDF5 files.

**fix_channel_names.py** -- Harmonizes channel names across vehicles. Different vehicles use different naming conventions for the same sensors (e.g., `TrailK1` vs `Ch_1_labV12`). This module loads a synonym mapping from `data_extras/all_measurement_channels_name.csv` and renames datasets in-place within the HDF5 files.

**features.py** -- Central feature extraction module. Extracts 23+ scalar features per audio channel:

- Time-domain: RMS, mean, standard deviation, max, crest factor, zero-crossing rate
- Frequency-domain (Welch PSD): spectral centroid, rolloff (95%), flatness, bandwidth, 5 band powers (0--200, 200--500, 500--1k, 1k--2k, 2k--5k Hz)
- MFCCs: 13 mean MFCC coefficients via librosa

An alternative "vehicle-invariant" feature set uses normalized and ratio-based features (log-RMS, normalized band energies, cross-band ratios, MFCC means/stds/deltas) designed to generalize across different vehicles. Speed and acceleration statistics are also extracted from CAN bus data.

## Notebooks

Run these in order to reproduce the results.

### 00file_naming.ipynb -- File Renaming and Dataset Construction

Standardizes HDF5 file naming conventions from the original measurement system format to a consistent `track{id}_{vehicle}_tyre{id}_{measurement}_{date}.h5` pattern. Investigates naming inconsistencies by comparing filename metadata against the embedded HDF5 attributes (TrackID, carName, TyreID). Tests the feature extraction pipeline on sample files and builds the initial feature dataset by iterating over all recordings.

### 01explore.ipynb -- Exploratory Data Analysis

Comprehensive overview of the dataset structure and recording characteristics. Loads metadata tables (channels, speeds, tracks, tyres) and inventories all HDF5 files, enumerating every dataset within each file along with its shape and sample rate. Filters channels by relevance. Generates a full suite of distribution plots saved as PDFs: experiment counts by tyre, vehicle, and track (pie charts, bar charts, heatmaps, stacked bars), sample rate distributions by variable name, and recording duration distributions (boxplots and histograms) broken down by channel type, vehicle, tyre, and track. Prints summary statistics for the full dataset.

### 02analyze.ipynb -- Signal Analysis and Data Cleaning

Performs deep signal analysis across multiple domains and implements the data cleaning pipeline.

### 03train.ipynb -- Model Training

Trains and evaluates 12 classifier configurations for binary road-type classification using Leave-One-Group-Out (LOGO) cross-validation. Groups can be defined by vehicle (4 folds) or measurement type (6 folds).

### 04autoencoder.ipynb -- Convolutional Autoencoder

Learns compressed latent representations from multi-channel MFCC spectrograms using a convolutional autoencoder with an auxiliary classification head, implemented in PyTorch Lightning.
This is an experimental feature and we did not persue this any further as we did not have enough data.

### 05plot_scores.ipynb -- Results Visualization

Loads all model evaluation CSVs from the `results/` directory and creates comparison plots.

### 06feature_importances.ipynb -- SHAP Feature Importance Analysis

Computes and compares SHAP-based feature importances for the four best tree-based and linear models: PCA + Logistic Regression, Random Forest, XGBoost, and LightGBM.

## Models

All models use Leave-One-Group-Out cross-validation where each fold holds out either one vehicle or one measurement type. This tests whether the classifier generalizes to conditions it has never seen during training.

| Model                  | Description                                                      |
|------------------------|------------------------------------------------------------------|
| PCA + Logistic Reg.    | PCA dimensionality reduction followed by logistic regression     |
| Random Forest          | 300-tree ensemble with threshold tuning                          |
| SVM                    | Support vector machine with RBF or polynomial kernel             |
| XGBoost                | 400-tree gradient boosting with extensive hyperparameter search  |
| LightGBM               | 400-tree gradient boosting (leaf-wise)                           |
| TabPFN                 | Prior-fitted network for small tabular datasets                  |
| TabPFN + RF/XGB/LGBM   | TabPFN transductive embeddings as input to tree models           |
| TabPFN + PCA_LR/SVC    | TabPFN embeddings as input to linear/kernel models               |
| Conv. Autoencoder      | Latent features from MFCC spectrograms with classification head  |

## Results

Model evaluation results are stored as CSV files in `results/`. Generated figures (confusion matrices, comparison bar charts, SHAP plots) are saved as PDFs in `plots/`.

## Setup

This project uses [pixi](https://pixi.sh) for environment and dependency management.

```bash
# Clone the repository
git clone https://github.com/m-j-w-f/ROAR.git
cd ROAR

# Install dependencies
pixi install
```

The project requires Python 3.13. Key dependencies include polars, h5py, librosa, PyTorch, Lightning, scikit-learn, XGBoost, LightGBM, TabPFN, and SHAP. See `pyproject.toml` for the full list.
