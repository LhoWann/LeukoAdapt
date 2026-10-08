<div align="center">

# LeukoAdapt

### Attention-Guided Unsupervised Domain Adaptation for Robust Acute Lymphocytic Leukemia (ALL) Diagnosis

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14%2Bcu130-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-000000.svg?style=flat&logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Tests: 8 Passed](https://img.shields.io/badge/Tests-8%20Passed-2ea44f.svg?style=flat&logo=pytest&logoColor=white)](#9-quality-assurance--verification)
[![Hardware: RTX 3050](https://img.shields.io/badge/Hardware-RTX%203050%20(4GB%20VRAM)-76B900.svg?style=flat&logo=nvidia&logoColor=white)](#5-setup--installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](LICENSE)

<p align="center">
  A PyTorch deep learning framework for <b>Unsupervised Domain Adaptation (UDA)</b> and <b>Attention-Guided Cross-Center Translation</b> in peripheral blood smear diagnosis.
</p>

</div>

---

> [!NOTE]
> This repository replicates and advances the state-of-the-art methodology published in:  
> **Yusuf Yargı Baydilli (2025)**, *"Unsupervised attention-guided domain adaptation model for Acute Lymphocytic Leukemia (ALL) diagnosis"*, **Biomedical Signal Processing and Control**, Vol. 101, 107159. [DOI: 10.1016/j.bspc.2024.107159](https://doi.org/10.1016/j.bspc.2024.107159).

---

## Table of Contents

- [1. Overview & Problem Definition](#1-overview--problem-definition)
- [2. Key Architectural Features & Rigorous Data Integrity](#2-key-architectural-features--rigorous-data-integrity)
- [3. End-to-End Methodology & Pipeline Flow](#3-end-to-end-methodology--pipeline-flow)
- [4. Repository Architecture](#4-repository-architecture)
- [5. Setup & Installation](#5-setup--installation)
- [6. Dataset Preprocessing Pipeline](#6-dataset-preprocessing-pipeline)
- [7. Execution Guide (`train.py`)](#7-execution-guide-trainpy)
- [8. Command-Line Options Reference](#8-command-line-options-reference)
- [9. Quality Assurance & Verification](#9-quality-assurance--verification)
- [10. References](#10-references)

---

## 1. Overview & Problem Definition

### A. The Challenge of Medical Domain Shift
Deep learning models trained on blood smears from one hospital routinely fail when deployed at external clinics. 
Marginal data distributions differ significantly ($P(X_s) \neq P(X_t)$) due to variations in:
- Optical microscope sensors and camera hardware.
- Jenner-Giemsa staining reaction times and chemical batches.
- Erythrocyte (RBC) background densities and illumination glare.

### B. The Conceptual Analogy

> [!TIP]
> **The Fruit Sorting Analogy**  
> Imagine two market stalls selling apples:
> - **Stall A (C-NMC 2019)** sells pre-cut, washed red apples on clean transparent trays (*isolated cells with black backgrounds*).
> - **Stall B (ALL-IDB1)** sells raw apples inside unwashed straw baskets (*cells surrounded by complex red blood cell backgrounds*).
> 
> An automated sorter trained only on Stall A fails at Stall B due to unfamiliar straw and lighting. **Domain adaptation acts as an optical adapter**: it adapts Stall A's images into Stall B's realistic appearance while strictly preserving cellular anatomy.

---

## 2. Key Architectural Features & Rigorous Data Integrity

### A. Architectural Highlights & Memory Optimization
1. **Attention-Guided Cellular Preservation**:
   - Integrated spatial attention modules ($A_S, A_T$) isolate leukocyte nuclei and cytoplasm from background noise.
   - PatchGAN discriminator receives attention-masked inputs ($s_a \odot t$ and $s_a \odot s'$), preventing background-memorization artifacts.
2. **Dual Evaluation Protocols**:
   - **Scenario 1 (Cell-Level Split / Baydilli Baseline)**: 659 training and 200 testing patches randomly partitioned to replicate the published ~89% accuracy benchmark.
   - **Scenario 2 (Patient-Independent Split / Leak-Free Evaluation)**: 20 entire whole-slide images isolated strictly for testing, evaluating true generalization across unseen patients.
3. **PyTorch Lightning Aesthetics & Hardware Efficiency**:
   - Terminal diagnostics display parameter tables, model sizes, and hardware metrics.
   - Configured with FP16 Automatic Mixed Precision (AMP) and an Image History Pool (50 buffers), operating seamlessly on GPUs with 4 GB VRAM (RTX 3050 Laptop).

### B. Scientific Rationale: Why Pure ALL-IDB1 Extraction Over ALL-IDB2 Integration?

A foundational design decision in this framework is the exclusive extraction of all 859 target cells directly from `ALL-IDB1` while completely omitting `ALL-IDB2`. This decision is governed by four scientific and methodological principles:

1. **Physical Subset Identity**:
   - Official documentation confirms: *"ALL-IDB2 is a collection of cropped areas of interest ... extracted from the ALL-IDB1 dataset."*
   - ALL-IDB2 contains 260 cropped cells (130 blasts, 130 normal) that are physical duplicates of cells already present within the 108 ALL-IDB1 whole-slide images.
2. **Elimination of GAN Target Distribution Distortion**:
   - In unsupervised domain translation, the discriminator models the empirical target distribution $P(X_t)$.
   - Feeding both ALL-IDB1 crops and ALL-IDB2 images into the GAN pool causes those 260 overlapping cells to be sampled twice as frequently ($2\times$ oversampling).
   - This induces artificial sampling bias, causing the generator to overfit to the specific staining and lighting nuances of those 260 cells rather than learning the global clinical distribution.
3. **Prevention of Catastrophic Information Leakage**:
   - If ALL-IDB2 were allocated as the testing benchmark while ALL-IDB1 trained the GAN, the GAN would have already processed the exact test cells during unsupervised adaptation.
   - In medical imaging benchmarks, this constitutes severe information leakage: test accuracy would be artificially inflated because the domain adapter already memorized the target test cells.
4. **Metric Integrity & Exact Baseline Reproduction**:
   - Evaluating on ALL-IDB1 and ALL-IDB2 simultaneously would double-count 260 cells, invalidating diagnostic accuracy and F1 metrics.
   - Extracting 510 blasts (via `.xyc` coordinates) and 349 normal leukocytes directly from ALL-IDB1 achieves the exact 859-cell target pool specified by Baydilli (2025) with complete provenance logged in [metadata_target_cells.json](data/processed/target_all_idb/metadata_target_cells.json).

---

## 3. End-to-End Methodology & Pipeline Flow

The framework operates via a three-phase pipeline bridging the domain gap between cleanly segmented source cells and natural microscopic target blood smears:

<p align="center">
  <img src="docs/assets/methodology.png" alt="LeukoAdapt Methodology Pipeline" width="100%" />
</p>

### Phase 1: Attention-Guided Domain Translation (GAN)
- **Objective**: Translate 2,000 labeled C-NMC source cells ($s$) into the realistic visual style of the ALL-IDB target domain ($t$) without altering cellular diagnosis.
- **Attention-Guided Generator ($G_{S \to T}$ & $A_S$)**: Produces a synthetic target image and computes a spatial attention mask ($s_a \in [0, 1]$). The blended translated cell is formed via:

  $$s' = (s_a \odot G_{S \to T}(s)) + ((1 - s_a) \odot s)$$

- **Attention-Masked PatchGAN Discriminator ($D_T$)**: Receives attention-masked real images ($s_a \odot t$) and attention-masked fake images ($s_a \odot s'$). This ensures adversarial guidance focuses purely on cell morphology and staining rather than background artifacts.
- **Optimization**: Minimized using least-squares adversarial loss ($\lambda_{gan}=0.5$), cycle-consistency loss ($\lambda_{cycle}=10.0$), and pixel identity loss ($\lambda_{pixel}=1.0$). Target labels are completely unseen and unused.

### Phase 2: Classifier Training on Transferred Domain
- **Data Source**: 2,000 translated C-NMC images ($s'$).
- **Label Inheritance**: Each translated cell inherits its ground-truth diagnosis from its source C-NMC origin ($Y_s \in \{\text{ALL}, \text{Normal}\}$).
- **Model**: ResNet34 deep convolutional neural network initialized with ImageNet pre-trained weights.
- **Training Protocol**: 50 epochs with Adam optimizer ($\text{LR}=10^{-3}$, batch size 32) using standard Cross-Entropy classification loss.

### Phase 3: Zero-Leakage Target Benchmark Evaluation
- **Target Test Set**: Evaluated on 200 real, unmodified ALL-IDB target cells (100 ALL blasts + 100 Normal leukocytes).
- **Blind Inference**: ResNet34 has never observed these test images during GAN adaptation or classifier training.
- **Target Performance**: Replicates published benchmark metrics: $\ge 89.00\%$ Accuracy, $0.8878$ F1-Score, $0.9063$ Recall, and $0.8750$ Specificity.

---

## 4. Repository Architecture

```text
ALL-IDB-Generalization/
├── configs/
│   └── config.yaml               # Hyperparameters for GAN and ResNet34
├── data/
│   ├── raw/                      # Raw datasets
│   │   ├── ALL-IDB/              # ALL_IDB1 (.jpg whole-slides & .xyc coordinates)
│   │   └── C-NMC_2019/           # ISBI 2019 C-NMC (fold_0, fold_1, fold_2)
│   └── processed/                # Normalized patches (128x128 resolution)
│       ├── source_cnmc/          # 2,000 C-NMC patches (1,000 ALL + 1,000 HEM)
│       └── target_all_idb/       # 859 target patches extracted from ALL-IDB1
│           ├── cell_level/       # Scenario 1 (659 train, 200 test)
│           ├── patient_level/    # Scenario 2 (patient-isolated train and test)
│           └── metadata_target_cells.json # Full provenance tracking
├── docs/
│   └── assets/                   # Publication-quality diagram assets
│       └── methodology.png       # 300 DPI architecture diagram
├── src/
│   ├── data/
│   │   ├── dataset.py            # Unpaired and labeled PyTorch DataLoaders
│   │   ├── extract_all_idb.py    # ALL-IDB1 centroid refinement & extraction
│   │   └── sample_cnmc.py        # Balanced sampling from C-NMC fold_0
│   ├── models/
│   │   ├── attention.py          # Spatial Attention & Fusion Modules
│   │   ├── generator.py          # Generator with 6 residual blocks & skips
│   │   ├── discriminator.py      # 70x70 PatchGAN with attention masking
│   │   ├── cyclegan.py           # Unified Attention-CycleGAN with composite losses
│   │   └── classifier.py         # ResNet34 classifier & feature extractor
│   ├── training/
│   │   ├── train_gan.py          # GAN training loop (AMP, linear LR decay)
│   │   ├── translate.py          # Batch domain translation script
│   │   └── train_classifier.py   # ResNet34 classifier training and validation
│   └── utils/
│       ├── lightning_logger.py   # Lightning-style terminal output formatting
│       ├── metrics.py            # Accuracy, F-Score, Precision, Specificity, PSNR
│       └── image_pool.py         # 50-image buffer for discriminator stabilization
├── tests/
│   ├── test_models.py            # Unit tests for tensor shapes and forward passes
│   └── test_dataset.py           # Unit tests for data loading and batch shapes
├── PRD.md                        # Formal Product Requirements Document
├── pyproject.toml                # Ruff configuration (PEP8 / 120-char line limit)
├── requirements.txt              # Project dependencies
├── train.py                      # Central CLI pipeline runner
└── README.md                     # Project documentation
```

---

## 5. Setup & Installation

### A. Virtual Environment Configuration
A dedicated Python 3.11 environment is pre-configured in `.venv`.
To activate the virtual environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### B. Dependency Installation
If setting up in a fresh environment, install dependencies using:

```bash
# 1. Install PyTorch with CUDA acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. Install computer vision and utility libraries
pip install -r requirements.txt
```

---

## 6. Dataset Preprocessing Pipeline

The processed datasets are structured and validated in `data/processed/`.
To reproduce the extraction and sampling from raw sources:

```bash
# 1. Sample 2,000 balanced single-cell images from C-NMC fold_0
python src/data/sample_cnmc.py

# 2. Extract 859 target cells from ALL-IDB1 and build dual split scenarios
python src/data/extract_all_idb.py
```

### Dataset Distribution Summary

| Domain | Partition | ALL (Blast) | HEM (Normal) | Total | Purpose |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Source ($D_s$)** | `source_cnmc/train` | 1,000 | 1,000 | **2,000** | Labeled training images for domain transfer |
| **Target ($D_t$) S1** | `target_all_idb/cell_level/train` | 410 | 249 | **659** | Unlabeled target images for GAN adaptation |
| **Target ($D_t$) S1** | `target_all_idb/cell_level/test` | 100 | 100 | **200** | Hidden benchmark test set (Replication) |
| **Target ($D_t$) S2** | `target_all_idb/patient_level/train` | 333 | 318 | **651** | Unlabeled target images from 39 slides |
| **Target ($D_t$) S2** | `target_all_idb/patient_level/test` | 177 | 31 | **208** | Unseen patient slides (Leak-Free Test) |

### Source Cohort Selection & Status of Remaining C-NMC Folds

The ISBI 2019 C-NMC challenge provides 10,661 single-cell images partitioned by patient into three training folds (`fold_0`, `fold_1`, and `fold_2`). In strict accordance with the baseline protocol from Baydilli (2025), our pipeline selects a balanced cohort of **2,000 images** (1,000 ALL + 1,000 HEM) sampled exclusively from `fold_0`.

#### A. Methodological Rationale (Why only 2,000 images from Fold 0?)
1. **Prior Class Balance**: Raw C-NMC data exhibits severe class imbalance (7,272 ALL vs 3,389 HEM, ~2.15:1 ratio). Drawing exactly 1,000 samples per class guarantees an unbiased uniform prior ($P(\text{ALL}) = P(\text{Normal}) = 0.5$).
2. **CycleGAN Computational Feasibility**: Training 2 generators and 2 PatchGAN discriminators with 10,661 images would require over 36 hours on a 4GB GPU. A 2,000-image source cohort matches the target domain scale (859 cells) while keeping epoch runtimes practical (~2.3 min/epoch).
3. **Structured Patient Grouping**: ISBI organized folds by patient ID. Drawing 2,000 samples strictly from `fold_0` prevents demographic bleeding and preserves patient-level encapsulation.

#### B. Allocation Status of Remaining Folds (`fold_1` & `fold_2`)
Neither `fold_1` nor `fold_2` is currently active in the training loop; both remain dormant in `data/raw/C-NMC_2019/`:

| Subset | Sample Volume | Pipeline Status | Role in Framework |
| :--- | :---: | :---: | :--- |
| **`fold_0` (Selected)** | 2,000 cells | **Active** | Source images translated via GAN $\rightarrow$ Train ResNet34 |
| **`fold_0` (Surplus)** | 1,527 cells | *Dormant* | Unused source cells (1,397 ALL + 130 HEM) |
| **`fold_1`** | 3,567 cells | *Dormant* | Available for source validation or data volume scaling |
| **`fold_2`** | 3,567 cells | *Dormant* | Available for cross-cohort source testing |

#### C. Why Fold 1 is Not Used as Validation in UDA
In Unsupervised Domain Adaptation (UDA), the primary scientific objective is assessing **cross-domain transferability** to the target clinical clinic (ALL-IDB), not in-domain accuracy within the source hospital (C-NMC). Evaluating ResNet34 on `fold_1` measures only source-domain memorization. Consequently, Baydilli (2025) benchmarks classifier performance directly on 200 blind ALL-IDB target cells. Researchers may optionally designate `fold_1` as a source validation set for early stopping if strictly isolating hyperparameter tuning from target test data.

---

## 7. Execution Guide (`train.py`)

All experimental stages are managed through the centralized CLI runner [train.py](train.py):

### Stage 1: Train Attention-Guided CycleGAN
Trains the generator and PatchGAN discriminator to adapt C-NMC images to ALL-IDB style:

```bash
# Full 200 epochs on Scenario 1 (Baseline)
python train.py --stage gan --scenario cell_level --epochs_gan 200

# Smoke test (e.g. 5 epochs)
python train.py --stage gan --scenario cell_level --epochs_gan 5
```

### Stage 2: Translate Source Dataset
Uses the trained generator checkpoint to synthesize 2,000 target-style images:

```bash
python train.py --stage translate --scenario cell_level
```

### Stage 3: Train Classifier & Evaluate
Trains ResNet34 for 50 epochs on translated images and evaluates accuracy on 200 test samples:

```bash
python train.py --stage classifier --scenario cell_level --epochs_clf 50
```

### Full End-to-End Pipeline
Runs all three stages sequentially in a single command:

```bash
python train.py --stage all --scenario cell_level --epochs_gan 200 --epochs_clf 50
```

---

## 8. Command-Line Options Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--stage` | str | `all` | Pipeline stage: `gan`, `translate`, `classifier`, or `all` |
| `--scenario` | str | `cell_level` | Evaluation scenario: `cell_level` or `patient_level` |
| `--epochs_gan` | int | `200` | Number of training epochs for Attention-CycleGAN |
| `--decay_epoch_gan` | int | `100` | Starting epoch for linear learning rate decay |
| `--epochs_clf` | int | `50` | Number of training epochs for ResNet34 classifier |
| `--batch_size_gan` | int | `1` | Batch size for GAN (1 recommended for 4GB VRAM) |
| `--batch_size_clf` | int | `32` | Batch size for ResNet34 training |
| `--lr_gan` | float | `0.0001` | Initial Adam learning rate for GAN (`beta1=0.5, beta2=0.999`) |
| `--lr_clf` | float | `0.001` | Initial Adam learning rate for ResNet34 |
| `--checkpoint_gan` | str | `None` | Path to custom `.pth` checkpoint for translation |
| `--no_amp` | flag | `False` | Disable Automatic Mixed Precision |
| `--device` | str | `cuda` | Hardware execution target (`cuda` or `cpu`) |

---

## 9. Quality Assurance & Verification

Unit tests and code linting can be executed via:

```bash
# Run the 8-test validation suite (tensor shapes, loss gradients, data loading)
pytest -v tests/

# Execute Ruff static linter and formatting checks
ruff check src tests train.py
ruff format --check src tests train.py
```

- **PyTest Result**: `8 passed in 11.08s`
- **Linter Status**: `All checks passed! (0 errors, 100% formatted)`

---

## 10. References

1. **Baydilli, Y. Y. (2025)**. *Unsupervised attention-guided domain adaptation model for Acute Lymphocytic Leukemia (ALL) diagnosis*. Biomedical Signal Processing and Control, 101, 107159.
2. **Labati, R. D., Piuri, V., & Scotti, F. (2011)**. *All-IDB: The acute lymphoblastic leukemia image database for image processing*. In 18th IEEE International Conference on Image Processing (ICIP), pp. 2045–2048.
3. **Gupta, A., et al. (2019)**. *ISBI 2019 C-NMC Challenge: Classification of Normal vs Malignant Cells in B-ALL White Blood Cancer Microscopic Images*. TCIA.