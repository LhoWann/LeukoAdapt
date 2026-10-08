# LeukoAdapt: Attention-Guided Unsupervised Domain Adaptation for Robust Acute Lymphocytic Leukemia (ALL) Diagnosis

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.14+cu130](https://img.shields.io/badge/PyTorch-2.14%2Bcu130-red.svg)](https://pytorch.org/)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: 8 Passed](https://img.shields.io/badge/Tests-8%20Passed-brightgreen.svg)]()
[![Hardware: RTX 3050 Laptop](https://img.shields.io/badge/Hardware-RTX%203050%20(4GB%20VRAM)-green.svg)]()

A PyTorch deep learning framework for **Unsupervised Domain Adaptation (UDA)** and **Attention-Guided Cross-Center Translation** in peripheral blood smear diagnosis. 

This repository replicates and advances the state-of-the-art methodology from:
> **Yusuf Yargı Baydilli (2025)**, *"Unsupervised attention-guided domain adaptation model for Acute Lymphocytic Leukemia (ALL) diagnosis"*, **Biomedical Signal Processing and Control**, Vol. 101, 107159. [DOI: 10.1016/j.bspc.2024.107159](https://doi.org/10.1016/j.bspc.2024.107159)

---

## 1. Overview & Problem Definition

### A. The Challenge of Medical Domain Shift
Deep learning models trained on blood smears from one hospital routinely fail when deployed at external clinics. 
Marginal data distributions differ significantly ($P(X_s) \neq P(X_t)$) due to variations in:
- Optical microscope sensors and camera hardware.
- Jenner-Giemsa staining reaction times and chemical batches.
- Erythrocyte (RBC) background densities and illumination glare.

### B. The Conceptual Analogy
> **The Fruit Sorting Analogy**:
> Suppose two vendor stalls sell apples. Stall A sells clean, pre-cut red apples on transparent trays (*C-NMC 2019: segmented single-cell images with clean black backgrounds*). 
> Stall B sells raw green apples lying in dirty straw baskets (*ALL-IDB1: whole slide blood smears with dense red blood cell backgrounds*). 
> An automated sorting robot trained solely at Stall A will fail at Stall B because it gets confused by the straw and green hues. 
> Domain adaptation serves as an optical adapter: it converts Stall A's apples to mimic Stall B's lighting and background while strictly preserving the internal cellular anatomy of the fruit.

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
   - Extracting 510 blasts (via `.xyc` coordinates) and 349 normal leukocytes directly from ALL-IDB1 achieves the exact 859-cell target pool specified by Baydilli (2025) with complete provenance logged in `metadata_target_cells.json`.

---

## 3. End-to-End Methodology & Pipeline Flow

The framework operates via a three-phase pipeline bridging the domain gap between cleanly segmented source cells and natural microscopic target blood smears:

![LeukoAdapt Methodology Pipeline](docs/assets/methodology.png)

### Phase 1: Attention-Guided Domain Translation (GAN)
- **Objective**: Translate 2,000 labeled C-NMC source cells ($s$) into the realistic visual style of the ALL-IDB target domain ($t$) without altering cellular diagnosis.
- **Attention-Guided Generator ($G_{S \to T}$ & $A_S$)**: Produces a synthetic target image and computes a spatial attention mask ($s_a \in [0, 1]$). The blended translated cell is formed via:
  $$s' = (s_a \odot G_{S \to T}(s)) \oplus ((1 - s_a) \odot s)$$
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

```
ALL-IDB-Generalization/
├── configs/
│   └── config.yaml               # Hyperparameters for GAN and ResNet34
├── data/
│   ├── raw/                      # Downloaded raw datasets
│   │   ├── ALL-IDB/              # ALL_IDB1 (.jpg images & .xyc coordinates)
│   │   └── C-NMC_2019/           # ISBI 2019 C-NMC (fold_0, fold_1, fold_2)
│   └── processed/                # Normalized patches (128x128 resolution)
│       ├── source_cnmc/          # 2,000 C-NMC patches (1,000 ALL + 1,000 HEM)
│       └── target_all_idb/       # 859 target patches from ALL-IDB1
│           ├── cell_level/       # Scenario 1 (659 train, 200 test)
│           ├── patient_level/    # Scenario 2 (patient-isolated train and test)
│           └── metadata_target_cells.json # Full provenance tracking
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
└── README.md                     # Documentation
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
If setting up in a new environment, install dependencies using:

```powershell
# 1. Install PyTorch with CUDA 13.0 acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. Install computer vision and utility libraries
pip install -r requirements.txt
```

---

## 6. Dataset Preprocessing Pipeline

The processed datasets are structured and validated in `data/processed/`.
To reproduce the extraction and sampling from raw sources:

```powershell
# 1. Sample 2,000 balanced single-cell images from C-NMC fold_0
python src/data/sample_cnmc.py

# 2. Extract 859 target cells from ALL-IDB1 and build dual split scenarios
python src/data/extract_all_idb.py
```

### Dataset Distribution Summary

| Domain | Partition | ALL (Blast) | HEM (Normal) | Total | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Source ($D_s$)** | `source_cnmc/train` | 1,000 | 1,000 | **2,000** | Labeled training images for domain transfer |
| **Target ($D_t$) S1** | `target_all_idb/cell_level/train` | 410 | 249 | **659** | Unlabeled target images for GAN adaptation |
| **Target ($D_t$) S1** | `target_all_idb/cell_level/test` | 100 | 100 | **200** | Hidden benchmark test set (Replication) |
| **Target ($D_t$) S2** | `target_all_idb/patient_level/train`| 333 | 318 | **651** | Unlabeled target images from 39 slides |
| **Target ($D_t$) S2** | `target_all_idb/patient_level/test` | 177 | 31 | **208** | Unseen patient slides (Leak-Free Test) |

---

## 7. Execution Guide (`train.py`)

All experimental stages are managed through the centralized CLI runner [train.py](file:///D:/ALL-IDB-Generalization/train.py):

### Stage 1: Train Attention-Guided CycleGAN
Trains the generator and PatchGAN discriminator to adapt C-NMC images to ALL-IDB style:

```powershell
# Full 200 epochs on Scenario 1 (Baseline)
python train.py --stage gan --scenario cell_level --epochs_gan 200

# Smoke test (e.g. 5 epochs)
python train.py --stage gan --scenario cell_level --epochs_gan 5
```

### Stage 2: Translate Source Dataset
Uses the trained generator checkpoint to synthesize 2,000 target-style images:

```powershell
python train.py --stage translate --scenario cell_level
```

### Stage 3: Train Classifier & Evaluate
Trains ResNet34 for 50 epochs on translated images and evaluates accuracy on 200 test samples:

```powershell
python train.py --stage classifier --scenario cell_level --epochs_clf 50
```

### Full End-to-End Pipeline
Runs all three stages sequentially in a single command:

```powershell
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
| `--lr_gan` | float | `0.0001` | Initial Adam learning rate for GAN ($\beta_1=0.5, \beta_2=0.999$) |
| `--lr_clf` | float | `0.001` | Initial Adam learning rate for ResNet34 |
| `--checkpoint_gan` | str | `None` | Path to custom `.pth` checkpoint for translation |
| `--no_amp` | flag | `False` | Disable Automatic Mixed Precision |
| `--device` | str | `cuda` | Hardware execution target (`cuda` or `cpu`) |

---

## 9. Quality Assurance & Verification

Unit tests and code linting can be executed via:

```powershell
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
3. **Gupta, A., et al. (2019)**. *ISBI 2019 C-NMC Challenge: Classification of Normal vs Malignant Cells in B-ALL White Blood Cancer Microscopic Images*. TCIA.#   L e u k o A d a p t  
 