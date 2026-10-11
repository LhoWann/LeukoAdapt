<div align="center">

# LeukoAdapt

### Attention-Guided Unsupervised Domain Adaptation for Acute Lymphoblastic Leukemia (ALL) Diagnosis

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14%2Bcu130-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-000000.svg?style=flat&logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)

<p align="center">
  A PyTorch framework that translates segmented C-NMC leukocytes into the visual style of ALL-IDB blood smears with an attention-guided CycleGAN, then trains a classifier that is tested on real ALL-IDB cells it has never seen.
</p>

<p align="center"><b>English</b> | <a href="README.id.md">Bahasa Indonesia</a></p>

</div>

---

> [!NOTE]
> This repository is a **modified version** of the method of
> **Yusuf Yargı Baydilli (2025)**, *"Unsupervised attention-guided domain adaptation model for Acute Lymphocytic Leukemia (ALL) diagnosis"*, **Biomedical Signal Processing and Control**, Vol. 101, 107159. [DOI: 10.1016/j.bspc.2024.107159](https://doi.org/10.1016/j.bspc.2024.107159).
>
> Implemented exactly as written, the paper's GAN collapses to an identity mapping on this data. Two changes make it work (source cells pasted onto real ALL-IDB backgrounds, and a discriminator that sees whole images); everything else follows the paper. The literal paper GAN remains available with `--real_mask source --no_composite`. **Scenario 1** uses the paper's evaluation protocol; **Scenario 2** is a stricter, leak-free split. All deviations are listed in [Section 5](#5-reproduction-notes-deviations-from-the-paper).

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Methodology](#2-methodology)
- [3. Datasets](#3-datasets)
- [4. Evaluation Scenarios](#4-evaluation-scenarios)
- [5. Reproduction Notes: Deviations from the Paper](#5-reproduction-notes-deviations-from-the-paper)
- [6. Repository Structure](#6-repository-structure)
- [7. Setup](#7-setup)
- [8. Usage](#8-usage)
- [9. Command-Line Options](#9-command-line-options)
- [10. Testing](#10-testing)
- [11. References](#11-references)

---

## 1. Overview

### A. Problem: Domain Shift Between Laboratories

A classifier trained on blood-smear images from one laboratory usually degrades on images from another, because the marginal distributions differ ($P(X_s) \neq P(X_t)$):
- microscope optics and camera hardware;
- Jenner-Giemsa staining time and reagent batches;
- background: C-NMC cells are segmented on black, ALL-IDB cells sit among red blood cells.

Labelling a new dataset for every laboratory is expensive, so the target domain is treated as **unlabelled**.

### B. Approach

> [!TIP]
> **Analogy.** Stall A (C-NMC) sells washed apples on clean trays; Stall B (ALL-IDB) sells the same apples in straw baskets. A sorter trained only at Stall A fails at Stall B. The GAN acts as an adapter that re-dresses Stall A's apples as Stall B's, without changing the apples themselves, so the sorter can be trained on labelled Stall A data and still work at Stall B.

---

## 2. Methodology

<p align="center">
  <img src="docs/assets/methodology.png" alt="LeukoAdapt methodology overview" width="100%" />
  <br><em>Figure 1. Three-phase pipeline: unsupervised domain translation, classifier training on the translated source, blind evaluation on the target domain.</em>
</p>

### Phase 1: Attention-Guided Domain Translation (GAN)

- **Objective**: translate the labelled C-NMC source cells ($s$) into the style of the ALL-IDB target domain ($t$) without changing their diagnosis.
- **Background compositing** (modification): every source cell is pasted onto a real, cell-free ALL-IDB background patch before translation ([Section 3.E](#e-target-background-bank)), so the GAN adapts the cell instead of trying to paint red blood cells into a black area.
- **Attention-guided generator ($G_{S \to T}$, $A_S$)**: an encoder with spatial attention after every convolution block, six residual blocks, and a decoder with skip connections produce a content image $G(s)$. The attention module computes a one-channel mask from it, $s_a = \sigma\big(f_{7\times7}([\mathrm{AvgPool}(G(s)); \mathrm{MaxPool}(G(s))])\big)$, and blends:

  $$s' = s_a \odot G_{S \to T}(s) + (1 - s_a) \odot s$$

- **Cycle**: $F_{T \to S}$ and $A_T$ map $s'$ back to $s''$. Only the $S \to T \to S$ cycle and the target discriminator are trained, as in the paper.
- **PatchGAN discriminator ($D_T$)** (modification): compares the whole translated image $s'$ with a whole real target image $t$. The paper's Eq. 3 compares $s_a \odot s'$ with $s_a \odot t$, which has a degenerate optimum ([Section 5.A](#a-why-the-gan-is-modified)). A 50-image history buffer stabilises the discriminator.
- **Objective** (as in the paper): $\mathcal{L} = 0.5\,\mathcal{L}_{GAN} + 10\,\lVert s - s'' \rVert_1 + 1\,\lVert s - s' \rVert_1$, with least-squares adversarial terms and the discriminator loss halved. The pixel term is the SimGAN self-regularisation cited by the paper (ref. [105]).
- **Training** (as in the paper): 200 epochs, Adam ($\beta_1 = 0.5$, $\beta_2 = 0.999$), learning rate $10^{-4}$ decaying linearly to 0 from epoch 100, horizontal-flip augmentation, target classes padded to 1:1 with flipped copies (paper Section 5.1.2).
- **Checkpoint choice**: a checkpoint and a preview grid (composite source / $s'$ / $s_a$) are saved every 5 epochs. The paper keeps the visually best checkpoint; pass it with `--checkpoint_gan`. A `WARNING` is printed if the translation collapses ($s' \approx s$).

### Phase 2: Classifier Training on the Translated Source

- **Data**: the 2,000 translated C-NMC images $s'$, each inheriting the label of its source image ($Y_s \in \{\text{ALL}, \text{Normal}\}$).
- **Model**: ResNet34 with ImageNet weights.
- **Training**: 50 epochs, Adam, learning rate $10^{-3}$, batch size 32, cross-entropy, no weight decay.
- **Model selection**: the last epoch, as in the paper. `--source_val` instead selects the epoch on translated C-NMC `fold_2` cells; target data is never used for selection.

### Phase 3: Blind Evaluation on the Target Domain

- The classifier is run once on real, unmodified ALL-IDB test cells it has never seen.
- `results.json` stores TN, TP, FP, FN, accuracy with a Wilson 95% interval, balanced accuracy, precision, recall, specificity, NPV, F-score, AUROC, the same numbers in the paper's Table 5 column convention ([Section 5.B](#b-table-5-metric-labels)), and per-image predictions.
- `python main.py evaluate` gathers every run into one comparison table and runs pairwise McNemar tests, as in the paper's Section 5.1.3.
- **Published reference** (paper Table 5, proposed model, Scenario 1 protocol): TN = 91, TP = 87, accuracy 0.8900, F-score 0.8878.

---

## 3. Datasets

### A. Source Domain: C-NMC 2019

The ISBI 2019 C-NMC training data contains 10,661 segmented single cells in three subject-disjoint folds. The paper uses 1,000 ALL + 1,000 Normal (HEM) images without naming a fold; this repository samples them from `fold_0`, cycling over subjects so that no subject dominates (19 ALL and 9 HEM subjects are represented).

| Fold | ALL | HEM | Use |
| :--- | :---: | :---: | :--- |
| `fold_0` | 2,397 | 1,130 | 1,000 ALL + 1,000 HEM **training source** |
| `fold_1` | 2,418 | 1,163 | Unused |
| `fold_2` | 2,457 | 1,096 | 500 ALL + 500 HEM optional validation for `--source_val` (subject-disjoint from training) |

### B. Target Domain: ALL-IDB1 Images, Expert Annotations Only

All target patches are cropped from the **ALL-IDB1** slides with one procedure (257 x 257 crop centred on the refined nucleus centroid, bicubic resize to 128 x 128). Every label comes from the dataset's own annotations:

| Class | Annotation source | Cells | Slides |
| :--- | :--- | :---: | :---: |
| ALL (blast) | ALL-IDB1 `.xyc` blast centroids | 510 | 49 ALL slides |
| Normal | ALL-IDB2 `*_0.tif` crops (*"the cell placed in the center of the image is not a blast"*, healthy individuals), located in their ALL-IDB1 source slide by template matching | 125 | 58 healthy slides |

- **ALL-IDB2 is used only as an annotation.** Template matching confirms that all 260 ALL-IDB2 crops are cut from ALL-IDB1 slides (normalised correlation > 0.9 for every crop). Adding them as extra images would duplicate cells, and cropping both classes from the same ALL-IDB1 JPEGs prevents the classifier from separating the classes by file format (ALL-IDB2 is TIFF).
- **Duplicates removed.** ALL-IDB1 `Im108_0.jpg` is a pixel-identical copy of `Im093_0.jpg` and is never used. Five ALL-IDB2 Normal crops repeat a cell (`Im222`-`Im227` vs `Im256`-`Im260`), so the 130 crops contain 125 unique cells.
- **Provenance** (source slide, centroid, ALL-IDB2 file names, match score, split assignments) is logged in `data/processed/target_all_idb/metadata_target_cells.json`.

### C. Normal Cells: Why 125 Instead of the Paper's 349

> [!IMPORTANT]
> The paper reports 510 ALL + 349 Normal cells (Table 1) without saying where the Normal cells come from. ALL-IDB1 annotates only blasts, and its healthy slides contain few white blood cells: a stain-based detector finds about 106 leukocyte-sized objects there, 98 of which are already among the 125 ALL-IDB2 cells. Of the other 8, two are platelet clumps, one is a pair of touching neutrophils, three lie on the duplicate slide `Im108_0`, and two look like valid leukocytes (`Im090_0`, `Im091_0`) but carry no dataset label. **ALL-IDB contains no source of 349 correctly labelled Normal cells**, so this repository uses the 125 expert-labelled cells.

**The earlier reconstruction was unreliable.** An earlier version filled the 349 Normal cells by running the stain detector on healthy slides (162 detections) and then on ALL slides away from annotated blasts (187 detections). Thirty evenly spaced patches from each source were inspected (non-expert visual check):

<p align="center">
  <img src="docs/assets/normal_cells_healthy_slides.png" alt="Old Normal patches from healthy slides" width="100%" />
  <br><em>Figure 2. Old reconstruction, healthy slides (sample of 30, numbered). Intact leukocytes are mixed with platelets and small fragments (for example 4, 6, 7, 8, 10, 12, 14).</em>
</p>

<p align="center">
  <img src="docs/assets/normal_cells_all_slides.png" alt="Old Normal patches from ALL slides" width="100%" />
  <br><em>Figure 3. Old reconstruction, ALL slides (sample of 30, numbered). Mostly smudge cells, blast-like cells and annotation arrows (6, 7, 16), which a patient's slide cannot guarantee to be normal.</em>
</p>

| Source of old Normal patch | Intact leukocyte | Platelet / fragment | Smudge / broken cell | Blast-like cell | Annotation arrow / empty | Ambiguous |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Healthy slides (162) | 17 / 30 | 13 / 30 | 0 | 0 | 0 | 0 |
| ALL slides (187) | 5 / 30 | 0 | 12 / 30 | 6 / 30 | 5 / 30 | 2 / 30 |

**The current Normal set** uses only the cells the dataset authors labelled. Every patch is centred on its annotated cell. About 27 of them are small, pale cells; they are kept because the dataset labels them "not a blast", and no label is changed by judgement in this repository.

<p align="center">
  <img src="docs/assets/normal_cells_final_idb2.png" alt="All 125 Normal patches in use" width="100%" />
  <br><em>Figure 4. All 125 Normal patches in use, cropped from ALL-IDB1 at the ALL-IDB2 annotation positions.</em>
</p>

### D. Distribution Summary

| Domain | Partition | ALL | Normal | Total | Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Source | `source_cnmc/train` | 1,000 | 1,000 | **2,000** | Labelled images translated by the GAN; classifier training data |
| Source | `source_cnmc/val` | 500 | 500 | **1,000** | Optional validation (`--source_val`) |
| Target, Scenario 1 | `target_all_idb/cell_level/train` | 410 | 25 | **435** | GAN target pool |
| Target, Scenario 1 | `target_all_idb/cell_level/test` | 100 | 100 | **200** | Blind test set (paper protocol) |
| Target, Scenario 1 | `target_all_idb/cell_level/background` | - | - | **640** | Background bank |
| Target, Scenario 2 | `target_all_idb/slide_level/train` | 382 | 95 | **477** | GAN target pool |
| Target, Scenario 2 | `target_all_idb/slide_level/test` | 113 | 25 | **138** | Blind test set on unseen slide groups |
| Target, Scenario 2 | `target_all_idb/slide_level/background` | - | - | **889** | Background bank |

Target labels are used only to flip-balance the GAN target pool, as in the paper; the classifier never sees them.

### E. Target Background Bank

For each scenario, [backgrounds.py](src/data/backgrounds.py) crops 257 x 257 patches from the slides on the **training side** of that scenario's split (up to 12 per slide), keeping a patch only when its centre is at least 220 px from every annotated cell and it contains no nucleus-like stain and no orange annotation arrow. Each source cell is then pasted onto a patch: the cell mask is the non-black area of the segmented C-NMC image, eroded by 2 px (the resized segmentation edge is blended with black) and feathered by 1 px. During GAN training the patch is drawn at random; for translation and for the `composite` baseline it is chosen deterministically from the file name, so results are reproducible.

<p align="center">
  <img src="docs/assets/composite_examples.png" alt="C-NMC cells and their composites" width="100%" />
  <br><em>Figure 5. Top: segmented C-NMC cells (four ALL, four Normal). Bottom: the same cells pasted onto ALL-IDB background patches, the input of the GAN.</em>
</p>

---

## 4. Evaluation Scenarios

Both scenarios run exactly the same GAN, translation and classifier; only the target split (and therefore the background bank) differs. The gap between their results estimates how much the cell-level split flatters the method.

### A. Scenario 1: Cell-Level Split (Paper Protocol)

- Seeded, stratified random split of **cells**: 100 ALL + 100 Normal for testing, as in the paper; the remaining 410 ALL + 25 Normal form the GAN target pool (paper: 249 Normal, see [Section 3.C](#c-normal-cells-why-125-instead-of-the-papers-349)).
- All 510 `.xyc` blasts are kept, as in the paper. Cells from one photograph can therefore fall on both sides, and 5 physical cells photographed twice have one copy in train and one in test. Background patches never contain annotated cells.
- The classifier uses the last epoch, as in the paper.

### B. Scenario 2: Slide-Group Split (Leak-Free)

**1. Overlapping fields of view are grouped.** ALL-IDB1 contains photographs of overlapping fields of the same smear, re-taken with a different exposure (for example, `Im036_0` and `Im103_0` show the same neutrophil shifted by 152 px). [slide_overlap.py](src/data/slide_overlap.py) cuts a 5 x 5 grid of templates from every slide at 1/8 resolution and matches them, with normalised cross-correlation > 0.9, against every slide of the same size and class. It finds **15 overlapping pairs** forming **8 groups**, each assigned to train or test as a whole:

| Group | Slides | Class |
| :--- | :--- | :--- |
| 1 | `Im009_1`, `Im010_1`, `Im011_1` | ALL |
| 2 | `Im013_1`, `Im014_1`, `Im015_1` | ALL |
| 3 | `Im018_1`, `Im019_1` | ALL |
| 4 | `Im020_1`, `Im021_1`, `Im022_1` | ALL |
| 5 | `Im036_0`, `Im102_0`, `Im103_0` | Healthy |
| 6 | `Im038_0`, `Im039_0`, `Im044_0` | Healthy |
| 7 | `Im078_0`, `Im089_0` | Healthy |
| 8 | `Im094_0`, `Im095_0` | Healthy |

The search takes about 10 minutes and is cached in `data/processed/target_all_idb/slide_overlaps.json`; delete that file to repeat it.

**2. Each physical cell is counted once.** Using the offsets of the overlapping pairs, an annotated cell that maps to within 40 px of a same-class cell on the other slide is the same physical cell. **20 such copies** (15 ALL, 5 Normal) are excluded (`"scenario_2_split": "duplicate"` in the metadata), leaving 495 ALL and 120 Normal unique cells.

**3. Test groups are drawn per stratum.** ALL-IDB1 has two acquisition formats: `Im001_1`-`Im033_1` are 1712 x 1368 px, the other ALL slides and the healthy slides are 2592 x 1944 px (one healthy slide is 1226 x 652 px). Blast nuclei have a similar size in both formats (median area about 22,500 vs 21,100 px), but the test set should cover both. For every (class, image size) stratum, slide groups are shuffled with the seed and moved to the test set until it holds at least 20% of the stratum's unique cells.

| Stratum | Train | Test | Excluded copies |
| :--- | :---: | :---: | :---: |
| ALL, 1712 x 1368 | 156 | 49 | 15 |
| ALL, 2592 x 1944 | 226 | 64 | 0 |
| Normal, 2592 x 1944 | 95 | 24 | 5 |
| Normal, 1226 x 652 | 0 | 1 | 0 |
| **Total** | **477** (382 ALL + 95 Normal) | **138** (113 ALL + 25 Normal) | **20** |

Test slides: `Im003_1`, `Im005_1`, `Im016_1`, `Im048_1`, `Im049_1`, `Im052_1`, `Im056_1` (ALL) and `Im037_0`, `Im041_0`, `Im043_0`, `Im070_0`, `Im076_0`, `Im080_0`, `Im087_0`, `Im092_0`, `Im101_0`, `Im105_0`, `Im107_0` (healthy).

**4. Background bank.** Scenario 2's 889 background patches come only from the 87 slides of the training groups, so no pixel of a test slide reaches the GAN or the classifier.

**5. Model selection.** Scenario 2 is always run with `--source_val` (`python main.py run-all` adds it): the classifier epoch is chosen on translated C-NMC `fold_2` cells, so no target data influences model selection.

**6. Checks.** `extract_all_idb.py` raises an error if a slide group appears on both sides, and [test_splits.py](tests/test_splits.py) verifies on the generated metadata that no slide group and no physical cell crosses the split.

**7. Reading the results.**
- The test set is imbalanced (113 ALL vs 25 Normal), so **balanced accuracy and AUROC** are the primary metrics: predicting ALL for every cell already gives 81.9% plain accuracy.
- The Wilson 95% interval shows the uncertainty of a 138-cell test set; McNemar tests in `reports/summary.md` show whether two methods differ significantly.

**8. Limits.** ALL-IDB1 has no patient IDs. Grouping removes the leakage visible in the pixels, but different, non-overlapping photographs of one patient can still fall on both sides. Scenario 2 is leak-free at the level of photographs and physical cells, not guaranteed at the level of patients.

---

## 5. Reproduction Notes: Deviations from the Paper

| Topic | Paper | This repository | Why |
| :--- | :--- | :--- | :--- |
| Source input of the GAN | Segmented C-NMC cells on black | Cells pasted onto real ALL-IDB backgrounds (`--no_composite` for the paper) | A generator cannot paint red blood cells into a flat black area ([5.A](#a-why-the-gan-is-modified)) |
| Discriminator input | $s_a \odot t$ vs $s_a \odot s'$ (Eq. 3) | Whole $t$ vs whole $s'$ (`--real_mask source` for the paper) | Eq. 3 collapses to $s' = s$ ([5.A](#a-why-the-gan-is-modified)) |
| Loss weights, optimiser, epochs | $\lambda$ = 0.5 / 10 / 1, Adam $10^{-4}$, 200 epochs | Same | - |
| Normal cells | 349, source not stated | 125 expert-labelled cells | No other correctly labelled source exists ([3.C](#c-normal-cells-why-125-instead-of-the-papers-349)) |
| Scenario 1 target pool | 410 ALL + 249 Normal | 410 ALL + 25 Normal | Test protocol (100 + 100) kept exactly |
| C-NMC fold | Not stated | `fold_0`, subject round-robin | Reproducible choice |
| Table 5 labels | Precision / Recall / Specificity columns | Correct metrics plus the paper's convention | Columns are mislabelled ([5.B](#b-table-5-metric-labels)) |
| Classifier details | Optimiser and pretraining not stated | Adam, ImageNet weights, no weight decay | Common defaults |
| Numerical precision | Not stated | FP16 mixed precision (`--no_amp` to disable) | Fits a 4 GB GPU; does not change the method |
| Extra baseline | - | `composite`: composites without the GAN | Separates the effect of the GAN from the effect of the background |

### A. Why the GAN Is Modified

**The literal objective collapses.** Because the real and fake discriminator inputs of Eq. 3 share the same mask $s_a$, a closed mask ($s_a = 0$) makes both inputs zero, so $D_T$ can no longer tell them apart, while the pixel and cycle losses both reach 0 at $s' = s$. On C-NMC $\rightarrow$ ALL-IDB this optimum is reached within one epoch: the 200-epoch checkpoints of the original implementation had $s_a = 0.000$ everywhere.

**Removing the mask alone does not help.** With whole images, the discriminator separates the domains by the background (black vs red blood cells) and wins outright, and the generator, unable to paint red blood cells into a flat black area, drifts back towards the identity.

**Compositing fixes both.** With the source cell already on a real ALL-IDB background, the discriminator has to judge the cell itself; the adversarial game stays balanced and the translation does not collapse, with the paper's loss weights unchanged.

Probes of 4,000 to 6,000 steps (2 to 3 epochs, FP32, Scenario 1 data):

| Source input | Discriminator input | $\lambda_{pixel}$ / $\lambda_{identity}$ | Mask mean | Mean $\lvert s' - s \rvert$ | $D_T$ loss | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Black background (paper) | $s_a \odot t$ vs $s_a \odot s'$ (paper) | 1 / 0 | 0.000 | 0.000 | 0.19 | Collapsed ($s' = s$) |
| Black background (paper) | $s_a \odot t$ vs $s_a \odot s'$ (paper) | 0 or 0.1 / 0 | 0.000 | 0.000 | 0.18-0.19 | Collapsed |
| Black background (paper) | $t_a \odot t$ vs $s_a \odot s'$ (UAIT [21]) | 1 / 0 | 0.013 | 0.003 | 0.005 | Collapsed |
| Black background (paper) | Whole images | 0 / 0 | 0.497 | 0.058 (falling) | 0.002 | $D_T$ wins; drifts to identity |
| Black background (paper) | Whole images | 0 / 5 | 0.204 | 0.044 (falling) | 0.003 | $D_T$ wins; drifts to identity |
| **Composite (default)** | **Whole images (default)** | **1 / 0** | 0.296 | 0.072 (stable) | 0.086 | **Balanced game, stain adapted** |
| Composite | Whole images | 0 / 5 | 0.357 | 0.078 (stable) | 0.085 | Balanced game |

The default therefore keeps the paper's losses ($\lambda_{pixel} = 1$, no identity loss). `--lambda_identity` and `--real_mask` remain available for ablations.

### B. Table 5 Metric Labels

The paper's Table 5 columns are mislabelled. From its own confusion matrix for the proposed model (TN = 91, TP = 87 on 100 + 100 test cells):

| Paper column | Printed value | Actual quantity | Correct value of the named metric |
| :--- | :---: | :--- | :---: |
| Precision | 0.8700 | Recall $TP/(TP+FN)$ | 0.9063 |
| Recall | 0.9063 | Precision $TP/(TP+FP)$ | 0.8700 |
| Specificity | 0.8750 | NPV $TN/(TN+FN)$ | 0.9100 |

Accuracy (0.8900) and F-score (0.8878) are unaffected. `results.json` stores the correct metrics in `test_metrics` and the paper's convention in `test_metrics_paper_table5_columns`; [test_metrics.py](tests/test_metrics.py) reproduces the paper's row.

---

## 6. Repository Structure

```text
ALL-IDB-Generalization/
├── data/
│   ├── raw/
│   │   ├── ALL-IDB/                 # ALL_IDB1 (.jpg slides, .xyc blast centroids), ALL_IDB2 (.tif crops)
│   │   └── C-NMC_2019/              # fold_0, fold_1, fold_2
│   └── processed/                   # 128 x 128 patches
│       ├── source_cnmc/             # train (fold_0) and val (fold_2)
│       └── target_all_idb/
│           ├── cell_level/          # Scenario 1: train, test, background
│           ├── slide_level/         # Scenario 2: train, test, background
│           ├── metadata_target_cells.json
│           └── slide_overlaps.json  # Cached overlap search
├── docs/
│   ├── PRD.md                       # Product requirements document
│   ├── generate_methodology_diagram.py
│   └── assets/                      # Figures 1-5
├── src/
│   ├── pipeline/                    # Command implementations behind main.py
│   │   ├── train.py                 # main.py train: one stage or one baseline
│   │   ├── run_all.py               # main.py run-all: the whole study
│   │   ├── prepare_data.py          # main.py prepare-data
│   │   └── evaluate.py              # main.py evaluate: final comparison
│   ├── data/
│   │   ├── dataset.py               # Unpaired (GAN) and labelled (classifier) datasets
│   │   ├── sample_cnmc.py           # C-NMC sampling
│   │   ├── extract_all_idb.py       # ALL-IDB cropping, both splits, background banks
│   │   ├── slide_overlap.py         # Overlapping field-of-view detection
│   │   ├── backgrounds.py           # Background bank and compositing
│   │   └── stain_norm.py            # Reinhard baseline
│   ├── models/
│   │   ├── attention.py             # Spatial attention and attention fusion (A_S, A_T)
│   │   ├── generator.py             # Attention generator
│   │   ├── discriminator.py         # PatchGAN
│   │   ├── cyclegan.py              # GAN losses (paper and modified discriminator inputs)
│   │   └── classifier.py            # ResNet34
│   ├── training/
│   │   ├── train_gan.py             # GAN training, checkpoints, previews, collapse check
│   │   ├── translate.py             # Source translation
│   │   └── train_classifier.py      # Classifier training and blind test
│   └── utils/
│       ├── metrics.py               # Classification metrics, Table 5 mapping, McNemar test, PSNR
│       ├── image_pool.py            # History buffer
│       └── console.py               # Terminal output
├── tests/                           # Unit and data-integrity tests
├── checkpoints/                     # Created by training (models, previews, results.json)
├── reports/                         # Created by main.py evaluate (summary.md, summary.json)
├── logs/                            # Created by main.py run-all
├── main.py                          # Single entry point
├── pyproject.toml                   # Ruff configuration
└── requirements.txt
```

---

## 7. Setup

A Python 3.11 environment is expected in `.venv` (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

In a fresh environment:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Place the raw datasets as shown in [Section 6](#6-repository-structure). The default settings (GAN batch size 1, mixed precision) fit a 4 GB GPU; one GAN epoch takes about two minutes on an RTX 3050 Laptop GPU, so a full run of both scenarios takes roughly 15 hours.

---

## 8. Usage

### A. Run the Whole Study (Recommended)

```powershell
python main.py run-all
```

`python main.py run-all` executes, in order, and stops at the first failing step:

| Step | Command |
| :--- | :--- |
| 1 | `python main.py prepare-data` (first run about 10 minutes) |
| 2 | `python -m pytest -q tests` |
| 3 | Scenario 1: `python main.py train --stage all --scenario cell_level` |
| 4-7 | Scenario 1 baselines: `source_only`, `composite`, `reinhard`, `target_supervised` |
| 8 | Scenario 2: `python main.py train --stage all --scenario slide_level --source_val` |
| 9-12 | Scenario 2 baselines (with `--source_val`) |
| 13 | `python main.py evaluate` |

Useful variants:

```powershell
python main.py run-all --dry_run                                   # print the commands only
python main.py run-all --scenarios slide_level --skip_data         # Scenario 2 only, data already prepared
python main.py run-all --skip_data --epochs_gan 1 --epochs_clf 1   # smoke test (about 15 minutes)
```

Every banner and the final summary are also written to `logs/run_all_<timestamp>.log`.

### B. Run Stages Separately

```powershell
python main.py prepare-data
python main.py train --stage gan --scenario slide_level
python main.py train --stage translate --scenario slide_level --source_val --checkpoint_gan checkpoints/cyclegan_slide_level/attention_cyclegan_epoch_150.pth
python main.py train --stage classifier --scenario slide_level --source_val
python main.py train --stage baseline --baseline composite --scenario slide_level --source_val
python main.py evaluate
```

Without `--checkpoint_gan`, translation uses the latest checkpoint. To follow the paper's visual selection, inspect `checkpoints/cyclegan_<scenario>/preview_epoch_*.png` first and repeat the `translate` and `classifier` stages with the chosen checkpoint.

The literal paper GAN (for comparison; it collapses) is:

```powershell
python main.py train --stage all --scenario cell_level --real_mask source --no_composite
```

### C. Baselines

| Baseline | Training data of the classifier |
| :--- | :--- |
| `source_only` | Raw C-NMC cells (the paper's "Source only" row) |
| `composite` | C-NMC cells pasted onto target backgrounds, without the GAN |
| `reinhard` | C-NMC cells colour-normalised to the target training cells |
| `target_supervised` | Labelled target training cells (an upper bound that uses target labels) |

### D. Outputs

| Path | Content |
| :--- | :--- |
| `checkpoints/cyclegan_<scenario>/` | GAN checkpoints and preview grids every 5 epochs |
| `data/processed/translated_source/<scenario>/` | Translated C-NMC images |
| `checkpoints/classifier_<scenario>/results.json` | Test metrics, Table 5 view, per-image predictions |
| `checkpoints/classifier_<scenario>/last_classifier.pth` | Classifier weights (`selected_classifier.pth` with `--source_val`) |
| `checkpoints/baseline_<name>_<scenario>/` | Baseline results |
| `reports/summary.md`, `reports/summary.json` | Final comparison of every run with McNemar tests |
| `logs/run_all_<timestamp>.log` | Log of `python main.py run-all` |

---

## 9. Command-Line Options

### A. `python main.py train`

| Option | Default | Description |
| :--- | :--- | :--- |
| `--stage` | `all` | `gan`, `translate`, `classifier`, `baseline`, or `all` (GAN + translation + classifier) |
| `--scenario` | `cell_level` | `cell_level` (Scenario 1) or `slide_level` (Scenario 2) |
| `--baseline` | none | `source_only`, `composite`, `reinhard`, or `target_supervised`; required by `--stage baseline` |
| `--epochs_gan` | `200` | GAN epochs |
| `--decay_epoch_gan` | `100` | Epoch at which the linear learning-rate decay starts |
| `--epochs_clf` | `50` | Classifier epochs |
| `--batch_size_gan` | `1` | GAN batch size |
| `--batch_size_clf` | `32` | Classifier batch size |
| `--lr_gan` | `0.0001` | GAN learning rate (Adam, $\beta_1 = 0.5$, $\beta_2 = 0.999$) |
| `--lr_clf` | `0.001` | Classifier learning rate (Adam) |
| `--lambda_gan` | `0.5` | Adversarial loss weight |
| `--lambda_cycle` | `10.0` | Cycle-consistency loss weight |
| `--lambda_pixel` | `1.0` | Pixel loss weight |
| `--lambda_identity` | `0.0` | Target identity loss weight (not in the paper) |
| `--real_mask` | `none` | Discriminator inputs: `none` (whole images), `source` (paper Eq. 3), `target` ($t_a \odot t$ vs $s_a \odot s'$) |
| `--no_composite` | off | Translate the black-background source cells, as in the paper |
| `--no_target_balance` | off | Do not pad the minority target class with flipped copies |
| `--source_val` | off | Select the classifier epoch on translated C-NMC `fold_2` instead of using the last epoch |
| `--checkpoint_gan` | latest | GAN checkpoint used for translation |
| `--seed` | `42` | Random seed |
| `--no_amp` | off | Disable mixed precision |
| `--device` | `cuda` if available | `cuda` or `cpu` |

### B. `python main.py run-all`

| Option | Default | Description |
| :--- | :--- | :--- |
| `--scenarios` | `cell_level slide_level` | Scenarios to run, in the given order |
| `--skip_data` | off | Skip C-NMC sampling and ALL-IDB extraction |
| `--skip_tests` | off | Skip the unit tests |
| `--skip_baselines` | off | Skip the baseline classifiers |
| `--epochs_gan`, `--epochs_clf` | `main.py train` defaults | Forwarded to every `python main.py train` call (smoke runs) |
| `--device` | `main.py train` default | Forwarded to every `python main.py train` call |
| `--dry_run` | off | Print the numbered command list and exit |

### C. `python main.py evaluate`

| Option | Default | Description |
| :--- | :--- | :--- |
| `--checkpoints_dir` | `checkpoints` | Where the `results.json` files are read |
| `--output_dir` | `reports` | Where `summary.md` and `summary.json` are written |

---

## 10. Testing

```bash
python -m pytest -v tests
ruff check .
ruff format --check .
```

The 29 tests cover model shapes and losses for every discriminator mode, data loading and target balancing, compositing, the Table 5 metric mapping and the McNemar test, the Scenario 2 grouping and leakage checks, the `python main.py run-all` command list and the `main.py` dispatcher.

---

## 11. References

1. **Baydilli, Y. Y. (2025)**. *Unsupervised attention-guided domain adaptation model for Acute Lymphocytic Leukemia (ALL) diagnosis*. Biomedical Signal Processing and Control, 101, 107159.
2. **Labati, R. D., Piuri, V., & Scotti, F. (2011)**. *All-IDB: The acute lymphoblastic leukemia image database for image processing*. In 18th IEEE International Conference on Image Processing (ICIP), pp. 2045–2048.
3. **Gupta, A., et al. (2019)**. *ISBI 2019 C-NMC Challenge: Classification of Normal vs Malignant Cells in B-ALL White Blood Cancer Microscopic Images*. TCIA.
4. **Alami Mejjati, Y., et al. (2018)**. *Unsupervised Attention-guided Image-to-Image Translation*. NeurIPS 31.
5. **Shrivastava, A., et al. (2017)**. *Learning from Simulated and Unsupervised Images through Adversarial Training*. CVPR.
6. **Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017)**. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. ICCV.
