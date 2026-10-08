# Product Requirements Document (PRD)
## Unsupervised Attention-Guided Domain Adaptation untuk Klasifikasi Acute Lymphocytic Leukemia (ALL)

---

### 1. Ringkasan Eksekutif & Latar Belakang

- **Konteks Klinis**: Leukemia Limfoblastik Akut (ALL) membutuhkan diagnosis dini yang cepat dan presisi melalui apusan darah tepi atau sumsum tulang.
- **Masalah Utama (*Domain Shift*)**: Model deep learning yang dilatih pada satu fasilitas kesehatan sering kali gagal saat diuji pada data dari fasilitas lain karena perbedaan mikroskop, teknik pulasan (*staining*), pencahayaan, dan latar belakang sel darah merah (RBC).
- **Pendekatan Solusi**: Mengembangkan kerangka kerja *Unsupervised Domain Adaptation* (UDA) berbasis *Attention-Guided CycleGAN* untuk mentranslasikan citra sel darah tanpa merusak morfologi leukosit, dilanjutkan dengan klasifikasi ResNet34.

---

### 2. Analogi Sederhana Domain Adaptation

> **Analogi Toko Buah**:
> Bayangkan dua toko sama-sama menjual apel (ruang fitur sama: apel). Toko A hanya menjual apel merah mengkilap dalam keranjang bersih (*C-NMC: sel tersegmentasi tanpa latar*), sedangkan Toko B menjual apel hijau berdebu di atas tumpukan jerami (*ALL-IDB: sel dengan latar belakang RBC kompleks*). Mesin penyortir yang hanya belajar mengenali apel merah bersih di Toko A akan bingung saat ditaruh di Toko B karena terdistraksi oleh jerami dan warna hijau. Domain adaptation bertindak seperti kacamata pintar yang mengubah apel merah Toko A agar memiliki tekstur Toko B tanpa mengubah jenis buah aslinya.

---

### 3. Definisi Masalah Matematis

Secara formal, masalah transfer domain didefinisikan sebagai berikut:

- **Domain $\mathcal{D} = \{\mathcal{X}, P(X)\}$**: Terdiri dari ruang fitur $\mathcal{X}$ dan distribusi probabilitas marginal $P(X)$.
- **Task $\mathcal{T} = \{\mathcal{Y}, f(\cdot)\}$**: Terdiri dari ruang label $\mathcal{Y}$ dan fungsi prediktif objektif $f(\cdot)$.
- **Domain Sumber ($D_s$)**: C-NMC 2019 $\rightarrow D_s = \{X_s, P(X_s)\}, T_s = \{Y_s, f_s(\cdot)\}$.
- **Domain Target ($D_t$)**: ALL-IDB $\rightarrow D_t = \{X_t, P(X_t)\}, T_t = \{Y_t, f_t(\cdot)\}$.
- **Kondisi UDA**: $X_s \neq X_t$ dan $P(X_s) \neq P(X_t)$, namun ruang kelas tugas identik $Y_s = Y_t = \{\text{ALL}, \text{Normal}\}$. Label pada domain target sama sekali tidak digunakan saat pelatihan adaptasi.

---

### 4. Keterbatasan Penelitian Terdahulu & Novelty Riset

#### A. Evaluasi Metode Pendahulu
- **CycleGAN Klasik**: Mentranslasikan seluruh kanvas citra tanpa membedakan objek vital (sel leukemia) dan latar belakang (RBC/plasma), memicu distorsi sitoplasma.
- **Model Attention Kompleks (UAIT, SAGAN, InstaGAN)**: Memerlukan *mask* segmentasi manual tambahan, arsitektur *teacher network* kedua, atau multi-mask yang membebani komputasi.
- **Paper Acuan (Baydilli, 2025)**:
  - Jumlah data target sangat terbatas (659 citra latih vs 2.000 citra sumber).
  - *Discriminator* rentan mengalami *overfitting* pada 659 sampel spesifik.
  - Terdapat ketidakseimbangan kelas pada domain target (410 ALL vs 249 Normal).
  - Pembagian dataset acak di tingkat sel (*cell-level random split*) berisiko tinggi memicu *patient/image-level data leakage*.

#### B. Kebaruan (*Novelty*) & Perbaikan yang Diusulkan
1. **Peniadaan ALL-IDB2 untuk Mencegah Duplikasi Fisik**:
   - ALL-IDB2 merupakan potongan langsung dari ALL-IDB1.
   - Seluruh 859 sel target diekstrak murni dari ALL-IDB1 untuk menjamin tidak ada duplikasi data.
2. **Evaluasi Ketahanan Lintas Pasien (*Patient-Independent Split*)**:
   - Membandingkan evaluasi replikasi acak dengan evaluasi murni lintas preparat/pasien baru.
   - Menguji apakah model adaptasi mampu bertahan tanpa bantuan kebocoran karakteristik preparat yang sama.
3. **Penyelidikan Regularisasi & Few-Shot Domain Adaptation**:
   - Mencegah *discriminator memorization* melalui pembatasan kapasitas buffer dan stabilisasi LSGAN.

---

### 5. Strategi & Spesifikasi Data

#### A. Pembagian Dataset

| Domain | Dataset | Karakteristik Akuisisi | Total Sampel | Latih (*Train*) | Uji (*Test*) | Peran dalam Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Source ($D_s$)** | C-NMC 2019 | Nikon Eclipse-200, $2560 \times 1920$, BMP, tersegmentasi bersih | 2.000 citra | 2.000 citra (1.000 ALL + 1.000 Normal) | 0 | Mengajarkan pengenal fitur kelas dengan label lengkap |
| **Target ($D_t$)** | ALL-IDB1 | Canon PowerShot G5, $2592 \times 1944$, JPG, latar preparat alami | 859 sel | 659 sel (410 ALL + 249 Normal) | 200 sel (100 ALL + 100 Normal) | Memberikan panduan distribusi domain target tanpa label |

#### B. Protokol Ekstraksi Target ALL-IDB1
1. **Ekstraksi 510 Sel Blast (ALL)**:
   - Membaca koordinat sentroid $(x, y)$ ground-truth dari 49 file `.xyc`.
   - Memotong *bounding box* $257 \times 257$ berpusat tepat pada koordinat tersebut.
2. **Ekstraksi 349 Sel Leukosit Sehat (Normal)**:
   - Mengambil citra mikroskop pasien sehat (`*_0.jpg`).
   - Melakukan deteksi nukleus berbasis kontur warna Giemsa (daya serap ungu/violet pekat) untuk memotong 349 sel normal secara presisi.
3. **Standardisasi Citra**:
   - Seluruh citra diubah ukurannya (*resized*) menjadi $128 \times 128$ piksel dengan normalisasi rentang $[-1, 1]$.

---

### 6. Arsitektur Teknis Sistem

```
+-------------------------------------------------------------------------+
|                              Tahap 1: UDA                               |
|                                                                         |
|  [Source: C-NMC] ---> [Generator G_S->T] ---> [Attention Module A_S]    |
|                                                      |                  |
|                                                      v                  |
|  [Target: ALL-IDB] ---> [Masking: s_a * Real] ---> [PatchGAN D_T]       |
|                                ^                     ^                  |
|                                |                     |                  |
|                                +--- s_a * Fake_T ----+                  |
+-------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------+
|                       Tahap 2: Classifier Training                      |
|                                                                         |
|   [Fake_T (Gaya ALL-IDB)] + [Label Asli C-NMC] ---> [ResNet34 (50 Ep)]  |
+-------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------+
|                          Tahap 3: Testing Murni                         |
|                                                                         |
|   [200 Citra Uji ALL-IDB (Unseen)] ---> [ResNet34 Evaluator]            |
|                                          - Akurasi Target: ~89%         |
|                                          - F-Score Target: ~0.8878      |
+-------------------------------------------------------------------------+
```

#### A. Komponen Attention-Guided CycleGAN
- **Spatial Attention Module ($A_S, A_T$)**:
  - Menggabungkan *Average Pooling* dan *Max Pooling* kanal $\rightarrow$ Konvolusi $7 \times 7 \rightarrow$ Sigmoid.
  - Menghasilkan peta atensi spasial $s_a \in [0, 1]$.
  - Citra tertranslasi: $s' = (s_a \odot G(s)) \oplus ((1 - s_a) \odot s)$.
- **Generator**:
  - 3 blok *encoder* konvolusional dengan modul atensi spasial dan *skip connections*.
  - 6 blok residual (*bottleneck*) dengan *Instance Normalization*.
  - 2 blok *decoder* *transposed convolution* dengan *skip concatenation*.
- **Discriminator**:
  - 70x70 PatchGAN menghasilkan peta probabilitas $14 \times 14$.
  - Menerima masukan bertopeng: $s_a \odot t$ (real) dan $s_a \odot s'$ (fake).
  - Memaksa diskriminator hanya menilai domain sel, bukan latar belakang.

#### B. Formulasi Fungsi Loss
- **Least Squares GAN Loss ($\mathcal{L}_{AGAN}$)**:
  $$\mathcal{L}_{AGAN} = \mathbb{E}[(D_T(s_a \odot t) - 1)^2] + \mathbb{E}[(D_T(s_a \odot s'))^2]$$
- **Cycle-Consistency Loss ($\mathcal{L}_{cycle}$)**:
  $$\mathcal{L}_{cycle} = \| s - s'' \|_1$$
- **Pixel Identity Loss ($\mathcal{L}_{pixel}$)**:
  $$\mathcal{L}_{pixel} = \| s - s' \|_1$$
- **Total Generator Loss**:
  $$\mathcal{L}_{total} = \lambda_{gan}\mathcal{L}_{AGAN} + \lambda_{cycle}\mathcal{L}_{cycle} + \lambda_{pixel}\mathcal{L}_{pixel}$$
  *(Bobot default: $\lambda_{gan} = 0.5, \lambda_{cycle} = 10.0, \lambda_{pixel} = 1.0$)*.

---

### 7. Perbandingan Tiga Tahapan Pelatihan

| Parameter | Tahap 1: Domain Adaptation | Tahap 2: Classifier Training | Tahap 3: Testing Akhir |
| :--- | :--- | :--- | :--- |
| **Input Data** | 2.000 C-NMC (unlabeled) + 659 ALL-IDB (unlabeled) | 2.000 Citra C-NMC tertranslasi ($s'$) + Label Asli | 200 Citra Asli ALL-IDB target (100 ALL + 100 Normal) |
| **Model yang Dilatih** | Generator ($G, F$) & Discriminator ($D_T$) | ResNet34 Classifier | Tidak ada pelatihan (hanya inferensi) |
| **Fungsi Objektif** | Adversarial + Cycle + Pixel Identity Loss | Cross-Entropy Loss | Tidak ada (evaluasi metrik uji) |
| **Epoch & Optimizer** | 200 Epoch, Adam ($\beta_1=0.5, \beta_2=0.999$), LR $10^{-4}$ | 50 Epoch, Adam, LR $10^{-3}$, Batch 32 | Single-pass forward feed |
| **Tujuan Utama** | Menghilangkan celah domain tanpa merusak morfologi sel | Memetakan fitur sel tertranslasi ke label biner | Mengukur daya generalisasi pada domain target murni |

---

### 8. Target Kinerja & Metrik Evaluasi

#### A. Kinerja Kualitas Translasi Citra (GAN)
- **Structural Similarity Index (SSIM)**: $\ge 0.54$ (Source-to-Target).
- **Peak Signal-to-Noise Ratio (PSNR)**: $\ge 18.41\text{ dB}$.
- **Fréchet Inception Distance (FID)**: $\le 107.91$.

#### B. Kinerja Klasifikasi ResNet34 (Baseline Benchmark)
- **Akurasi Uji**: Target minimal $89.00\%$ pada 200 sampel uji.
- **F-Score**: Target minimal $0.8878$.
- **Sensitivitas / Recall**: Target minimal $0.9063$.
- **Spesifisitas**: Target minimal $0.8750$.
- **Presisi**: Target minimal $0.8700$.

---

### 9. Rencana Kerja Kronologis (Step-by-Step Roadmap)

1. **Persiapan Data & Ekstraksi**:
   - Menjalankan `extract_all_idb.py` untuk mengumpulkan 859 sel murni dari ALL-IDB1.
   - Menjalankan `sample_cnmc.py` untuk mengambil subset 2.000 sel seimbang dari C-NMC fold_0.
2. **Replikasi Baseline Skenario 1 (Cell-Level Split)**:
   - Melatih Attention-CycleGAN selama 200 epoch menggunakan konfigurasi `configs/config.yaml`.
   - Mengenerasi citra translasi domain sumber ke gaya target.
   - Melatih ResNet34 selama 50 epoch pada citra translasi dan mengevaluasi pada 200 data uji.
3. **Pengujian Skenario 2 (Patient-Independent Split)**:
   - Mengelompokkan citra ALL-IDB1 berdasarkan ID file preparat utuh.
   - Memisahkan 20 preparat khusus untuk uji, melatih ulang model, dan mengukur disparitas penurunan performa.
4. **Implementasi Metode Peningkatan Generalisasi**:
   - Mengintegrasikan teknik regularisasi discriminative atau pseudo-labeling terarah.
   - Membandingkan hasil akhir terhadap baseline untuk dokumentasi publikasi ilmiah.

---

### 10. Manajemen Risiko & Alokasi Perangkat Keras

- **Keterbatasan VRAM GPU (RTX 3050 4GB)**:
  - Pelatihan GAN beresiko kehabisan memori (*CUDA Out of Memory*).
  - Mitigasi: Mengaktifkan *Automatic Mixed Precision* (AMP FP16), ukuran batch = 1, dan pembersihan *cache* berkala.
- **Ketidakseimbangan Kelas Target**:
  - Target latih terdiri dari 410 ALL vs 249 Normal.
  - Mitigasi: Menerapkan *horizontal/vertical flip augmentasi* pada kelas minoritas selama iterasi pelatihan GAN.
- **Discriminator Mode Collapse**:
  - Diskriminator dapat menghafal pola latar belakang spesifik dari 659 sampel.
  - Mitigasi: Menggunakan *Image History Pool* berkapasitas 50 citra dan memperlambat laju belajar diskriminator dengan membagi loss menjadi 50%.
