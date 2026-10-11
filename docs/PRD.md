# Product Requirements Document (PRD)
## LeukoAdapt: Attention-Guided Unsupervised Domain Adaptation untuk Diagnosis Acute Lymphoblastic Leukemia (ALL)

Dokumen ini menetapkan kebutuhan proyek dalam kondisinya saat ini. Rincian teknis, angka dataset, dan alasan setiap keputusan tercantum di [README.id.md](../README.id.md) (versi Inggris: [README.md](../README.md)); jika terdapat perbedaan, README yang berlaku.

---

### 1. Ringkasan & Latar Belakang

- **Konteks klinis**: ALL didiagnosis antara lain melalui pemeriksaan apusan darah. Classifier yang dilatih pada citra dari satu laboratorium umumnya menurun kinerjanya pada citra dari laboratorium lain karena perbedaan optik mikroskop, kamera, pewarnaan Jenner-Giemsa, dan latar belakang (sel C-NMC tersegmentasi di atas latar hitam, sel ALL-IDB berada di antara sel darah merah).
- **Pendekatan**: sel C-NMC berlabel (domain sumber) ditranslasikan ke gaya ALL-IDB (domain target, diperlakukan tidak berlabel) dengan CycleGAN berpanduan atensi, lalu classifier ResNet34 dilatih pada hasil translasi dan diuji secara buta pada sel ALL-IDB asli.
- **Status repositori**: proyek ini adalah **versi modifikasi** dari metode Baydilli (2025). Diimplementasikan persis seperti tertulis, GAN pada paper mengalami collapse menjadi pemetaan identitas ($s' = s$) pada data ini. Tiga modifikasi membuatnya berfungsi:
  1. setiap sel sumber ditempelkan pada patch latar belakang ALL-IDB asli yang bebas sel (background bank) sebelum translasi;
  2. fusion mask generator adalah cell mask yang diketahui dari sel C-NMC tersegmentasi ditambah tepi 3 px (`--fusion cell`, default), bukan attention mask $A_S$ yang dipelajari seperti pada paper (`--fusion learned`); patch latar belakang tidak diubah;
  3. discriminator membandingkan citra utuh ($s'$ vs $t$), bukan citra bertopeng $s_a \odot s'$ vs $s_a \odot t$ (Pers. 3 paper).

  Bobot loss, optimizer, dan jumlah epoch mengikuti paper; blok spatial attention di dalam generator (Algorithm 2) tetap dipertahankan. Modifikasi kedua ditambahkan karena run penuh pertama yang hanya memakai modifikasi 1 dan 3 mengalami collapse pada epoch 4 (learned mask turun dari 0.12 ke 0.0001), padahal probe singkat 2 sampai 3 epoch tampak stabil; dengan cell mask, run 6 epoch tetap stabil (perubahan di dalam sel sekitar 0.20). GAN versi literal paper tetap tersedia melalui `--fusion learned --real_mask source --no_composite --allow_collapse` sebagai pembanding.
- **Hubungan dengan replikasi murni**: versi salinan paper yang murni dikembangkan secara terpisah oleh dosen pembimbing. Repositori ini **tidak** menjanjikan replikasi eksak maupun pencapaian akurasi 89% dari paper.

---

### 2. Tujuan

1. Menyediakan pipeline UDA yang dapat direproduksi dari data mentah hingga laporan akhir dengan satu perintah (`python main.py run-all`).
2. Menghasilkan translasi C-NMC $\rightarrow$ ALL-IDB yang tidak collapse, dengan bobot loss paper yang tidak diubah.
3. Mengukur kinerja classifier pada dua protokol evaluasi:
   - **Skenario 1** (split tingkat sel, protokol uji paper);
   - **Skenario 2** (split grup slide, bebas kebocoran).
4. Mengukur kontribusi GAN secara terpisah dari kontribusi latar belakang, melalui baseline `composite` dan uji McNemar.
5. Mendokumentasikan setiap penyimpangan dari paper beserta alasannya (README Bagian 5).

**Bukan tujuan**: replikasi eksak paper, pencapaian angka Tabel 5 paper sebagai target, penggunaan klinis, dan pelabelan ulang sel oleh penulis repositori.

---

### 3. Ruang Lingkup

**Termasuk**:
- Sampling C-NMC, ekstraksi sel ALL-IDB1 dari anotasi pakar, kedua split target, dan background bank per skenario.
- Pelatihan GAN, translasi sumber, pelatihan classifier, dan empat baseline.
- Evaluasi akhir lintas metode dengan uji McNemar (`reports/summary.md`).
- Unit test dan test integritas data (termasuk pemeriksaan kebocoran Skenario 2).

**Tidak termasuk**:
- Sel Normal di luar anotasi dataset (rekonstruksi 349 sel Normal paper tidak dilakukan; lihat Bagian 5).
- Split berbasis pasien (ALL-IDB tidak memiliki ID pasien).
- Antarmuka pengguna, deployment, atau validasi klinis.

---

### 4. Definisi Masalah

#### A. Formulasi Domain Adaptation

- **Domain** $\mathcal{D} = \{\mathcal{X}, P(X)\}$: ruang fitur $\mathcal{X}$ dan distribusi marginal $P(X)$.
- **Task** $\mathcal{T} = \{\mathcal{Y}, f(\cdot)\}$: ruang label $\mathcal{Y}$ dan fungsi prediktif $f(\cdot)$.
- **Domain sumber**: C-NMC 2019, $\mathcal{D}_s = \{\mathcal{X}_s, P(X_s)\}$, $\mathcal{T}_s = \{\mathcal{Y}_s, f_s(\cdot)\}$.
- **Domain target**: ALL-IDB1, $\mathcal{D}_t = \{\mathcal{X}_t, P(X_t)\}$, $\mathcal{T}_t = \{\mathcal{Y}_t, f_t(\cdot)\}$.
- **Kondisi UDA**: $P(X_s) \neq P(X_t)$, sedangkan ruang label identik, $\mathcal{Y}_s = \mathcal{Y}_t = \{\text{ALL}, \text{Normal}\}$. Classifier tidak pernah melihat label target. Label target hanya dipakai untuk menyeimbangkan pool target GAN dengan salinan hasil flip (sesuai paper) dan oleh baseline `target_supervised` sebagai batas atas.

#### B. Metode (Ringkas)

- **Fase 1, translasi domain**: sel sumber komposit $s$ diproses generator $G_{S \to T}$ (encoder dengan spatial attention setelah setiap blok konvolusi, Algorithm 2) yang menghasilkan citra konten $G_{S \to T}(s)$, lalu digabung dengan fusion mask $s_a$:
  $$s' = s_a \odot G_{S \to T}(s) + (1 - s_a) \odot s.$$
  Secara default, $s_a$ adalah cell mask sel C-NMC tersegmentasi yang diperlebar 3 px (`--fusion cell`), sehingga generator mengubah gaya sel dan memadukan tepinya, sedangkan patch latar belakang tetap utuh. Dengan `--fusion learned`, $s_a$ adalah attention mask $A_S$ yang dipelajari seperti pada paper (Algorithm 1). $F_{T \to S}$ dan $A_T$ memetakan $s'$ kembali ke $s''$. Fungsi objektif sesuai paper:
  $$\mathcal{L} = 0.5\,\mathcal{L}_{GAN} + 10\,\lVert s - s'' \rVert_1 + 1\,\lVert s - s' \rVert_1,$$
  dengan least-squares adversarial loss, discriminator PatchGAN $D_T$ yang membandingkan citra utuh, history buffer 50 citra, 200 epoch, Adam ($\beta_1 = 0.5$, $\beta_2 = 0.999$), learning rate $10^{-4}$ yang meluruh linear mulai epoch 100. Pelatihan dihentikan dengan error bila translasi collapse selama 3 epoch berturut-turut, kecuali `--allow_collapse` diberikan.
- **Fase 2, classifier**: ResNet34 (bobot ImageNet) dilatih pada 2,000 citra $s'$ dengan label sumbernya; 50 epoch, Adam, learning rate $10^{-3}$, batch size 32. Model dipilih dari epoch terakhir (sesuai paper) atau, dengan `--source_val`, berdasarkan sel C-NMC `fold_2` hasil translasi.
- **Fase 3, evaluasi buta**: classifier dijalankan satu kali pada sel test ALL-IDB asli yang belum pernah dilihat.

---

### 5. Kebutuhan Data

| Domain | Partisi | ALL | Normal | Total | Peran |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Sumber | `source_cnmc/train` (C-NMC `fold_0`) | 1,000 | 1,000 | 2,000 | Input GAN dan data latih classifier |
| Sumber | `source_cnmc/val` (C-NMC `fold_2`) | 500 | 500 | 1,000 | Validasi opsional (`--source_val`) |
| Target, Skenario 1 | `cell_level/train` | 410 | 25 | 435 | Pool target GAN |
| Target, Skenario 1 | `cell_level/test` | 100 | 100 | 200 | Set test buta (protokol paper) |
| Target, Skenario 1 | `cell_level/background` | - | - | 640 | Background bank |
| Target, Skenario 2 | `slide_level/train` | 382 | 95 | 477 | Pool target GAN |
| Target, Skenario 2 | `slide_level/test` | 113 | 25 | 138 | Set test buta pada grup slide yang tidak terlihat |
| Target, Skenario 2 | `slide_level/background` | - | - | 889 | Background bank |

Ketentuan data:
- **D-1. Sumber**: 1,000 ALL + 1,000 HEM diambil dari C-NMC `fold_0` secara round-robin antarsubjek; `fold_2` (subject-disjoint) hanya untuk validasi opsional.
- **D-2. Sel ALL**: 510 blast dari sentroid `.xyc` ALL-IDB1 (49 slide ALL).
- **D-3. Sel Normal**: 125 sel berlabel pakar, yaitu crop ALL-IDB2 `*_0.tif` yang dilokasikan pada slide sumber ALL-IDB1 dengan template matching. ALL-IDB2 hanya dipakai sebagai anotasi, karena seluruh 260 crop-nya berasal dari ALL-IDB1.
- **D-4. Alasan 125, bukan 349**: paper melaporkan 349 sel Normal tanpa menyebutkan sumbernya. ALL-IDB1 hanya menganotasi blast, dan slide sehatnya memuat sedikit leukosit; ALL-IDB tidak menyediakan 349 sel Normal yang berlabel benar. Rekonstruksi lama berbasis detektor pewarnaan terbukti memuat trombosit, fragmen, smudge cell, sel mirip blast, dan panah anotasi (README Bagian 3.C).
- **D-5. Duplikat**: `Im108_0.jpg` (salinan identik `Im093_0.jpg`) tidak digunakan; lima crop Normal ALL-IDB2 yang berulang dihitung satu kali.
- **D-6. Pra-pemrosesan**: crop 257 x 257 berpusat pada sentroid nukleus, di-resize bicubic ke 128 x 128; kedua kelas di-crop dari JPEG ALL-IDB1 yang sama agar classifier tidak dapat membedakan kelas dari format file.
- **D-7. Provenance**: slide sumber, sentroid, nama file ALL-IDB2, skor pencocokan, dan penugasan split dicatat di `data/processed/target_all_idb/metadata_target_cells.json`.
- **D-8. Background bank**: patch diambil hanya dari slide sisi train skenario terkait, jauh dari sel teranotasi, tanpa noda mirip nukleus maupun panah anotasi.

---

### 6. Kebutuhan Fungsional

| ID | Kebutuhan | Perintah / komponen |
| :--- | :--- | :--- |
| F-1 | Seluruh studi dijalankan berurutan (persiapan data, test, kedua skenario beserta baseline, evaluasi) dan berhenti pada langkah pertama yang gagal; log ditulis ke `logs/run_all_<timestamp>.log` | `python main.py run-all` (`--dry_run`, `--scenarios`, `--skip_data`, `--skip_tests`, `--skip_baselines`) |
| F-2 | Sampling C-NMC, ekstraksi ALL-IDB1, kedua split, dan background bank | `python main.py prepare-data` |
| F-3 | Pelatihan satu tahap (`gan`, `translate`, `classifier`, `all`) atau satu baseline untuk skenario yang dipilih | `python main.py train --stage ... --scenario cell_level\|slide_level` |
| F-4 | GAN menyimpan checkpoint dan preview grid (sumber komposit / $s'$ / $s_a$) setiap 5 epoch, mencatat mask mean dan `translation_delta` (rata-rata $\lvert s' - s \rvert$ di dalam fusion mask), mencetak `WARNING` bila translasi collapse ($s' \approx s$ di dalam $s_a$), dan menghentikan pelatihan dengan error bila `translation_delta` di bawah 0.02 selama 3 epoch berturut-turut | `src/training/train_gan.py`; `--allow_collapse` untuk tetap melanjutkan pelatihan |
| F-5 | Translasi dapat memakai checkpoint pilihan (seleksi visual seperti paper) | `--checkpoint_gan` |
| F-6 | Empat baseline: `source_only`, `composite` (komposit tanpa GAN), `reinhard`, `target_supervised` (batas atas) | `--stage baseline --baseline <nama>` |
| F-7 | GAN literal paper tersedia untuk perbandingan | `--fusion learned --real_mask source --no_composite --allow_collapse` |
| F-8 | Skenario 2 selalu memakai `--source_val`, sehingga data target tidak memengaruhi pemilihan model | `run-all` menambahkannya otomatis |
| F-9 | `results.json` per run memuat confusion matrix, akurasi dengan Wilson 95% CI, balanced accuracy, precision, recall, specificity, NPV, F-score, AUROC, tampilan konvensi kolom Tabel 5 paper, dan prediksi per citra | `src/training/train_classifier.py` |
| F-10 | Seluruh run dihimpun ke satu tabel per skenario dengan uji McNemar berpasangan | `python main.py evaluate` $\rightarrow$ `reports/summary.md`, `reports/summary.json` |
| F-11 | Fusion mask generator dapat dipilih: cell mask yang diketahui ditambah tepi 3 px (default) atau attention mask $A_S$ yang dipelajari (paper); opsi ini, `--real_mask`, dan `--lambda_identity` tersedia untuk ablasi | `--fusion cell\|learned` |

Catatan: tidak ada lagi `train.py`, `run_all.py`, `evaluate_all.py`, atau `configs/config.yaml` di root repositori; `main.py` adalah satu-satunya entry point dan hyperparameter diatur melalui opsi command-line.

---

### 7. Kebutuhan Non-Fungsional

- **NF-1. Reprodusibilitas**: seed tetap (default 42); pemilihan patch latar untuk translasi dan baseline `composite` deterministik berdasarkan nama file; hasil pencarian overlap disimpan dalam cache.
- **NF-2. Perangkat keras**: berjalan pada GPU 4 GB (GAN batch size 1, FP16 mixed precision; `--no_amp` untuk menonaktifkan). Satu epoch GAN sekitar dua menit pada RTX 3050 Laptop GPU; run lengkap kedua skenario sekitar 15 jam.
- **NF-3. Integritas data**: `extract_all_idb.py` memunculkan error jika grup slide muncul di kedua sisi split Skenario 2; `tests/test_splits.py` memverifikasinya pada metadata yang dihasilkan.
- **NF-4. Kualitas kode**: `python -m pytest -v tests`, `ruff check .`, dan `ruff format --check .` lulus (30 test); test mencakup bentuk model dan loss untuk setiap mode discriminator dan fusion, cell fusion yang mempertahankan latar belakang, pemuatan data dan penyeimbangan target, compositing, pemetaan metrik Tabel 5 dan uji McNemar, pengelompokan dan pemeriksaan kebocoran Skenario 2, daftar perintah `run-all`, dan dispatcher `main.py`.
- **NF-5. Transparansi**: setiap penyimpangan dari paper didokumentasikan di README Bagian 5; tidak ada label yang diubah berdasarkan penilaian penulis repositori.
- **NF-6. Lingkungan**: Python 3.11, PyTorch dengan CUDA, dependensi di `requirements.txt`.

---

### 8. Protokol Evaluasi & Kriteria Keberhasilan

#### A. Protokol

- **Skenario 1 (protokol paper)**: split acak terstratifikasi atas sel dengan seed; 100 ALL + 100 Normal untuk test. Sel dari satu foto dapat berada di kedua sisi, dan 5 sel fisik yang difoto dua kali muncul di train dan test. Classifier memakai epoch terakhir. Metrik utama: accuracy, balanced accuracy, F-score.
- **Skenario 2 (bebas kebocoran)**: bidang pandang yang tumpang tindih dikelompokkan (15 pasangan, 8 grup); 20 salinan sel fisik dikecualikan; grup test diambil per stratum (kelas, ukuran citra) hingga memuat sekurang-kurangnya 20% sel unik stratum. Background bank hanya dari slide grup train. Metrik utama: **balanced accuracy dan AUROC**, karena set test tidak seimbang (memprediksi ALL untuk semua sel sudah menghasilkan accuracy 81.9%).
- **Referensi paper**: baris Tabel 5 (TN = 91, TP = 87, accuracy 0.8900, F-score 0.8878) ditampilkan sebagai acuan pada Skenario 1, bukan sebagai target. Kolom Precision, Recall, dan Specificity pada Tabel 5 paper salah label (berturut-turut sebenarnya recall, precision, dan NPV; README Bagian 5.B); repositori melaporkan metrik yang benar sekaligus konvensi paper.

#### B. Kriteria Keberhasilan

Kriteria berikut dapat diukur dan tidak menetapkan angka akurasi tertentu:

| ID | Kriteria | Cara verifikasi |
| :--- | :--- | :--- |
| K-1 | GAN default tidak collapse: tidak ada `WARNING` collapse selama pelatihan, pelatihan tidak dihentikan oleh pemeriksaan collapse (`translation_delta`, yaitu rata-rata $\lvert s' - s \rvert$ di dalam fusion mask, tidak berada di bawah 0.02 selama 3 epoch berturut-turut), dan preview menunjukkan sel yang benar-benar diubah gayanya | Log pelatihan, statistik `mask_mean` dan `translation_delta` di checkpoint, preview grid |
| K-2 | Pipeline berjalan end to end untuk kedua skenario dan keempat baseline tanpa langkah gagal | `python main.py run-all` dan log-nya |
| K-3 | Tidak ada kebocoran train/test pada Skenario 2 di tingkat grup slide, sel fisik, dan background bank | `tests/test_splits.py` lulus; error guard di `extract_all_idb.py` tidak terpicu |
| K-4 | Setiap run melaporkan akurasi dengan Wilson 95% CI | `results.json`, `reports/summary.md` |
| K-5 | Skenario 2 dilaporkan dengan balanced accuracy dan AUROC sebagai metrik utama | `reports/summary.md` |
| K-6 | Metode yang diusulkan dibandingkan dengan setiap baseline, termasuk `composite`, melalui uji McNemar, sehingga efek GAN terpisah dari efek latar belakang | Tabel McNemar di `reports/summary.md` |
| K-7 | Selisih kinerja Skenario 1 vs Skenario 2 dilaporkan sebagai estimasi seberapa jauh split tingkat sel membuat metode tampak lebih baik | Tabel kedua skenario di `reports/summary.md` |
| K-8 | Seluruh test dan pemeriksaan Ruff lulus | `python -m pytest`, `ruff check .`, `ruff format --check .` |

Hasil negatif (misalnya GAN tidak berbeda signifikan dari `composite`) tetap merupakan hasil yang sah dan harus dilaporkan apa adanya.

---

### 9. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
| :--- | :--- | :--- |
| GAN collapse ke identitas ($s' = s$), seperti pada formulasi literal paper; learned mask yang tertutup memenuhi seluruh reconstruction loss sekaligus, sehingga collapse dapat terjadi bahkan dengan input komposit | Translasi tidak bermakna; classifier setara dengan `source_only` | Komposit latar ALL-IDB, discriminator citra utuh, dan cell mask sebagai fusion mask (`--fusion cell`) sehingga generator tidak dapat "mematikan" dirinya; deteksi collapse otomatis (`WARNING`) dan penghentian pelatihan bila `translation_delta` di bawah 0.02 selama 3 epoch berturut-turut; probe singkat (2 sampai 3 epoch) dapat menyesatkan, karena run penuh pertama baru collapse pada epoch 4, sehingga preview setiap 5 epoch harus diperiksa; checkpoint dipilih secara visual dengan `--checkpoint_gan` |
| Artefak warna pada epoch awal (misalnya sel kehijauan) | Checkpoint awal dapat menghasilkan translasi yang tidak realistis | Preview grid diperiksa sebelum memilih checkpoint; checkpoint awal tidak dipakai untuk translasi tanpa pemeriksaan visual |
| Sedikit sel Normal (125, bukan 349) | Pool target Skenario 1 hanya memuat 25 Normal; set test Skenario 2 hanya 25 Normal; estimasi berketidakpastian tinggi | Kelas minoritas pool target diseimbangkan dengan salinan flip (sesuai paper); Wilson 95% CI dan uji McNemar; balanced accuracy dan AUROC pada Skenario 2 |
| ALL-IDB tidak memiliki ID pasien | Foto berbeda yang tidak tumpang tindih dari pasien yang sama dapat berada di kedua sisi split | Pengelompokan overlap dan deduplikasi sel fisik; keterbatasan dinyatakan secara eksplisit: Skenario 2 bebas kebocoran di tingkat foto dan sel fisik, tidak dijamin di tingkat pasien |
| Kebocoran pada Skenario 1 (sel satu foto dan 5 sel fisik di kedua sisi) | Kinerja Skenario 1 dapat terlalu optimis | Skenario 1 hanya dipakai sebagai protokol pembanding paper; Skenario 2 menjadi acuan generalisasi; selisih keduanya dilaporkan |
| Waktu GPU panjang (sekitar 15 jam pada GPU 4 GB) | Iterasi eksperimen lambat | Mixed precision, batch size 1; smoke test `--epochs_gan 1 --epochs_clf 1`; `--scenarios` dan `--skip_data` untuk menjalankan sebagian |
| Sekitar 27 sel Normal berukuran kecil dan pucat | Dapat tampak atipikal dan memengaruhi metrik kelas Normal | Tetap dipertahankan karena dataset melabelinya "bukan blast"; tidak ada label yang diubah berdasarkan penilaian; keputusan didokumentasikan |
| Pemilihan checkpoint GAN secara visual bersifat subjektif | Hasil dapat bergantung pada pilihan checkpoint | Default memakai checkpoint terakhir; checkpoint yang dipilih dinyatakan eksplisit dengan `--checkpoint_gan`, lalu tahap `translate` dan `classifier` diulang |
| Kolom Tabel 5 paper salah label | Perbandingan metrik yang keliru | Metrik yang benar dan konvensi paper disimpan berdampingan; `tests/test_metrics.py` mereproduksi baris paper |

---

### 10. Deliverables

1. **Kode**: `main.py` dengan perintah `run-all`, `prepare-data`, `train`, `evaluate`; modul di `src/` (data, model, training, pipeline, utils).
2. **Data terproses**: `data/processed/` (sumber, kedua split target, background bank) beserta `metadata_target_cells.json` dan `slide_overlaps.json`.
3. **Model dan hasil per run**: `checkpoints/cyclegan_<scenario>/` (checkpoint dan preview), `data/processed/translated_source/<scenario>/`, `checkpoints/classifier_<scenario>/results.json` beserta bobot classifier, `checkpoints/baseline_<name>_<scenario>/`.
4. **Laporan akhir**: `reports/summary.md` dan `reports/summary.json` (tabel metrik per skenario, referensi paper, uji McNemar).
5. **Log**: `logs/run_all_<timestamp>.log`.
6. **Test**: `tests/` (unit test dan test integritas data).
7. **Dokumentasi**: `README.md`, `README.id.md` (metodologi, dataset, skenario, penyimpangan dari paper, penggunaan), dan dokumen ini.

---

### 11. Referensi

1. Baydilli, Y. Y. (2025). *Unsupervised attention-guided domain adaptation model for Acute Lymphocytic Leukemia (ALL) diagnosis*. Biomedical Signal Processing and Control, 101, 107159.
2. Labati, R. D., Piuri, V., & Scotti, F. (2011). *All-IDB: The acute lymphoblastic leukemia image database for image processing*. ICIP 2011.
3. Gupta, A., et al. (2019). *ISBI 2019 C-NMC Challenge: Classification of Normal vs Malignant Cells in B-ALL White Blood Cancer Microscopic Images*. TCIA.
