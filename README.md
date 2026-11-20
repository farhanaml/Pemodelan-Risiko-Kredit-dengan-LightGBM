<h1 align="center">📊 Pemodelan Risiko Kredit dengan LightGBM</h1>
<h3 align="center">Klasifikasi Good Loan vs Bad Loan pada Data Lending Club 2007–2014 (Project-Based Internship ID/X Partners)</h3>

---

## 📑 Daftar Isi

1. [Problem yang Ingin Diselesaikan](#1--problem-yang-ingin-diselesaikan)
2. [Dataset](#2--dataset)
3. [Insight dari Data](#3--insight-dari-data)
4. [Model Final](#4--model-final)
5. [Performa Model](#5--performa-model)
6. [Catatan Penting & Keterbatasan](#6--catatan-penting--keterbatasan)
7. [Proses Pengerjaan](#7--proses-pengerjaan)
8. [Struktur Proyek & Cara Menjalankan](#8--struktur-proyek--cara-menjalankan)
9. [Pengembangan Selanjutnya](#9--pengembangan-selanjutnya)
10. [Referensi](#10--referensi)

---

## 1. 🎯 Problem yang Ingin Diselesaikan

Perusahaan pemberi pinjaman (*lender*) menanggung kerugian setiap kali peminjam gagal bayar. Semakin cepat dan akurat sebuah pinjaman bisa dikenali sebagai **berisiko (bad loan)**, semakin besar peluang perusahaan untuk menekan kerugian, menyiapkan cadangan (provisi), dan memprioritaskan penagihan.

**Pertanyaan bisnis:** *Dapatkah kita memprediksi apakah sebuah pinjaman termasuk Good Loan (0) atau Bad Loan (1) berdasarkan data pinjaman dan peminjam?*

**Tantangan utama:**
- Data **tidak seimbang** (±89% good loan vs ±11% bad loan), sehingga akurasi saja menyesatkan.
- Data finansial memiliki **banyak missing value, kemiringan (skew) ekstrem, dan outlier**.
- Biaya kesalahan tidak simetris: bad loan yang lolos jauh lebih mahal daripada good loan yang salah ditandai.

---

## 2. 🗂️ Dataset

Dataset: `loan_data_2007_2014.csv` (data pinjaman Lending Club periode 2007–2014).

| Item | Nilai |
|---|---|
| Jumlah baris | **466.285** pinjaman |
| Jumlah kolom | **75** (54 numerik, 22 kategorikal, termasuk kolom `Unnamed: 0`) |
| Ukuran di memori | ± 267 MB |
| Data duplikat | Tidak ada |

**Definisi target (`credit_risk`)** dari kolom `loan_status`:

| Label | Arti | Status asli |
|---|---|---|
| `0` | Good Loan | Fully Paid, Current, In Grace Period, *Does not meet the credit policy. Status:Fully Paid* |
| `1` | Bad Loan | Charged Off, Default, Late (16–30 days), Late (31–120 days), *Does not meet the credit policy. Status:Charged Off* |

**Distribusi target (pemetaan final):** Good Loan ±88,8% vs Bad Loan ±11,2%. Proporsi bad loan hanya sekitar **1 dari 9 pinjaman**, jadi kasus ini tergolong *imbalanced*.

**Kualitas data yang ditemukan:**
- **17 kolom 100% kosong** (mis. `annual_inc_joint`, `dti_joint`, `open_acc_6m`, `open_il_6m`, `total_bal_il`), dibuang.
- Kolom `mths_since_last_record` (±87% kosong), `mths_since_last_major_derog` (±79%), dan `mths_since_last_delinq` (±54%) kosong secara sistematis karena kejadiannya memang belum pernah terjadi.
- Kolom identitas/teks (`id`, `member_id`, `url`, `desc`, `title`, `emp_title`, `zip_code`, `Unnamed: 0`) tidak bernilai prediktif.
- Kolom konstan/hampir konstan: `policy_code`, `application_type`, `pymnt_plan`.
- Skew ekstrem: `tot_coll_amt` (skew ≈ 618), `total_rev_hi_lim` (≈ 101), `annual_inc` (≈ 30), `revol_bal` (≈ 19).

---

## 3. 💡 Insight dari Data

### 🥇 Insight 1: Tujuan pinjaman (*purpose*) sangat menentukan tingkat gagal bayar, tetapi segmen paling berisiko justru kecil

**Temuan** (default rate per `purpose`, dari EDA):

| Segmen | Default rate |
|---|---|
| 🔴 `small_business` | **21,0%** |
| 🔴 `renewable_energy` | 15,1% |
| 🔴 `moving` | 15,0% |
| ⚪ `debt_consolidation` (segmen terbesar) | 11,5% |
| 🟢 `major_purchase` | 9,6% |
| 🟢 `car` | 8,8% |
| 🟢 `credit_card` (terendah) | **8,8%** |

- `small_business` punya risiko **±2,4× lipat** dibanding `credit_card`.
- Dari grafik jumlah pinjaman, `debt_consolidation` mendominasi portofolio (jauh melampaui kategori lain), sedangkan `small_business`, `renewable_energy`, dan `moving` volumenya sangat kecil.
- *Catatan: angka di atas dihitung dengan definisi bad loan 4-status pada tahap EDA, sehingga sedikit lebih rendah dibanding definisi final (yang menambahkan dua status "Does not meet the credit policy").*

**✅ Action yang direkomendasikan:**
1. **Risk-based pricing & limit** untuk `small_business`, `renewable_energy`, dan `moving`: bunga lebih tinggi, plafon lebih ketat, atau verifikasi tambahan (dokumen usaha/agunan). Karena volumenya kecil, dampak ke pertumbuhan relatif terbatas, sedangkan pengurangan kerugiannya bisa signifikan.
2. **Campaign akuisisi** pada segmen berisiko rendah (`credit_card`, `car`, `major_purchase`), misalnya penawaran refinancing kartu kredit, agar porsi portofolio berisiko rendah bertambah.
3. **Pantau `debt_consolidation` secara khusus** karena ukurannya besar: selisih kecil pada default rate-nya berdampak besar pada total kerugian.

> ⚠️ Sebelum diterapkan, rekomendasi ini perlu divalidasi dengan margin keuntungan per segmen dan uji coba terbatas (A/B test).

---

### 🥈 Insight 2: Sinyal risiko paling tajam datang dari *perilaku pembayaran setelah pinjaman berjalan*, bukan dari profil saat pengajuan

**Temuan:**
- Profil saat pengajuan hanya memisahkan good/bad secara **tipis**: median DTI bad loan ±18,3 vs good loan ±16,7 (dibaca dari boxplot), dan median lama bekerja kedua kelompok sama (±6 tahun).
- Sebaliknya, model yang memakai variabel perilaku pembayaran (`total_rec_prncp`, `recoveries`, `last_pymnt_amnt`, dll.) mencapai **ROC-AUC 0,9949** dan **recall bad loan 94,9%** pada data uji.
- Pada heatmap korelasi, variabel pembayaran saling berkorelasi kuat, menandakan semuanya merekam proses pelunasan yang sama.
- *Catatan jujur: attribution ke fitur tertentu belum diuji dengan feature importance/ablation, jadi ini kesimpulan yang masuk akal dan perlu dikonfirmasi (lihat [bagian 6](#6--catatan-penting--keterbatasan)).*

**✅ Action yang direkomendasikan:**
1. **Bangun *early-warning system* & prioritas penagihan** untuk portofolio yang sedang berjalan: pinjaman dengan probabilitas bad loan tinggi segera masuk antrean penagihan lebih awal, sehingga peluang pemulihan lebih besar.
2. **Gunakan skor model untuk provisi/cadangan kerugian** pada portofolio aktif.
3. **Untuk keputusan persetujuan pinjaman (origination), latih model terpisah** yang hanya memakai fitur yang tersedia saat pengajuan (tanpa variabel pembayaran), karena model saat ini tidak bisa dipakai pada tahap itu.

---

## 4. 🤖 Model Final

**Algoritma:** LightGBM (Gradient Boosting Decision Tree), dipilih karena cepat pada data tabular besar, menangani fitur kategorikal secara native, dan mendukung penanganan imbalance.

### Fitur
Total **46 fitur** (32 numerik + 14 kategorikal) setelah membuang kolom identitas, teks, kosong total, dan konstan.

- **Kategorikal (14):** `term`, `grade`, `sub_grade`, `emp_length`, `home_ownership`, `verification_status`, `issue_d`, `purpose`, `addr_state`, `earliest_cr_line`, `initial_list_status`, `last_pymnt_d`, `next_pymnt_d`, `last_credit_pull_d`
- **Numerik (32):** `loan_amnt`, `funded_amnt`, `funded_amnt_inv`, `int_rate`, `installment`, `annual_inc`, `dti`, `delinq_2yrs`, `inq_last_6mths`, `mths_since_last_delinq`, `mths_since_last_record`, `mths_since_last_major_derog`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `acc_now_delinq`, `collections_12_mths_ex_med`, `tot_coll_amt`, `tot_cur_bal`, `total_rev_hi_lim`, serta variabel pembayaran: `out_prncp`, `out_prncp_inv`, `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`

### Preprocessing
| Tahap | Perlakuan |
|---|---|
| Pembagian data | Train 60% (279.771) / Validation 20% (93.257) / Test 20% (93.257), *stratified*, `random_state=42` |
| Missing value numerik | Imputasi **median** (dihitung dari train, diterapkan ke val & test, mencegah kebocoran statistik) |
| Missing value kategorikal | Diisi kategori `"Unknown"` |
| Transformasi skew | `log1p` pada `annual_inc`, `revol_bal`, `tot_coll_amt`, `total_rev_hi_lim` |
| Encoding | Tipe `category` (ditangani langsung oleh LightGBM, tanpa one-hot) |
| Imbalance | `scale_pos_weight` = jumlah good / jumlah bad pada train ≈ **7,93** |

### Hyperparameter
```python
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l1": 0.2,
    "lambda_l2": 0.2,
    "scale_pos_weight": scale_pos_weight,  # ≈ 7.93
    "random_state": 42,
    "verbose": -1,
}
```

### Strategi pelatihan
- **Stratified 5-Fold Cross-Validation** pada data train, `num_boost_round` maksimum 5000, *early stopping* 100 ronde.
- **Prediksi Out-of-Fold (OOF)** untuk evaluasi tanpa bias antar-fold.
- **Prediksi test** = rata-rata probabilitas dari kelima model fold (*ensemble*).
- **Optimasi threshold:** dipilih threshold yang memaksimalkan F1 pada kurva Precision–Recall data OOF, bukan memakai 0,5 default.

---

## 5. 📈 Performa Model

### Validasi silang (data train)
| Fold | Best iteration | AUC validasi |
|---|---|---|
| 1 | 67 | 0,9937 |
| 2 | 199 | 0,9942 |
| 3 | 72 | 0,9944 |
| 4 | 160 | 0,9948 |
| 5 | 146 | 0,9942 |

- **OOF ROC-AUC: 0,9936**
- **Threshold terbaik (OOF): 0,5946**, **F1 OOF: 0,9651**

### Data uji (hold-out 93.257 baris, threshold 0,5946)
| Metrik | Nilai |
|---|---|
| Accuracy | 0,9929 |
| Precision (Bad Loan) | 0,9874 |
| Recall (Bad Loan) | 0,9489 |
| F1-score (Bad Loan) | 0,9678 |
| ROC-AUC | 0,9949 |
| Macro F1 / Weighted F1 | 0,9819 / 0,9929 |

**Confusion matrix**

| Aktual \ Prediksi | Good Loan | Bad Loan |
|---|---|---|
| **Good Loan** | 82.694 | 126 |
| **Bad Loan** | 533 | 9.904 |

**Laporan klasifikasi**

| Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bad Loan | 0,9874 | 0,9489 | 0,9678 | 10.437 |
| Good Loan | 0,9936 | 0,9985 | 0,9960 | 82.820 |

**Uji sensitivitas threshold (F1 pada data uji):**

| Threshold | 0,60 | 0,50 | 0,40 | 0,30 | 0,25 | 0,20 |
|---|---|---|---|---|---|---|
| F1 | 0,9678 | 0,9638 | 0,9503 | 0,9241 | 0,8958 | 0,8470 |

Kurva F1 vs threshold pada data uji relatif datar di kisaran 0,4–0,85 (puncak ±0,69 dengan F1 ≈ 0,968), sehingga hasil tidak sensitif terhadap pergeseran kecil threshold. Threshold final tetap **0,5946** (dari OOF), bukan dari data uji.

---

## 6. ⚠️ Catatan Penting & Keterbatasan

**Model ini bersifat *retrospektif*, bukan model persetujuan kredit.**

Dari 46 fitur, **13 fitur baru diketahui setelah pinjaman berjalan** atau bahkan setelah hasil akhirnya terlihat:
`out_prncp`, `out_prncp_inv`, `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`, `last_pymnt_d`, `next_pymnt_d`, `last_credit_pull_d`.

Contohnya, `recoveries` hanya bernilai positif setelah pinjaman *charged off*, dan `next_pymnt_d` kosong pada pinjaman yang sudah selesai/dihapus. Fitur-fitur ini sangat dekat dengan label itu sendiri (*target leakage*; Kaufman et al., 2012). Konsekuensinya:

- Angka AUC ±0,995 dan F1 ±0,97 adalah **batas atas** yang tidak realistis untuk memprediksi gagal bayar saat pengajuan. Bentuk kurva ROC (hampir 95% bad loan terdeteksi dengan false positive nyaris nol) konsisten dengan hal ini.
- Fitur-fitur tersebut sengaja dipertahankan untuk kebutuhan **analisis retrospektif/monitoring portofolio** (lihat Insight 2), bukan untuk keputusan persetujuan pinjaman.
- Pemisahan train/val/test dan threshold bebas dari kebocoran *statistik* antar-split, tetapi **tidak** bebas dari kebocoran *fitur* di atas.

**Keterbatasan lain:**
- Imputasi median diterapkan ke semua fitur numerik, termasuk `mths_since_last_*` (yang kosongnya bermakna "belum pernah terjadi"). Langkah pengisian `-1` di kode berjalan setelah imputasi median sehingga tidak berpengaruh.
- Kolom tanggal (`issue_d`, `earliest_cr_line`, dll.) diperlakukan sebagai kategori berkardinalitas tinggi, belum diubah menjadi fitur numerik/waktu.
- Belum ada analisis *feature importance* atau SHAP, serta belum ada perbandingan dengan model baseline.
- Set validasi 20% dibuat namun belum dipakai terpisah (validasi dilakukan lewat K-Fold pada data train).
- Data berasal dari 2007–2014 dan pembagian dilakukan acak (bukan berdasarkan waktu), sehingga performa pada data periode baru belum teruji.

---

## 7. 🔄 Proses Pengerjaan

1. **Persiapan lingkungan.** Google Colab, pengecekan dependensi, `RANDOM_STATE = 42`.
2. **Pemuatan data.** Membaca CSV, memeriksa dimensi, tipe data, dan ringkasan missing value.
3. **EDA.**
   - Definisi target dan distribusi kelas.
   - Distribusi `loan_amnt`, hubungan `annual_inc` vs `loan_amnt`, DTI per kelas, *purpose* vs risiko, lama bekerja vs risiko.
   - Default rate per tujuan pinjaman, heatmap korelasi.
   - Diagnostik kualitas data: missing value, kardinalitas, varians, duplikat, skewness, rasio outlier (IQR).
4. **Feature engineering & persiapan data.** Membuang fitur identitas/teks/kosong/konstan, memetakan target bisnis, split stratified, imputasi, `log1p`, tipe `category`, dan `scale_pos_weight`.
5. **Manajemen artefak.** Menyimpan X/y (Parquet) dan metadata, mengemasnya ke `credit_risk_artifacts.zip`, lalu memuat kembali untuk memastikan reprodusibilitas.
6. **Pemodelan.** LightGBM dengan Stratified 5-Fold CV, OOF, ensemble prediksi test, dan optimasi threshold berbasis F1.
7. **Penyimpanan model.** Lima model fold (`lgbm_fold1..5.txt`) dan metadata dikemas ke `credit_risk_models.zip`, lalu dimuat kembali.
8. **Evaluasi.** Metrik pada data uji, confusion matrix, kurva ROC, kurva Precision–Recall, kurva F1 vs threshold, dan uji sensitivitas threshold.
9. **Prediksi akhir.** Menyimpan seluruh fitur uji beserta `loan_status` bisnis, `y_true`, `y_proba`, dan `y_pred` ke `predictions/test_set_predictions_full.csv` untuk audit dan integrasi lanjutan.

---

## 8. 📂 Struktur Proyek & Cara Menjalankan

```
pbi-project/
├── realfinal_project_PBI_ID_X.ipynb   # notebook utama
├── README.md
└── (artefak yang dihasilkan notebook)
    ├── artifacts/ , credit_risk_artifacts.zip   # data hasil preprocessing + metadata
    ├── models/ , credit_risk_models.zip         # 5 model LightGBM + threshold
    └── predictions/test_set_predictions_full.csv
```

**Library yang benar-benar dipakai:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`, `joblib`, serta `pyarrow` (untuk Parquet).

**Menjalankan:**
1. Siapkan `loan_data_2007_2014.csv` (dataset Lending Club) dan sesuaikan `data_path` di notebook.
2. Instal dependensi: `pip install pandas numpy matplotlib seaborn scikit-learn lightgbm joblib pyarrow`
3. Jalankan seluruh sel secara berurutan.

---

## 9. 🔮 Pengembangan Selanjutnya

- **Buat model *application-time*:** hapus 13 fitur pasca-pinjaman agar performa mencerminkan kondisi nyata saat persetujuan, lalu bandingkan dengan model saat ini.
- **Perbaiki urutan imputasi** (`-1` untuk `mths_since_last_*` sebelum median) dan ubah kolom tanggal menjadi fitur numerik (usia riwayat kredit, bulan/tahun penerbitan).
- **Validasi berbasis waktu** (latih pada tahun awal, uji pada tahun akhir).
- **Explainability:** feature importance dan SHAP untuk mengonfirmasi Insight 2 secara kuantitatif.
- **Baseline & tuning:** bandingkan dengan Logistic Regression/Random Forest, lalu tuning hyperparameter (Optuna/Bayesian).
- **Threshold berbasis biaya bisnis** (biaya false negative vs false positive), bukan hanya F1.
- **Deployment** sebagai layanan prediksi (API) dan dashboard monitoring.

---

## 10. 📚 Referensi

- Lending Club Loan Data (sumber dataset publik, periode 2007–2014).
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM TKDD.
- Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLOS ONE.
- Dokumentasi LightGBM (parameter `scale_pos_weight`, fitur kategorikal native) dan scikit-learn (metrik evaluasi, `StratifiedKFold`).

---

## 📬 Kontak
Farhan, GitHub: [frhnaml](https://github.com/frhnaml)

🎉 Terima kasih sudah melihat proyek ini!
