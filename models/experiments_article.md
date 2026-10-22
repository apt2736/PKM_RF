# experiments_article

Tabel 2 artikel: baseline aturan (lembar triase IGD RS Siti Khodijah, NEWS2), perbandingan algoritma
(DT, RF, SVC, LightGBM), ablasi fitur, dan ketahanan terhadap galat sensor. Satu split, test set yang sama
(ESI 1-4, n = 12.346), metrik dalam 3 warna.

Jalankan dari repo ini: `cd models && jupyter nbconvert --execute experiments_article.ipynb` (butuh
`lightgbm optuna scikit-learn m2cgen pandas numpy joblib`, dan `cc` untuk mengukur kode C). Keluaran:
`article_table2.csv` di akar repo. Model hasil tuning di-cache di `models/cache/` (hapus untuk mengulang dari nol).
Pengaturan cepat: `N_TRIALS`, `SVC_TUNE_ROWS`, `NOISE_RUNS` lewat variabel lingkungan; `DRY=1` untuk uji asap.

Catatan metodologi yang perlu ada di artikel:
- `waktu_ms` = rata-rata satu prediksi dari kode C hasil m2cgen yang dikompilasi di laptop. BUKAN waktu ESP32.
- SVC dituning pada subsampel (12.000 baris) lalu dilatih penuh; RF dibatasi kedalaman 10 / 150 pohon karena hutan tanpa
  batas menghasilkan file C yang tidak bisa dikompilasi dalam waktu wajar.
- Ablasi: MICE dan scaler di-fit ulang pada fitur yang tersisa.
- Sensitivitas Merah dihitung dari 139 pasien Merah: selisih di bawah ~4 poin ada dalam derau pengambilan sampel.
