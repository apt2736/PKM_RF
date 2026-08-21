# Catatan buat Aslam

Yuke minta gw ngeliat kenapa akurasi model triase mentok. Ini yang gw ubah, plus
alasan tiap keputusan. Notebook lama di `models/` nggak gw sentuh satu pun, jadi
kalau ada yang nggak setuju sama pendekatan gw, tinggal hapus notebook baru gw
dan semuanya balik kayak semula.

Satu hal yang perlu gw akuin di depan: soal tekanan darah gw awalnya salah baca,
dan itu ada di bagian bawah. Baca sampai situ.

## Angkanya bukan 60%, tapi 32%

Gw hitung dari `holdout_test_confusion_matrix.png` yang udah kecommit di repo.
Diagonalnya 10 + 2083 + 1898 + 993 + 312 = 5296, dari total 16445 baris. Jadi
32,2%.

Masalahnya, kalau model cuma jawab "ESI 3" terus tanpa mikir sama sekali, dia
dapet 43,2%, karena ESI 3 itu 43,2% dari dataset. Model kita kalah sama tebakan
buta.

Gw replikasi pipeline-nya di split bersih 70/15/15 dan dapet 28,6%. Beda tipis
sama 32%, wajar buat split yang beda.

## Penyebab pertama: satu argumen di meta-learner

```python
meta_logreg = LogisticRegression(class_weight='balanced', ...)
```

`class_weight='balanced'` itu yang makan 21 poin akurasi. Dia maksa model
nganggep ESI 1 (208 baris, 0,13% dataset) sama pentingnya sama ESI 3 (71.067
baris, 43%). Efeknya model jadi kebanyakan nebak kelas langka.

Gw hapus satu argumen itu doang, tanpa ganti apa-apa lagi: 28,6% naik jadi
49,5%.

## Penyebab kedua: bukan modelnya, tapi fiturnya

Ini yang bikin gw agak kaget. Gw coba LightGBM multiclass biasa, satu model,
tanpa hierarki, tanpa SMOTE, tanpa 23 fitur rekayasa, tanpa stacking. Hasilnya
50,3%.

Jadi seluruh arsitektur 4 layer plus SMOTE plus meta-learner itu selisihnya 0,3
poin dari LightGBM polos. Kerja bertahun-tahun di 49 notebook, hasilnya sama
dengan 10 baris kode.

Terus gw cek: apa 50% itu batas atasnya? Gw ukur pakai k-NN dengan k=100 di
ruang 15 fitur vital. Local purity-nya 0,481. Artinya kalau kita ambil 100
pasien yang vital sign-nya paling mirip, cuma 48 yang label ESI-nya sama. Nggak
ada model apa pun yang bisa lewat ~48,5% di ruang fitur itu. Overlap-nya nyata.

Tapi overlap itu properti **fitur**, bukan properti tugasnya. Dataset kita punya
200 kolom `cc_*` (chief complaint / keluhan utama), dan pipeline lama pakai satu:
`cc_breathingdifficulty`.

Gw ukur di split yang sama:

| fitur | akurasi 5 kelas |
|---|---|
| selalu jawab ESI 3 | 43,2% |
| 15 fitur vital (yang dipakai sekarang) | 48,8% |
| **keluhan utama doang, tanpa vital sama sekali** | **65,1%** |
| vital + 200 keluhan utama | 67,0% |
| vital + keluhan + konteks kedatangan/riwayat | 69,4% |

Keluhan utama sendirian, tanpa satu pun vital sign, ngalahin semua vital sign
digabung. Selisihnya 16 poin. Ya masuk akal sih kalau dipikir: ESI itu dikasih
perawat triase berdasarkan keluhan pasien, status mental, dan prediksi kebutuhan
sumber daya. Vital sign cuma sebagian kecil dari pertimbangan dia.

Angka 68% juga muncul di tesis orang lain yang pakai dataset Yale 560k yang sama
persis. Jadi ~69% itu sekitar plafon jujur buat 5 kelas. Bukan 90%.

## Yang gw hapus, dan kenapa

**`na.omit` atas 15 kolom.** Ini yang paling besar dampaknya. Pipeline lama
buang tiap baris yang punya minimal 1 NA, jadi 560.486 baris tinggal 165.240.
70,5% dataset kebuang. Gw ukur missingness-nya per kolom:

```
triage_vital_o2       48,4% kosong
spo2_min / spo2_max   41,6% kosong
resp_min / resp_max   38,1% kosong
pulse_min / pulse_max 37,9% kosong
triage_vital_rr       30,4% kosong
triage_vital_hr       29,6% kosong
```

Mayoritas baris kebuang cuma karena SpO2 nggak tercatat. Padahal LightGBM belajar
arah default buat NaN, jadi baris itu tetep bisa dipake. Gw cuma buang baris yang
`esi`-nya kosong. Sisanya 558.029 baris, naik 3,4 kali lipat.

**SMOTE (`numpy_smote`).** Dua alasan. Pertama, dia jalan **setelah**
StandardScaler dan menginterpolasi 10 kolom indikator biner, jadi ngasilin baris
sintetis dengan `is_hypotension = 0,37`. Itu pasien apa. Kedua, dia bukan SMOTE:
nggak ada k-NN neighbourhood, cuma kombinasi konveks dua titik minoritas acak.

**StandardScaler.** Cuma ada buat ngelayanin LogisticRegression dan SMOTE. Tree
nggak butuh scaling. Begitu LR-nya hilang, ini ikut hilang.

**23 fitur rekayasa** (`shock_index`, `is_hypotension`, `rox_index`, `bif`, dst).
Gw ukur: 50,03% dengan, 50,25% tanpa. Nol kontribusi. Ambang klinis kayak
`sbp <= 90` itu hal yang decision tree temuin sendiri, dan nulisnya jadi kolom
biner malah buang resolusi. Kalau lu punya alasan klinis buat mempertahankan,
gw nggak keberatan, tapi angkanya nggak mendukung.

**Hierarki 4 layer + stacking.** Selisih 0,3 poin dari model polos, tapi biaya
kompleksitasnya besar. Dan ada bug di dalamnya: label early stopping layer 3B
itu `np.isin(y_val[m3b_val], [4, 5])`, yang konstan 1 di mask itu. Jadi
`binary_logloss`-nya nggak ada artinya dan early stopping-nya jalan di atas
noise. Label training-nya `(y == 4)`, jadi memang nggak match. Gw udah tes fix-nya
dan selisihnya nggak keukur, tapi tetep salah.

**Nama "OOF".** Notebook-nya namanya `train_oof_logistic_regression_stacking`
tapi nggak ada OOF sama sekali. Meta-learner-nya dilatih di probabilitas
validation set, bukan out-of-fold prediction dari train set.

## Yang gw tambah

**200 kolom keluhan utama.** Ini sumber kenaikan 20 poin. Rata-rata cuma 1,13
flag nyala per kunjungan dan 99,5% kunjungan punya minimal satu, jadi di UI ini
satu dropdown yang bisa dicari, bukan 200 checkbox. Yuke udah bilang nambah input
manual di UI nggak masalah.

**Label 3 kelas, bukan ESI 1 sampai 5.** RED = ESI 1-2, YELLOW = ESI 3,
GREEN = ESI 4-5. Alasannya: alat kita ngirim BLACK/RED/YELLOW/GREEN lewat
`lora_vital_priority_name()`, bukan ESI. Ngelatih head 5 kelas terus di-mapping
belakangan itu buang akurasi gratis. BLACK gw biarin jadi aturan firmware (nggak
ada nadi, nggak ada napas), bukan keputusan statistik.

**Tuning ambang RED.** Ini bagian yang menurut gw paling penting buat proposal.

## Kenapa akurasi bukan metrik yang bener

Akurasi ngasih hukuman sama buat "korban RED dikira GREEN" dan "korban GREEN
dikira YELLOW". Yang pertama orang mati, yang kedua cuma antrian salah.

Doktrin START/SALT ngukurnya pakai undertriage, targetnya di bawah 5%, dan
overtriage ditoleransi sampai 50%. Jadi gw tuning ambang RED di validation set
buat sensitivitas, dan holdout-nya nggak pernah liat pencarian ambangnya.

Hasil di holdout, ambang 0,17:

```
sensitivitas RED        90,95%
undertriage RED ke GREEN 3,25%   doktrin: < 5%      lolos
overtriage non-RED      36,05%   doktrin: <= 50%    lolos
akurasi di ambang       63,05%
akurasi argmax biasa    70,36%
AUC-ROC makro           0,8687

5 kelas (sekunder): akurasi 66,23%, QWK 0,6723, dalam +-1 tingkat 96,88%
```

Trade-off ambangnya:

| ambang | sens RED | undertriage | overtriage | akurasi |
|---|---|---|---|---|
| 0,50 (argmax) | 62,3% | 5,4% gagal | 8,4% | 70,2% |
| 0,25 | 84,0% | 4,3% | 24,8% | 67,1% |
| **0,17** | **91,0%** | **3,3%** | **36,1%** | **63,1%** |
| 0,10 | 96,2% | 1,8% | 52,6% gagal | 56,0% |

Argmax gagal doktrin undertriage, 0,10 gagal doktrin overtriage. 0,17 lolos dua
duanya. Kalau lu mau titik operasi lain, tabelnya ada di output sel 4.

Satu catatan jujur soal angka overtriage: gw awalnya lapor 15% ke Yuke, itu
salah, karena gw cuma ngitung GREEN yang jadi RED. Denominator yang bener itu
semua korban non-RED, soalnya mayoritas yang kenaikan ke RED itu YELLOW. Angka
sebenernya 36,1%. Udah gw perbaiki di `safety()` dan assert-nya pakai yang bener.

## Soal tekanan darah, ini bagian yang gw salah

Gw awalnya buang semua fitur BP dan alesannya gw bilang "hardware nggak punya
sensor tekanan". Itu salah, dan Yuke yang ngoreksi.

Yang gw baca itu komentar di `lora_vital.h:117` dan `tb_regs.h:298` yang bunyinya
"nothing measures pressure". Tapi itu ngomongin **wire STM32 ke ESP32**. STM32-nya
memang nggak punya sensor tekanan, jadi dia ngirim `bp_sys = 0`. Sementara ESP32
punya waveform mentahnya dan punya model buat nurunin BP dari situ:

```
components/triagebox_ml/bp_pipeline.c    27 KB   70 fitur PAT/PTT/morfologi PPG
components/triagebox_ml/lgbm_sbp.c      1,30 MB  regressor SBP
components/triagebox_ml/lgbm_dbp.c      1,21 MB  regressor DBP
```

Dua lapis beda, dan gw nyampur. Maaf.

### Tapi ada bug hidup di situ

`bp_predict_from_raw()` dikompilasi, ada di `CMakeLists.txt`. Cuma satu satunya
yang manggil dia itu `predict_bp_example.c`, dan baris 5 CMakeLists bunyinya:

> `# triage_demo.c dan predict_bp_example.c adalah contoh, deliberately not built.`

Sementara jalur yang beneran jalan di runtime:

```c
// tb_triage.c:47
fold(v->bp_sys, &w->sbp_min, &w->sbp_max, &w->sbp_sum, first);

// tb_triage_model.c:64
in.triage_vital_sbp = (float)w->sbp_sum / w->samples;
```

`v->bp_sys` itu field dari snapshot STM32, yang permanen 0 karena
`TB_FLAG_BP_VALID` clear. Jadi hari ini di hardware, `tb_triage_classify()`
nerima `triage_vital_sbp = 0`, `sbp_min = 0`, `sbp_max = 0`.

Dan itu lebih bahaya daripada BP-nya hilang. Di training ada guard
`np.where(sbp == 0, 1.0, sbp)` buat `shock_index`, jadi shock_index jadi
`hr / 1 = hr`, nilainya sekitar 80, padahal normalnya ~0,5. `sbp_min = 0` juga di
luar rentang dataset sepenuhnya (minimum aslinya 30). `is_hypotension` nyala buat
semua orang. Modelnya nggak nolak, dia nge-skor dengan yakin di atas empat angka
yang nggak pernah ada di data latihnya.

2,5 MB flash kepake buat `lgbm_sbp.c` plus `lgbm_dbp.c`, hasilnya kebuang.

### Argumen yang tetep bertahan

Terlepas dari wiring, ada satu hal yang tetep jadi masalah. `triage_vital_sbp` di
dataset Yale itu hasil pengukuran cuff, ground truth. Di alat, dia hasil estimasi
model. Cuffless BP calibration-free tipikal MAE-nya 8 sampai 15 mmHg, dan review
klinis 2015 sampai 2025 konsisten bilang nggak ada perangkat PTT yang tervalidasi
buat BP absolut tanpa kalibrasi cuff per orang.

Jadi model triase belajar dari fitur yang nggak akan dia terima saat inferensi.
Error dua model numpuk, dan itu nggak keliatan di holdout test mana pun soalnya
holdout-nya pakai cuff juga.

Di skenario bencana lebih parah lagi: kita ketemu tiap korban sekali, nggak ada
baseline buat dikalibrasi, dan PTT butuh ECG plus PPG bersih tanpa motion
artifact. Korban tergeletak di posisi acak, tangan bisa di atas atau di bawah
level jantung, dan itu nggeser bacaan BP secara hidrostatik.

Yang belum gw ukur: berapa nilai BP di model 3 kelas ini, dengan dan tanpa, plus
skenario BP disuntik noise 12 mmHg buat niru error estimator. Yang ketiga itu yang
paling relevan, karena pertanyaannya bukan "apakah BP cuff berguna" tapi "apakah
BP hasil PTT berguna". Bilang aja kalau mau, sekitar 10 menit.

Buat sekarang, angka gw yang paling relevan: BP nyumbang 0,2 poin (69,4% jadi
69,2%) begitu keluhan utama masuk. Diukur di 5 kelas dengan fitur konteks EHR,
jadi belum apple to apple sama model 3 kelas.

## Preprocessing sebelum vs sesudah

| langkah | pipeline lama | punya gw |
|---|---|---|
| sumber | `5v_cleandf.RData` lewat R + rpy2 | `5v_raw.csv` langsung via pandas |
| baris terpakai | 165.240 | 558.029 |
| fitur | 15 mentah + 23 rekayasa = 38 | 212 (3 manual + 9 terukur + 200 keluhan) |
| tekanan darah | 6 dari 38 fitur | dibuang, lihat bagian di atas |
| scaling | StandardScaler 26 kolom | nggak ada |
| resampling | `numpy_smote` per layer | nggak ada |
| missing value | baris dibuang | dibiarin NaN |
| label | `esi` 1-5 | RED/YELLOW/GREEN |
| split | R `stratified_sample` 70/20/10 | sklearn 70/15/15 |

Validasi rentang fisiologis gw nggak tambahin, dan itu sengaja. Gw cek dulu
seluruh 560.486 baris:

```
kolom               min      max   nilai nol  di luar rentang
triage_vital_hr    30,0    280,0          0    0 (0,00%)
triage_vital_rr     8,0     69,0          0    0 (0,00%)
triage_vital_o2    60,0     99,0          0    0 (0,00%)
triage_vital_temp  90,0    106,0          0    0 (0,00%)
pulse_min/max      30,0    280,0          0    0 (0,00%)
spo2_min/max       60,0     99,0          0    0 (0,00%)
```

Datanya udah di-clip di hulu sama pembuat dataset. Suhu satuannya Fahrenheit.
Jadi langkah itu nggak perlu, dan sekarang itu terukur bukan dugaan.

Gw juga cek duplikat: 0 vektor fitur duplikat dari 164.442 baris. Imputasi nggak
gw pake (`models/imputer.ipynb` nggak gw sentuh).

## Ukuran model buat ESP32-S3

27.900 node, sekitar 436 KB kalau disimpen sebagai const array flat di `.rodata`.

Saran gw: **jangan pakai m2cgen buat model ini.** Nested if/else di ukuran segitu
nggak akan ke-compile. Lu udah kena masalah itu sebenernya, `triage_pipeline.c`
sekarang 2,0 MB, `lgbm_sbp.c` 1,3 MB, `lgbm_dbp.c` 1,2 MB. Total 4,5 MB source C
cuma buat model.

Gantinya: simpen `{int16 feature, float threshold, int16 left, int16 right}` per
node terus jalanin pakai interpreter ~30 baris. Ukurannya jadi predictable dan
compile-nya instan.

Trade-off ukuran vs akurasi (3 kelas, semua 200 keluhan):

| konfigurasi | node | flash | akurasi argmax | AUC | sens RED |
|---|---|---|---|---|---|
| leaves=31, n_est=300 | 27.900 | 436 KB | 70,4% | 0,869 | 90,9% |
| leaves=31, n_est=150 | 13.950 | 218 KB | 69,8% | 0,863 | 90,3% |
| leaves=15, n_est=200 | 9.000 | 141 KB | 69,2% | 0,858 | 90,4% |
| leaves=15, n_est=120 | 5.400 | 84 KB | 68,0% | 0,848 | 90,7% |

Sensitivitas RED-nya stabil ~90% di semua konfigurasi, jadi kalau flash-nya mepet
turun ke 141 KB cuma rugi 1,2 poin akurasi.

## File yang gw tambah

```
models/train_disaster_triage_3class.ipynb   9 sel, udah gw jalanin end to end
.gitignore                                  tambah deploy/, plots/*.png, datasets/*.csv
```

Output notebook-nya (nggak dicommit, generated):

```
deploy/disaster_triage_manifest.json   urutan 212 fitur + ambang + metrik holdout
deploy/disaster_triage_3class.pkl      model + ambang
plots/disaster_triage_confusion_matrix.png
```

Manifest itu sengaja gw jadiin satu satunya sumber urutan fitur, biar exporter C
dan form dashboard nggak bisa melenceng dari model yang dilatih. Sel terakhir
notebook isinya assert buat semua angka di atas, jadi kalau ada yang retrain dan
salah satu klaim pecah, sel itu gagal. Nggak diem diem berubah di laporan.

Soal `.gitignore`: `datasets/5v_raw.csv` itu 1,35 GB dan isinya 560 ribu rekam
kunjungan IGD level pasien. Repo ini publik. Tolong jangan pernah kecommit.
`datasets/5v_cleandf.RData` (98,7 MB) udah ada di history sih, tapi itu urusan
lain.

## Yang belum gw kerjain

**Exporter C ke ESP32.** Belum gw bikin karena bentuknya tergantung keputusan yang
masih ngambang: keluhan utama masuk dari mana? Kalau dari dropdown dashboard pas
petugas scan RFID, inferensi jalan di backend dan ESP32 nggak butuh model sama
sekali, exporter-nya nggak kepake. Kalau harus di node, perlu ubah kontrak MQTT
(`hr/spo2/rr/priority` sekarang) lewat `station-change-request.md`. Gw nggak mau
nulis 400 baris yang mungkin langsung kebuang.

**Ukur nilai BP di model 3 kelas.** Lihat bagian BP di atas.

**Distribution shift `pulse_min/max`.** Di dataset ini itu agregat sepanjang
kunjungan IGD, bisa berjam-jam. Di node jadi rolling window beberapa menit.
Distribusinya beda dan gw baru nandain di komentar, belum benerin.

## Yang perlu kalian putusin, bukan gw

Dataset ini IGD Yale, bukan bencana. Label ESI dikasih perawat triase di IGD, dan
kolom `cc_*` isinya termasuk sakit tenggorokan dan sakit gigi, yang nggak muncul
di lokasi bencana. Doktrin bencana yang bener itu START/SALT (bisa jalan?
bernapas? perfusi? ikut perintah?), dan itu berbasis aturan, bukan ML.

Pemetaan ESI 1-2 ke RED itu aproksimasi yang perlu kalian justifikasi eksplisit di
proposal, soalnya penguji yang ngerti triase bencana pasti nanya. Kalau ada
dataset triase bencana asli yang bisa diakses, itu jauh lebih kuat. Tapi dengan
yang ada sekarang, aproksimasi ini pilihan terbaik yang tersedia.

Satu lagi buat PPT PKP2. Jangan taro akurasi sebagai angka utama. Bukan buat
nyembunyiin, tapi karena akurasi memang metrik yang salah di sini dan itu bisa
kalian pertahanin di depan penguji. Taro sensitivitas RED 91%, undertriage 3,3%,
overtriage 36,1%, AUC 0,869 di depan. Akurasi 70,4% taro di bawahnya sebagai
konteks, sama baseline 42,3% buat pembanding.

Terus tabel 48,8% ke 69,4% itu jadiin slide sendiri. Temuan bahwa keluhan utama
nyumbang lebih banyak daripada seluruh vital sign digabung itu hasil yang layak
dipresentasiin, bukan kelemahan yang disembunyiin.
