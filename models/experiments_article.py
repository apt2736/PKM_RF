# %% [markdown]
# # Eksperimen untuk Tabel 2 artikel
#
# Satu notebook, satu pembagian data. Semua baris dievaluasi pada **test set yang
# sama** (ESI 1-4, n = 12.346) dalam 3 warna: ESI1 = Merah, ESI2 = Kuning,
# ESI3-4 = Hijau.
#
# Kolom keluaran (`article_table2.csv`):
# `skenario, sens_merah, bal_acc, under_triage, over_triage, ukuran_kb, waktu_ms`
# (+ kolom tambahan rentang persentil untuk uji derau).
#
# * `sens_merah`  = recall kelas Merah
# * `bal_acc`     = rata-rata makro 3 warna dari (recall + spesifisitas) / 2 -- rumus yang
#   sama dengan `compute_translated_color_metrics` di notebook utama
#   (nilai acuan model utama: 0,9017 pada run Kaggle Aslam)
# * `under_triage` = Merah yang diprediksi Kuning/Hijau / total Merah
# * `over_triage`  = 1 - presisi Merah
# * `ukuran_kb`    = ukuran file .c hasil m2cgen (KB); kosong untuk baseline aturan
# * `waktu_ms`     = rata-rata satu prediksi (10.000 panggilan, kode C hasil m2cgen
#   dikompilasi di laptop). BUKAN waktu ESP32: waktu di board diukur dengan
#   perintah `bench` di firmware.

# %%
import json, os, subprocess, sys, tempfile, time, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
import optuna, m2cgen as m2c
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = ".." if os.path.basename(os.getcwd()) == "models" else "."
N_TRIALS = int(os.environ.get("N_TRIALS", 20))
SVC_TUNE_ROWS = int(os.environ.get("SVC_TUNE_ROWS", 12000))
NOISE_RUNS = int(os.environ.get("NOISE_RUNS", 100))
OUT = os.path.join(ROOT, "article_table2.csv")

# %% [markdown]
# ## 1. Data dan pembagian (sama persis dengan model utama)
#
# Stratified 70/15/15 pada SEMUA baris (termasuk ESI 5), `random_state` 42 dari
# `config/triage_conf.json`; baris ESI 5 baru dibuang sesudahnya. MICE dan
# StandardScaler di-fit hanya pada data latih (termasuk baris ESI 5 di data latih,
# persis seperti notebook utama). 6 fitur, tanpa DBP.

# %%
cfg = json.load(open(f"{ROOT}/config/triage_conf.json"))["training"]
test_size, val_size, seed = cfg["test_size"], cfg["val_size"], cfg["random_state"]
df = pd.read_csv(f"{ROOT}/datasets/fedmml_ed_triage_dataset.csv")
df["sex_encoded"] = df["sex"].astype(str).str.strip().str.upper().map(
    {"M": 1.0, "MALE": 1.0, "F": 0.0, "FEMALE": 0.0})
FEATS = ["age", "sex_encoded", "systolic_bp", "heart_rate", "respiratory_rate", "spo2"]
df = df.dropna(subset=["esi_level"]).copy()
df["esi_level"] = df["esi_level"].astype(int)
X_all, y_all = df[FEATS].values, df["esi_level"].values

X_rem, X_test_raw, y_rem, y_test = train_test_split(
    X_all, y_all, test_size=test_size, stratify=y_all, random_state=seed)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_rem, y_rem, test_size=val_size / (1 - test_size), stratify=y_rem,
    random_state=seed + 1)

mice = IterativeImputer(max_iter=10, random_state=42)
X_train_imp = mice.fit_transform(X_train_raw)
X_val_imp, X_test_imp = mice.transform(X_val_raw), mice.transform(X_test_raw)
CONT = [0, 2, 3, 4, 5]
scaler = StandardScaler().fit(X_train_imp[:, CONT])
def scale(Xi):
    X = Xi.copy(); X[:, CONT] = scaler.transform(Xi[:, CONT]); return X
X_train, X_val, X_test = scale(X_train_imp), scale(X_val_imp), scale(X_test_imp)

# ESI 5 dibuang: train/val/test 4 kelas
m_tr, m_va, m_te = y_train <= 4, y_val <= 4, y_test <= 4
Xtr, ytr = X_train[m_tr], y_train[m_tr]
Xva, yva = X_val[m_va], y_val[m_va]
Xte, yte = X_test[m_te], y_test[m_te]
Xte_imp_raw = X_test_imp[m_te]          # satuan asli, setelah MICE (untuk baseline aturan)
Xte_nan_raw = X_test_raw[m_te]          # satuan asli, NaN dipertahankan (untuk uji derau)
print(f"train {len(ytr):,}  val {len(yva):,}  test {len(yte):,}")
assert len(yte) == 12346, "test set harus 12.346 baris seperti model utama"

# %% [markdown]
# ## 2. Metrik (3 warna)

# %%
def to_color(esi):                      # 0 Merah, 1 Kuning, 2 Hijau
    return np.where(esi == 1, 0, np.where(esi == 2, 1, 2))

def metrics(y_true_color, y_pred_color):
    per = []
    for c in range(3):
        t, p = y_true_color == c, y_pred_color == c
        tp, fn = (t & p).sum(), (t & ~p).sum()
        fp, tn = (~t & p).sum(), (~t & ~p).sum()
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        per.append((rec, spec, tp, fp))
    rec_r, _, tp_r, fp_r = per[0]
    return dict(
        sens_merah=rec_r,
        bal_acc=float(np.mean([(r + s) / 2 for r, s, *_ in per])),
        under_triage=1 - rec_r,
        over_triage=(fp_r / (tp_r + fp_r)) if (tp_r + fp_r) else 0.0,
    )

yte_color = to_color(yte)
ROWS = []
def add(name, m, ukuran_kb="", waktu_ms="", **extra):
    ROWS.append(dict(skenario=name, **m, ukuran_kb=ukuran_kb, waktu_ms=waktu_ms, **extra))
    print(f"{name:55s} sens_merah={m['sens_merah']:.4f} bal_acc={m['bal_acc']:.4f} "
          f"under={m['under_triage']:.4f} over={m['over_triage']:.4f}")

# %% [markdown]
# ## A. Baseline berbasis aturan
#
# **A1 -- Lembar Triase Pasien IGD RS Siti Khodijah Sepanjang** (`docs/lembar_triage.pdf`).
# Hanya RR dan nadi yang ada di dataset; Airway, CRT, GCS, dan nyeri tidak
# tersedia, jadi tidak dipakai (keterbatasan baseline). Batas tepat 20 masuk Hijau.
# Lembar ini tidak memakai SpO2 maupun tekanan darah, sedangkan model TriageBox
# memakai keduanya. Nilai hilang: hasil MICE yang sama dengan model.

# %%
rr, hr = Xte_imp_raw[:, 4], Xte_imp_raw[:, 3]
rr_cat = np.where(rr > 30, 0, np.where(rr > 20, 1, 2))
hr_cat = np.where(hr > 120, 0, np.where(hr >= 100, 1, 2))
a1 = np.minimum(rr_cat, hr_cat)          # kategori terberat: Merah(0) < Kuning(1) < Hijau(2)
add("A1 Lembar triase IGD RS Siti Khodijah (RR + nadi)", metrics(yte_color, a1))

# %% [markdown]
# **A2 -- NEWS2 termodifikasi** (Royal College of Physicians, 2017), hanya parameter
# yang tersedia; suhu, kesadaran, dan oksigen tambahan diberi skor 0.

# %%
def news2_color(Xr):
    sbp, hr_, rr_, spo2 = Xr[:, 2], Xr[:, 3], Xr[:, 4], Xr[:, 5]
    s_rr = np.select([rr_ <= 8, rr_ <= 11, rr_ <= 20, rr_ <= 24], [3, 1, 0, 2], 3)
    s_o2 = np.select([spo2 <= 91, spo2 <= 93, spo2 <= 95], [3, 2, 1], 0)
    s_sbp = np.select([sbp <= 90, sbp <= 100, sbp <= 110, sbp <= 219], [3, 2, 1, 0], 3)
    s_hr = np.select([hr_ <= 40, hr_ <= 50, hr_ <= 90, hr_ <= 110, hr_ <= 130],
                     [3, 1, 0, 1, 2], 3)
    total = s_rr + s_o2 + s_sbp + s_hr
    any3 = (np.stack([s_rr, s_o2, s_sbp, s_hr]) == 3).any(axis=0)
    return np.where(total >= 7, 0, np.where((total >= 5) | any3, 1, 2))
add("A2 NEWS2 termodifikasi (RR, SpO2, SBP, HR)", metrics(yte_color, news2_color(Xte_imp_raw)))

# %% [markdown]
# ## B. Perbandingan algoritma
#
# Fitur dan split sama, `class_weight='balanced'`, Optuna 20 trial per model
# (objektif: balanced accuracy validasi, 4 kelas). **Penyimpangan yang disengaja:**
# tuning SVC memakai subsampel latih (`SVC_TUNE_ROWS`) karena RBF-SVC pada 57 ribu baris
# terlalu lambat untuk 20 trial; model akhir dilatih pada seluruh data latih.

# %%
def tune(make, space, X, y, n_trials=N_TRIALS):
    def obj(trial):
        m = make(trial)
        m.fit(X, y)
        return balanced_accuracy_score(yva, m.predict(Xva))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(obj, n_trials=n_trials)
    return st

MODELS = {}
t0 = time.time()
import joblib
os.makedirs("cache", exist_ok=True)
def cached(key, build):
    """Fit once, reuse on a re-run: tuning RF and SVC takes the better part of the run."""
    path = f"cache/{key}.joblib"
    if os.path.exists(path) and not os.environ.get("DRY"):
        return joblib.load(path)
    obj = build()
    if not os.environ.get("DRY"):
        joblib.dump(obj, path)
    return obj

def build_dt():
    st = tune(lambda t: DecisionTreeClassifier(class_weight="balanced", random_state=42,
              max_depth=t.suggest_int("max_depth", 3, 20),
              min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 200)), None, Xtr, ytr)
    print("DT best:", st.best_params)
    return DecisionTreeClassifier(class_weight="balanced", random_state=42, **st.best_params).fit(Xtr, ytr)

def build_rf():
    # depth capped at 10 / 150 trees: an unbounded forest exports to a C file that
    # cannot be compiled in reasonable time, which is itself a finding about RF on an MCU
    st = tune(lambda t: RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1,
              n_estimators=t.suggest_int("n_estimators", 50, 150),
              max_depth=t.suggest_int("max_depth", 4, 10),
              min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 100)), None, Xtr, ytr)
    print("RF best:", st.best_params)
    return RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, **st.best_params).fit(Xtr, ytr)

def build_svc():
    sub = np.random.default_rng(42).choice(len(ytr), size=min(SVC_TUNE_ROWS, len(ytr)), replace=False)
    st = tune(lambda t: SVC(kernel="rbf", class_weight="balanced", probability=False,
              C=t.suggest_float("C", 0.1, 100, log=True),
              gamma=t.suggest_float("gamma", 1e-3, 1.0, log=True)), None, Xtr[sub], ytr[sub])
    print("SVC best:", st.best_params)
    rows = sub if os.environ.get("DRY") else slice(None)     # DRY: smoke test only
    return SVC(kernel="rbf", class_weight="balanced", probability=False, **st.best_params).fit(Xtr[rows], ytr[rows])

MODELS["Decision Tree"] = cached("dt", build_dt)
MODELS["Random Forest"] = cached("rf", build_rf)
MODELS["SVC (RBF)"] = cached("svc", build_svc)
# LightGBM = konfigurasi model utama (hyperparameter dari run Kaggle/notebook utama)
MAIN_PARAMS = {'objective': 'multiclass', 'num_class': 4, 'metric': 'multi_logloss',
               'class_weight': 'balanced', 'n_estimators': 300, 'verbosity': -1,
               'random_state': 42, 'learning_rate': 0.021656987328212585, 'num_leaves': 62,
               'max_depth': 3, 'min_child_samples': 76, 'feature_fraction': 0.7440058863611535,
               'bagging_fraction': 0.6521141054801364, 'bagging_freq': 6,
               'reg_alpha': 0.10082276625455867, 'reg_lambda': 0.39314382315911356}
def fit_lgbm(X, y, Xv, yv, params=MAIN_PARAMS):
    m = lgb.LGBMClassifier(**params)
    m.fit(X, y - 1, eval_set=[(Xv, yv - 1)], callbacks=[lgb.early_stopping(30, verbose=False)])
    return m
lgbm = fit_lgbm(Xtr, ytr, Xva, yva)
MODELS["LightGBM (model utama)"] = lgbm
print(f"training+tuning done in {time.time()-t0:.0f}s")

def predict_esi(name, m, X):
    p = m.predict(X)
    return p + 1 if name.startswith("LightGBM") else p

# %% [markdown]
# ### Ekspor C (m2cgen), ukuran, dan waktu inferensi

# %%
HARNESS = r'''
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
%(code)s
int main(int argc, char **argv) {
    int n = atoi(argv[1]); int rows = atoi(argv[2]);
    static double X[20000][6]; static double out[8];
    FILE *f = fopen(argv[3], "r");
    for (int i = 0; i < rows; i++) for (int j = 0; j < 6; j++) fscanf(f, "%%lf", &X[i][j]);
    fclose(f);
    struct timespec a, b; double sink = 0;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (int i = 0; i < n; i++) { score(X[i %% rows], out); sink += out[0]; }
    clock_gettime(CLOCK_MONOTONIC, &b);
    double ms = ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec)) / 1e6 / n;
    printf("%%.6f %%f\n", ms, sink);
    for (int i = 0; i < rows && i < 200; i++) {      /* predicted class, for the agreement check */
        score(X[i], out); int best = 0;
        for (int k = 1; k < %(ncls)d; k++) if (out[k] > out[best]) best = k;
        printf("%%d\n", best);
    }
    return 0;
}
'''
def export_svc_c(m):
    """RBF-SVC as arrays + a loop (libsvm one-vs-one voting). m2cgen builds one
    expression as deep as the number of support vectors and overflows the stack."""
    sv, dc, ns = m.support_vectors_, m.dual_coef_, m.n_support_
    ncls, start = len(ns), np.concatenate([[0], np.cumsum(ns)])
    f = lambda a: ", ".join(f"{v:.12g}" for v in np.ravel(a))
    pairs = [(i, j) for i in range(ncls) for j in range(i + 1, ncls)]
    body = []
    for p, (i, j) in enumerate(pairs):
        body.append(f"    {{ double d = {m.intercept_[p]:.12g};")
        body.append(f"      for (int s = {start[i]}; s < {start[i+1]}; s++) d += DC[{j-1}][s] * k[s];")
        body.append(f"      for (int s = {start[j]}; s < {start[j+1]}; s++) d += DC[{i}][s] * k[s];")
        body.append(f"      out[d > 0 ? {i} : {j}] += 1.0; }}")
    return f"""#define NSV {sv.shape[0]}
static const double SV[NSV][6] = {{ {", ".join("{" + f(r) + "}" for r in sv)} }};
static const double DC[{ncls-1}][NSV] = {{ {", ".join("{" + f(r) + "}" for r in dc)} }};
void score(double *x, double *out) {{
    static double k[NSV];
    for (int c = 0; c < {ncls}; c++) out[c] = 0.0;
    for (int s = 0; s < NSV; s++) {{
        double d2 = 0.0;
        for (int j = 0; j < 6; j++) {{ double t = x[j] - SV[s][j]; d2 += t * t; }}
        k[s] = exp(-{m.gamma if isinstance(m.gamma, float) else m._gamma:.12g} * d2);
    }}
{chr(10).join(body)}
}}
"""

def c_size_and_time(name, model, n_calls=10000, rows=500):
    try:
        code = export_svc_c(model) if isinstance(model, SVC) else m2c.export_to_c(model)
    except Exception as e:                      # report and carry on rather than lose the run
        print(name, "export failed:", type(e).__name__, str(e)[:120]); return float("nan"), "", None
    kb = len(code.encode()) / 1024
    d = tempfile.mkdtemp()
    src, exe, data = f"{d}/m.c", f"{d}/m", f"{d}/x.txt"
    np.savetxt(data, Xte[:rows], fmt="%.9f")
    ncls = 4
    open(src, "w").write(HARNESS % dict(code=code, ncls=ncls))
    try:
        r = subprocess.run(["cc", "-O1", "-w", "-o", exe, src, "-lm"], capture_output=True, text=True, timeout=420)
    except subprocess.TimeoutExpired:
        print(f"   {name}: C file is {kb:.0f} KB and did not compile in 420 s -- size reported, no timing")
        return kb, "", None
    if r.returncode != 0:
        print(name, "compile failed:", r.stderr[:300]); return kb, "", None
    t = subprocess.run([exe, str(n_calls), str(rows), data], capture_output=True, text=True, timeout=1800)
    lines = t.stdout.split("\n")
    c_pred = np.array([int(v) + 1 for v in lines[1:201] if v.strip()])
    ref = predict_esi(name, model, Xte[:len(c_pred)])
    print(f"   {name}: C vs sklearn agree on {100*np.mean(c_pred == ref):.1f}% of {len(c_pred)} rows")
    return kb, float(lines[0].split()[0]), code

SIZES = {}
for name, m in MODELS.items():
    kb, ms, code = c_size_and_time(name, m)
    SIZES[name] = (round(kb, 1), ms)
    print(f"{name:25s} .c {kb:10.1f} KB   {ms} ms/prediksi (laptop, C -O2)")

# %%
for name, m in MODELS.items():
    pred = predict_esi(name, m, Xte)
    add(f"B {name}", metrics(yte_color, to_color(pred)),
        ukuran_kb=SIZES[name][0], waktu_ms=SIZES[name][1])

# %% [markdown]
# ## C. Ablasi fitur (LightGBM, hyperparameter sama)
#
# MICE dan scaler di-fit ulang pada fitur yang tersisa (supaya fitur yang dibuang
# tidak ikut bocor lewat imputasi fitur lain).

# %%
def ablate(drop_idx):
    keep = [i for i in range(6) if i != drop_idx]
    imp = IterativeImputer(max_iter=10, random_state=42)
    Xtr_i = imp.fit_transform(X_train_raw[:, keep])
    Xva_i, Xte_i = imp.transform(X_val_raw[:, keep]), imp.transform(X_test_raw[:, keep])
    cont = [k for k, i in enumerate(keep) if i in CONT]
    sc = StandardScaler().fit(Xtr_i[:, cont])
    def f(Xi):
        X = Xi.copy(); X[:, cont] = sc.transform(Xi[:, cont]); return X
    a, b, c = f(Xtr_i)[m_tr], f(Xva_i)[m_va], f(Xte_i)[m_te]
    m = fit_lgbm(a, ytr, b, yva)
    return metrics(yte_color, to_color(m.predict(c) + 1))

full = metrics(yte_color, to_color(lgbm.predict(Xte) + 1))
add("C LightGBM lengkap (6 fitur)", full)
for idx, label in [(2, "SBP"), (4, "RR"), (5, "SpO2")]:
    m = ablate(idx)
    add(f"C tanpa {label}", m, delta_sens_merah=round(m["sens_merah"] - full["sens_merah"], 4),
        delta_bal_acc=round(m["bal_acc"] - full["bal_acc"], 4))

imp = pd.Series(lgbm.booster_.feature_importance(importance_type="gain"), index=FEATS)
print("\nFeature importance (gain, % dari total):")
print((100 * imp / imp.sum()).round(1).sort_values(ascending=False).to_string())

# %% [markdown]
# ## D. Ketahanan terhadap galat sensor
#
# Model lengkap yang sudah terlatih, tidak dilatih ulang. Derau ditambahkan dalam
# satuan asli pada data uji (nilai hilang tetap hilang), lalu pipeline normal:
# imputer -> scaler -> model. Setiap skenario diulang `NOISE_RUNS` kali dengan seed
# berbeda; dilaporkan rata-rata dan persentil 2,5-97,5.

# %%
def run_pipeline(X_raw_noisy):
    Xi = mice.transform(X_raw_noisy)
    return to_color(lgbm.predict(scale(Xi)) + 1)

clean_pred = run_pipeline(Xte_nan_raw)
def noisy(rng, hr=None, spo2=None, sbp=None):
    X = Xte_nan_raw.copy()
    if hr:   X[:, 3] += rng.normal(hr[0], hr[1], len(X))
    if spo2: X[:, 5] = np.minimum(X[:, 5] + rng.normal(spo2[0], spo2[1], len(X)), 100)
    if sbp:  X[:, 2] += rng.normal(sbp[0], sbp[1], len(X))
    return X

def repeat(name, make):
    res = []
    for s in range(NOISE_RUNS):
        pred = run_pipeline(make(np.random.default_rng(1000 + s)))
        m = metrics(yte_color, pred); m["berubah_pct"] = 100 * float((pred != clean_pred).mean())
        res.append(m)
    d = pd.DataFrame(res); lo, hi = d.quantile(0.025), d.quantile(0.975)
    mean = d.mean()
    add(name, {k: float(mean[k]) for k in ["sens_merah", "bal_acc", "under_triage", "over_triage"]},
        sens_merah_p2_5=round(float(lo.sens_merah), 4), sens_merah_p97_5=round(float(hi.sens_merah), 4),
        bal_acc_p2_5=round(float(lo.bal_acc), 4), bal_acc_p97_5=round(float(hi.bal_acc), 4),
        kategori_berubah_pct=round(float(mean.berubah_pct), 2),
        kategori_berubah_p2_5=round(float(lo.berubah_pct), 2), kategori_berubah_p97_5=round(float(hi.berubah_pct), 2))

add("D0 tanpa derau (model lengkap)", metrics(yte_color, clean_pred), kategori_berubah_pct=0.0)
repeat("D1 derau HR N(0,16; 1,65) + SpO2 N(0,72; 1,02)", lambda r: noisy(r, hr=(0.16, 1.65), spo2=(0.72, 1.02)))
repeat("D2a derau SBP N(+5; 8) (ISO 81060-2, bias +5)", lambda r: noisy(r, sbp=(5, 8)))
repeat("D2b derau SBP N(-5; 8) (ISO 81060-2, bias -5)", lambda r: noisy(r, sbp=(-5, 8)))
nan_rr = Xte_nan_raw.copy(); nan_rr[:, 4] = np.nan
pred = run_pipeline(nan_rr)
add("D3 RR hilang (di-impute MICE)", metrics(yte_color, pred),
    kategori_berubah_pct=round(100 * float((pred != clean_pred).mean()), 2))

# %% [markdown]
# ## Keluaran

# %%
table = pd.DataFrame(ROWS)
base_cols = ["skenario", "sens_merah", "bal_acc", "under_triage", "over_triage", "ukuran_kb", "waktu_ms"]
extra = [c for c in table.columns if c not in base_cols]
table = table[base_cols + extra]
for c in ["sens_merah", "bal_acc", "under_triage", "over_triage"]:
    table[c] = table[c].astype(float).round(4)
table.to_csv(OUT, index=False)
print(table[base_cols].to_string(index=False))
print("\nsaved", OUT)
