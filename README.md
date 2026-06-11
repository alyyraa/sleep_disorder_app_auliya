# Sleep Disorder & Stress Prediction App

Aplikasi web Flask yang menggunakan XGBoost untuk memprediksi gangguan tidur dan tingkat stres.

## Ringkasan

Aplikasi ini menyediakan:

- **Prediksi gangguan tidur** dengan XGBoost Classifier menjadi tiga kategori:
  - `None`
  - `Insomnia`
  - `Sleep Apnea`
- **Prediksi tingkat stres** dengan XGBoost Regressor pada skala 1-10
- **Halaman web** untuk pelatihan model, prediksi, dan informasi dataset

## Dataset

File dataset utama:

- `data/Sleep_health_and_lifestyle_dataset.csv`

Fitur yang dipakai dalam preprocessing dan model:

- `Gender`
- `Age`
- `Occupation`
- `Sleep Duration`
- `Quality of Sleep`
- `Physical Activity Level`
- `BMI Category`
- `Heart Rate`
- `Daily Steps`
- `Systolic BP`
- `Diastolic BP`

Catatan:

- `Person ID` dibuang saat preprocessing karena bukan fitur prediksi
- Kolom `Blood Pressure` dipecah menjadi `Systolic BP` dan `Diastolic BP`
- Target klasifikasi: `Sleep Disorder`
- Target regresi: `Stress Level`

## Instalasi

1. Buat virtual environment:

```powershell
python -m venv venv
```

2. Aktifkan virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

3. Install dependency:

```powershell
pip install -r requirements.txt
```

## Menjalankan Aplikasi

Jalankan aplikasi:

```powershell
python app.py
```

Lalu buka browser di:

- `http://127.0.0.1:5000`

## Penggunaan

### Halaman Utama
- `http://127.0.0.1:5000`
- Menampilkan landing page dan navigasi ke fitur prediksi dan pelatihan

### Pelatihan Model
- `http://127.0.0.1:5000/train`
- Menjalankan pipeline pelatihan XGBoost
- Menyimpan model dan preprocessing artifacts di folder `models/`

### Prediksi
- `http://127.0.0.1:5000/predict`
- Input data pengguna sesuai fitur dataset
- Menampilkan prediksi `Sleep Disorder`, `Stress Level`, probabilitas kelas, dan rekomendasi kesehatan

## Arsitektur Kode

### `app.py`
- Routing Flask untuk halaman home, predict, train, dan about
- Memuat model melalui `SleepDisorderPredictor`

### `utils/preprocessing.py`
- Memuat dataset
- Membersihkan data
- Mengencode kategori
- Menyusun fitur dan target
- Menyiapkan input prediksi

### `models/train_model.py`
- Pipeline pelatihan XGBoost
- Early stopping dan cross-validation
- Menyimpan model dan artefak preprocessing

### `models/predict_model.py`
- Memuat model yang sudah dilatih
- Menyiapkan input pengguna untuk prediksi
- Memproduksi prediksi klasifikasi dan regresi
- Menghasilkan rekomendasi kesehatan
- Mengambil feature importance

## Struktur Folder

```
.
├── app.py
├── data/
│   └── Sleep_health_and_lifestyle_dataset.csv
├── models/
│   ├── feature_names.joblib
│   ├── label_encoders.joblib
│   ├── scaler.joblib
│   ├── xgboost_classifier.joblib
│   ├── xgboost_regressor.joblib
│   ├── predict_model.py
│   └── train_model.py
├── requirements.txt
├── README.md
├── templates/
│   ├── about.html
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   └── train.html
├── utils/
│   ├── eda.py
│   └── preprocessing.py
└── static/
    └── css/
        └── style.css
```

## Catatan

- Aplikasi ini dibuat untuk tujuan akademis dan bukan sebagai diagnosis medis.
- Pastikan dataset tersedia dan model sudah dilatih sebelum melakukan prediksi.
