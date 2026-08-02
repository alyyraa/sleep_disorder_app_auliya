# SLEEP STRESS PREDICTOR

Sistem Informasi Prediksi Gangguan Tidur dan Tingkat Stres berbasis Flask dan XGBoost. Aplikasi ini menyediakan pengelolaan data penelitian, data pasien, pelatihan model, prediksi berbasis data pasien, riwayat prediksi, serta ekspor laporan.

> Catatan: aplikasi ini dibuat untuk keperluan akademik dan skrining awal. Hasil prediksi bukan diagnosis medis dan tidak menggantikan konsultasi dengan tenaga kesehatan.

## Fitur Utama

- Login, logout, dan manajemen pengguna berbasis peran Admin.
- Dashboard analitik untuk pasien, prediksi, gangguan tidur, rata-rata stres, dan metadata model.
- Master Data:
  - Training Dataset (374 data penelitian awal, CSV import, tambah, ubah, hapus)
  - Patients
  - Occupations
  - BMI Categories
- Pelatihan model XGBoost dari Training Dataset yang tersimpan di SQLite.
- Evaluasi model setelah pelatihan: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, MAE, RMSE, dan R².
- Prediksi berbasis data pasien—tanpa memasukkan fitur kesehatan secara manual pada halaman prediksi.
- Hasil prediksi Sleep Disorder, probabilitas kelas, Stress Level, rekomendasi, feature importance, dan informasi model.
- Prediction History dengan pencarian, pagination, detail, dan hapus.
- Laporan Training Dataset, Patients, dan Prediction History dalam format PDF dan Excel.

## Teknologi

- Python dan Flask
- Flask-SQLAlchemy + SQLite
- Flask-Login dan Werkzeug Password Hashing
- Bootstrap 5, Jinja2, DataTables, dan Chart.js
- XGBoost, scikit-learn, pandas, dan joblib
- ReportLab untuk PDF dan openpyxl untuk Excel

## Alur Sistem

```text
Login
  → Dashboard
  → Master Data
      → Training Dataset / Patients / Occupations / BMI Categories
  → Machine Learning
      → Train Model / New Prediction / Prediction History
  → Reports
  → Logout
```

### Alur Prediksi

```text
Pilih Patient
  → muat data Patient dari SQLite
  → existing preprocessing + saved artifacts
  → XGBoost Classifier dan XGBoost Regressor
  → Prediction Result
  → simpan Prediction History
```

Sleep Disorder dan Stress Level diprediksi oleh dua model XGBoost yang independen. Oleh sebab itu, kelas `None` tetap dapat memiliki prediksi tingkat stres sedang atau tinggi.

## Dataset dan Fitur Model

Saat inisialisasi database, aplikasi melakukan seed terhadap 374 data penelitian awal ke tabel `training_dataset_records`.

Fitur model:

- Gender
- Age
- Occupation
- Sleep Duration
- Quality of Sleep
- Physical Activity Level
- BMI Category
- Heart Rate
- Daily Steps
- Systolic BP
- Diastolic BP

Target:

- Klasifikasi: `Sleep Disorder` (`None`, `Insomnia`, `Sleep Apnea`)
- Regresi: `Stress Level` (skala 1–10)

`Person ID` tidak dipakai sebagai fitur model. Kolom `Blood Pressure` pada CSV penelitian dipisahkan menjadi `Systolic BP` dan `Diastolic BP` sebelum digunakan model.

## Instalasi

1. Clone repository.

```powershell
git clone https://github.com/alyyraa/sleep_disorder_app_auliya.git
cd sleep_disorder_app_auliya
```

2. Buat dan aktifkan virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies.

```powershell
pip install -r requirements.txt
```

4. Jalankan aplikasi.

```powershell
python app.py
```

5. Buka `http://127.0.0.1:5000` pada browser.

Database SQLite `sleep_disorder.db` akan dibuat atau digunakan secara otomatis. Pada inisialisasi pertama, data master dan Training Dataset awal akan di-seed.

## Akun Admin Awal

| Username | Password |
|---|---|
| `admin` | `admin123` |

Untuk penggunaan di luar lingkungan pengembangan, segera ganti password Admin dan atur environment variable `SECRET_KEY` yang aman.

## Pelatihan Model

Menu **Machine Learning → Train Model** membaca seluruh record pada Training Dataset dari SQLite, membangun format CSV penelitian sementara, lalu menjalankan pipeline pelatihan yang tersedia.

Model dan preprocessing yang digunakan tetap berada pada:

- `models/train_model.py`
- `models/predict_model.py`
- `utils/preprocessing.py`

Model artifacts aktif disimpan dalam folder `models/`, termasuk:

- `xgboost_classifier.joblib`
- `xgboost_regressor.joblib`
- `scaler.joblib`
- `label_encoders.joblib`
- `feature_names.joblib`

## Struktur Proyek

```text
.
├── app.py
├── extensions.py
├── models/
│   ├── database.py
│   ├── train_model.py
│   ├── predict_model.py
│   └── *.joblib
├── routes/
│   ├── auth.py
│   ├── master_data.py
│   ├── prediction.py
│   ├── reports.py
│   ├── system.py
│   ├── training_dataset.py
│   └── users.py
├── services/
│   ├── database_seed.py
│   ├── prediction_service.py
│   ├── training_service.py
│   ├── pdf_report_service.py
│   └── excel_report_service.py
├── static/
├── templates/
├── utils/
│   ├── access.py
│   ├── preprocessing.py
│   └── timezone.py
├── data/
├── requirements.txt
└── README.md
```

## Database Utama

| Tabel | Fungsi |
|---|---|
| `users` | Akun pengguna aplikasi |
| `occupations` | Master occupation untuk data yang sesuai LabelEncoder |
| `bmi_categories` | Master BMI Category untuk data yang sesuai LabelEncoder |
| `training_dataset_records` | Data penelitian yang digunakan Train Model |
| `patients` | Data pasien baru untuk prediksi |
| `prediction_history` | Hasil prediksi yang tersimpan |
| `model_metadata` | Model aktif, versi, dan waktu pelatihan terakhir |

## Laporan

Menu **System → Reports** menyediakan:

- Training Dataset Report — PDF / Excel
- Patient Report — PDF / Excel
- Prediction History Report — PDF / Excel

PDF menggunakan kop surat resmi aplikasi dan tanggal Asia/Jakarta. Excel menggunakan header berformat, border, auto-width, dan freeze header row.

## Lisensi

Digunakan untuk keperluan penelitian dan tugas akhir.
