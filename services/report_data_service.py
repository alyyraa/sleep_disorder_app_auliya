"""Read-only data builders for Training Dataset, Patient, and Prediction reports."""

from sqlalchemy import func
from sqlalchemy.orm import joinedload

from extensions import db
from models.database import ModelMetadata, Patient, PredictionHistory, TrainingDatasetRecord
from utils.timezone import format_indonesian_date, jakarta_now


def _format_date(value):
    return format_indonesian_date(value, include_time=True)


def report_dashboard_cards():
    metadata = db.session.get(ModelMetadata, 1)
    return [
        {"key": "training_dataset", "title": "Training Dataset Report", "description": "Research records used to train the XGBoost models.", "total": TrainingDatasetRecord.query.count(), "last_updated": TrainingDatasetRecord.query.with_entities(func.max(TrainingDatasetRecord.updated_at)).scalar()},
        {"key": "patients", "title": "Patient Report", "description": "Patient records maintained for prediction.", "total": Patient.query.count(), "last_updated": Patient.query.with_entities(func.max(Patient.updated_at)).scalar()},
        {"key": "prediction_history", "title": "Prediction History Report", "description": "Saved patient prediction results and model versions.", "total": PredictionHistory.query.count(), "last_updated": PredictionHistory.query.with_entities(func.max(PredictionHistory.prediction_date)).scalar(), "model_version": metadata.model_version if metadata else "-"},
    ]


def training_dataset_report():
    records = TrainingDatasetRecord.query.options(joinedload(TrainingDatasetRecord.occupation), joinedload(TrainingDatasetRecord.bmi_category)).order_by(TrainingDatasetRecord.id).all()
    headers = ["ID Responden", "Jenis Kelamin", "Usia", "Pekerjaan", "Durasi Tidur", "Kualitas Tidur", "Tingkat Aktivitas Fisik", "Tingkat Stres", "Kategori BMI", "Tekanan Darah", "Detak Jantung", "Langkah Harian", "Gangguan Tidur"]
    rows = [[record.source_person_id or record.id, record.gender, record.age, record.occupation.name, record.sleep_duration, record.quality_of_sleep, record.physical_activity_level, record.stress_level, record.bmi_category.name, f"{record.systolic_bp}/{record.diastolic_bp}", record.heart_rate, record.daily_steps, record.sleep_disorder] for record in records]
    return {"title": "Laporan Dataset Pelatihan", "headers": headers, "rows": rows, "total_records": len(records), "summary": None}


def patient_report():
    records = Patient.query.options(joinedload(Patient.occupation), joinedload(Patient.bmi_category)).order_by(Patient.full_name).all()
    headers = ["Nama Pasien", "Jenis Kelamin", "Usia", "Pekerjaan", "Durasi Tidur", "Kualitas Tidur", "Tingkat Aktivitas Fisik", "Langkah Harian", "Kategori BMI", "Detak Jantung", "Tekanan Darah", "Tanggal Dibuat"]
    rows = [[record.full_name, record.gender, record.age, record.occupation.name, record.sleep_duration, record.quality_of_sleep, record.physical_activity_level, record.daily_steps, record.bmi_category.name, record.heart_rate, f"{record.systolic_bp}/{record.diastolic_bp}", _format_date(record.created_at)] for record in records]
    return {"title": "Laporan Data Pasien", "headers": headers, "rows": rows, "total_records": len(records), "summary": None}


def prediction_history_report():
    records = PredictionHistory.query.options(joinedload(PredictionHistory.patient)).order_by(PredictionHistory.prediction_date.desc()).all()
    metadata = db.session.get(ModelMetadata, 1)
    headers = ["Tanggal Prediksi", "Nama Pasien", "Gangguan Tidur", "Probabilitas Gangguan Tidur (%)", "Tingkat Stres", "Versi Model"]
    rows = [[_format_date(record.prediction_date), record.patient_snapshot_data.get("full_name", record.patient.full_name), record.sleep_disorder, f"{record.sleep_probability * 100:.1f}%" if record.sleep_probability is not None else "-", f"{record.stress_level:.1f}" if record.stress_level is not None else "-", record.model_version] for record in records]
    stress_values = [record.stress_level for record in records if record.stress_level is not None]
    summary = [
        ("Total Prediksi", len(records)),
        ("Total Insomnia", sum(record.sleep_disorder == "Insomnia" for record in records)),
        ("Total Sleep Apnea", sum(record.sleep_disorder == "Sleep Apnea" for record in records)),
        ("Total Tidak Ada Gangguan", sum(record.sleep_disorder == "None" for record in records)),
        ("Rata-rata Tingkat Stres", f"{sum(stress_values) / len(stress_values):.1f}" if stress_values else "-"),
        ("Versi Model Saat Ini", metadata.model_version if metadata else "-"),
        ("Tanggal Cetak", _format_date(jakarta_now())),
    ]
    return {"title": "Laporan Riwayat Prediksi", "headers": headers, "rows": rows, "total_records": len(records), "summary": summary}


REPORT_BUILDERS = {"training_dataset": training_dataset_report, "patients": patient_report, "prediction_history": prediction_history_report}


def get_report_data(report_key):
    try:
        return REPORT_BUILDERS[report_key]()
    except KeyError as error:
        raise ValueError("Unknown report type.") from error
