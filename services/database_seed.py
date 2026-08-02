"""Initial database data. Values mirror the trained model's known categories."""

from datetime import datetime
from pathlib import Path

import pandas as pd
from werkzeug.security import generate_password_hash

from extensions import db
from models.database import BmiCategory, ModelMetadata, Occupation, TrainingDatasetRecord, User
from utils.timezone import JAKARTA_TZ


MODEL_OCCUPATIONS = [
    "Accountant",
    "Doctor",
    "Engineer",
    "Lawyer",
    "Manager",
    "Nurse",
    "Sales Representative",
    "Salesperson",
    "Scientist",
    "Software Engineer",
    "Teacher",
]

MODEL_BMI_CATEGORIES = ["Normal", "Normal Weight", "Obese", "Overweight"]


def seed_database():
    """Create the default administrator and approved master-data choices once."""
    if User.query.filter_by(username="admin").first() is None:
        db.session.add(
            User(
                full_name="System Administrator",
                username="admin",
                password_hash=generate_password_hash("admin123"),
                role="Admin",
            )
        )

    for name in MODEL_OCCUPATIONS:
        if Occupation.query.filter_by(name=name).first() is None:
            db.session.add(Occupation(name=name))

    for name in MODEL_BMI_CATEGORIES:
        if BmiCategory.query.filter_by(name=name).first() is None:
            db.session.add(BmiCategory(name=name))

    if db.session.get(ModelMetadata, 1) is None:
        model_paths = [
            Path("models/xgboost_classifier.joblib"),
            Path("models/xgboost_regressor.joblib"),
        ]
        available_dates = [
            datetime.fromtimestamp(path.stat().st_mtime, JAKARTA_TZ).replace(tzinfo=None) for path in model_paths if path.exists()
        ]
        db.session.add(
            ModelMetadata(
                id=1,
                active_model="XGBoost Classifier and Regressor",
                model_version="v1",
                last_training_date=max(available_dates) if available_dates else None,
            )
        )

    db.session.commit()
    seed_original_training_dataset()


def seed_original_training_dataset():
    """Load the 374 original research records exactly once on a new database."""
    if TrainingDatasetRecord.query.count() > 0:
        return

    dataset_path = Path(__file__).resolve().parents[1] / "data" / "Sleep_health_and_lifestyle_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Original training dataset is missing: {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    occupations = {item.name: item.id for item in Occupation.query.all()}
    bmi_categories = {item.name: item.id for item in BmiCategory.query.all()}
    records = []

    for _, row in dataset.iterrows():
        try:
            systolic_bp, diastolic_bp = [int(value) for value in str(row["Blood Pressure"]).split("/", 1)]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid Blood Pressure value in original dataset: {row['Blood Pressure']}") from error

        occupation_name = str(row["Occupation"])
        bmi_name = str(row["BMI Category"])
        if occupation_name not in occupations or bmi_name not in bmi_categories:
            raise ValueError("Original dataset contains an occupation or BMI category outside master data.")

        records.append(
            TrainingDatasetRecord(
                source_person_id=int(row["Person ID"]),
                gender=str(row["Gender"]),
                age=int(row["Age"]),
                occupation_id=occupations[occupation_name],
                sleep_duration=float(row["Sleep Duration"]),
                quality_of_sleep=int(row["Quality of Sleep"]),
                physical_activity_level=int(row["Physical Activity Level"]),
                stress_level=float(row["Stress Level"]),
                bmi_category_id=bmi_categories[bmi_name],
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=int(row["Heart Rate"]),
                daily_steps=int(row["Daily Steps"]),
                sleep_disorder="None" if pd.isna(row["Sleep Disorder"]) else str(row["Sleep Disorder"]),
            )
        )

    if len(records) != 374:
        raise ValueError(f"Expected 374 original training records, found {len(records)}.")

    db.session.add_all(records)
    db.session.commit()
