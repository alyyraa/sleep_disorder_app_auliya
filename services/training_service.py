"""Adapter between database-backed training records and the existing ML pipeline."""

import hashlib
import json
import os
import tempfile

import pandas as pd
from sqlalchemy.orm import joinedload

from extensions import db
from models.database import BmiCategory, Occupation, TrainingDatasetRecord
from models.train_model import train_models_pipeline

TRAINING_DATASET_COLUMNS = [
    "Person ID",
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Blood Pressure",
    "Heart Rate",
    "Daily Steps",
    "Sleep Disorder",
]


def _training_records():
    return (
        TrainingDatasetRecord.query.options(
            joinedload(TrainingDatasetRecord.occupation),
            joinedload(TrainingDatasetRecord.bmi_category),
        )
        .order_by(TrainingDatasetRecord.id)
        .all()
    )


def training_dataset_signature():
    """Return a stable fingerprint of the current Training Dataset state."""
    return training_dataset_content_signature(training_dataset_dataframe())


def legacy_training_dataset_signature():
    """Read the pre-restore fingerprint format only for verified archive migration."""
    state = [
        {
            "id": record.id,
            "source_person_id": record.source_person_id,
            "gender": record.gender,
            "age": record.age,
            "occupation": record.occupation.name,
            "sleep_duration": record.sleep_duration,
            "quality_of_sleep": record.quality_of_sleep,
            "physical_activity_level": record.physical_activity_level,
            "stress_level": record.stress_level,
            "bmi_category": record.bmi_category.name,
            "systolic_bp": record.systolic_bp,
            "diastolic_bp": record.diastolic_bp,
            "heart_rate": record.heart_rate,
            "daily_steps": record.daily_steps,
            "sleep_disorder": record.sleep_disorder,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        }
        for record in _training_records()
    ]
    encoded_state = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded_state.encode("utf-8")).hexdigest()


def normalize_training_dataset_dataframe(dataset):
    """Validate and normalize the original research CSV representation."""
    missing = [column for column in TRAINING_DATASET_COLUMNS if column not in dataset.columns]
    if missing:
        raise ValueError(f"Training Dataset CSV is missing columns: {', '.join(missing)}")

    rows = []
    for row_number, (_, row) in enumerate(dataset[TRAINING_DATASET_COLUMNS].iterrows(), start=2):
        try:
            pressure = str(row["Blood Pressure"]).strip().split("/", 1)
            if len(pressure) != 2:
                raise ValueError
            sleep_disorder = "None" if pd.isna(row["Sleep Disorder"]) else str(row["Sleep Disorder"]).strip()
            normalized = {
                "Person ID": int(row["Person ID"]),
                "Gender": str(row["Gender"]).strip(),
                "Age": int(row["Age"]),
                "Occupation": str(row["Occupation"]).strip(),
                "Sleep Duration": float(row["Sleep Duration"]),
                "Quality of Sleep": int(row["Quality of Sleep"]),
                "Physical Activity Level": int(row["Physical Activity Level"]),
                "Stress Level": float(row["Stress Level"]),
                "BMI Category": str(row["BMI Category"]).strip(),
                "Blood Pressure": f"{int(pressure[0])}/{int(pressure[1])}",
                "Heart Rate": int(row["Heart Rate"]),
                "Daily Steps": int(row["Daily Steps"]),
                "Sleep Disorder": sleep_disorder,
            }
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Training Dataset CSV row {row_number} contains invalid values.") from error

        if normalized["Gender"] not in {"Male", "Female"}:
            raise ValueError(f"Training Dataset CSV row {row_number} has an invalid Gender.")
        if normalized["Sleep Disorder"] not in {"None", "Insomnia", "Sleep Apnea"}:
            raise ValueError(f"Training Dataset CSV row {row_number} has an invalid Sleep Disorder.")
        if not normalized["Occupation"] or not normalized["BMI Category"]:
            raise ValueError(f"Training Dataset CSV row {row_number} has incomplete master data.")
        if not 1 <= normalized["Age"] <= 120 or not 0 < normalized["Sleep Duration"] <= 24:
            raise ValueError(f"Training Dataset CSV row {row_number} has values outside the allowed range.")
        if not 1 <= normalized["Quality of Sleep"] <= 10 or not 1 <= normalized["Stress Level"] <= 10:
            raise ValueError(f"Training Dataset CSV row {row_number} has values outside the allowed range.")
        if not 0 <= normalized["Physical Activity Level"] <= 100:
            raise ValueError(f"Training Dataset CSV row {row_number} has values outside the allowed range.")
        rows.append(normalized)

    if not rows:
        raise ValueError("Training Dataset CSV does not contain any records.")
    return pd.DataFrame(rows, columns=TRAINING_DATASET_COLUMNS)


def training_dataset_content_signature(dataset):
    """Hash only stable research values, excluding database IDs and timestamps."""
    normalized = normalize_training_dataset_dataframe(dataset)
    content = normalized.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def restore_training_dataset_dataframe(dataset):
    """Replace active training records from a validated archived DataFrame."""
    normalized = normalize_training_dataset_dataframe(dataset)
    occupation_names = sorted(set(normalized["Occupation"]))
    bmi_names = sorted(set(normalized["BMI Category"]))

    occupations = {item.name: item for item in Occupation.query.all()}
    bmi_categories = {item.name: item for item in BmiCategory.query.all()}
    for name in occupation_names:
        if name not in occupations:
            item = Occupation(name=name)
            db.session.add(item)
            occupations[name] = item
    for name in bmi_names:
        if name not in bmi_categories:
            item = BmiCategory(name=name)
            db.session.add(item)
            bmi_categories[name] = item
    db.session.flush()

    TrainingDatasetRecord.query.delete(synchronize_session=False)
    restored = []
    for _, row in normalized.iterrows():
        systolic_bp, diastolic_bp = [int(value) for value in row["Blood Pressure"].split("/", 1)]
        restored.append(
            TrainingDatasetRecord(
                source_person_id=int(row["Person ID"]),
                gender=row["Gender"],
                age=int(row["Age"]),
                occupation_id=occupations[row["Occupation"]].id,
                sleep_duration=float(row["Sleep Duration"]),
                quality_of_sleep=int(row["Quality of Sleep"]),
                physical_activity_level=int(row["Physical Activity Level"]),
                stress_level=float(row["Stress Level"]),
                bmi_category_id=bmi_categories[row["BMI Category"]].id,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=int(row["Heart Rate"]),
                daily_steps=int(row["Daily Steps"]),
                sleep_disorder=row["Sleep Disorder"],
            )
        )
    db.session.add_all(restored)
    db.session.flush()
    return len(restored)


def training_dataset_dataframe():
    """Return all database training records in the original research CSV format."""
    records = _training_records()
    if not records:
        raise ValueError("Training Dataset is empty. Add training records before training the model.")

    return pd.DataFrame(
        [
            {
                "Person ID": record.source_person_id or record.id,
                "Gender": record.gender,
                "Age": record.age,
                "Occupation": record.occupation.name,
                "Sleep Duration": record.sleep_duration,
                "Quality of Sleep": record.quality_of_sleep,
                "Physical Activity Level": record.physical_activity_level,
                "Stress Level": record.stress_level,
                "BMI Category": record.bmi_category.name,
                "Blood Pressure": f"{record.systolic_bp}/{record.diastolic_bp}",
                "Heart Rate": record.heart_rate,
                "Daily Steps": record.daily_steps,
                "Sleep Disorder": record.sleep_disorder,
            }
            for record in records
        ]
    )


def train_models_from_database():
    """Export all training records in the original CSV format and run existing training."""
    dataset = training_dataset_dataframe()

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8", newline="", delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            dataset.to_csv(temporary_file, index=False)
        return train_models_pipeline(temporary_path), len(dataset)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
