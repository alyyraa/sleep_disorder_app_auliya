"""Adapter between database-backed training records and the existing ML pipeline."""

import os
import tempfile

import pandas as pd
from sqlalchemy.orm import joinedload

from models.database import TrainingDatasetRecord
from models.train_model import train_models_pipeline


def train_models_from_database():
    """Export all training records in the original CSV format and run existing training."""
    records = (
        TrainingDatasetRecord.query.options(
            joinedload(TrainingDatasetRecord.occupation),
            joinedload(TrainingDatasetRecord.bmi_category),
        )
        .order_by(TrainingDatasetRecord.id)
        .all()
    )
    if not records:
        raise ValueError("Training Dataset is empty. Add training records before training the model.")

    dataset = pd.DataFrame(
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

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8", newline="", delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            dataset.to_csv(temporary_file, index=False)
        return train_models_pipeline(temporary_path), len(records)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
