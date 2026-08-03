"""Database entities for the administrative information system."""

import json

from flask_login import UserMixin

from extensions import db
from utils.timezone import jakarta_now


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(80), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="Admin")
    created_at = db.Column(db.DateTime, nullable=False, default=jakarta_now)


class Occupation(db.Model):
    __tablename__ = "occupations"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False, unique=True, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=jakarta_now)

    training_dataset_records = db.relationship("TrainingDatasetRecord", back_populates="occupation")
    patients = db.relationship("Patient", back_populates="occupation")


class BmiCategory(db.Model):
    __tablename__ = "bmi_categories"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False, unique=True, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=jakarta_now)

    training_dataset_records = db.relationship("TrainingDatasetRecord", back_populates="bmi_category")
    patients = db.relationship("Patient", back_populates="bmi_category")


class TrainingDatasetRecord(db.Model):
    """A research record used exclusively to train the XGBoost models."""

    __tablename__ = "training_dataset_records"

    id = db.Column(db.Integer, primary_key=True)
    source_person_id = db.Column(db.Integer, nullable=True, index=True)
    gender = db.Column(db.String(20), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    occupation_id = db.Column(db.Integer, db.ForeignKey("occupations.id"), nullable=False, index=True)
    sleep_duration = db.Column(db.Float, nullable=False)
    quality_of_sleep = db.Column(db.Integer, nullable=False)
    physical_activity_level = db.Column(db.Integer, nullable=False)
    stress_level = db.Column(db.Float, nullable=False)
    bmi_category_id = db.Column(db.Integer, db.ForeignKey("bmi_categories.id"), nullable=False, index=True)
    systolic_bp = db.Column(db.Integer, nullable=False)
    diastolic_bp = db.Column(db.Integer, nullable=False)
    heart_rate = db.Column(db.Integer, nullable=False)
    daily_steps = db.Column(db.Integer, nullable=False)
    sleep_disorder = db.Column(db.String(80), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=jakarta_now)
    updated_at = db.Column(db.DateTime, nullable=False, default=jakarta_now, onupdate=jakarta_now)

    occupation = db.relationship("Occupation", back_populates="training_dataset_records")
    bmi_category = db.relationship("BmiCategory", back_populates="training_dataset_records")


class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    occupation_id = db.Column(db.Integer, db.ForeignKey("occupations.id"), nullable=False, index=True)
    sleep_duration = db.Column(db.Float, nullable=False)
    quality_of_sleep = db.Column(db.Integer, nullable=False)
    physical_activity_level = db.Column(db.Integer, nullable=False)
    daily_steps = db.Column(db.Integer, nullable=False)
    bmi_category_id = db.Column(db.Integer, db.ForeignKey("bmi_categories.id"), nullable=False, index=True)
    heart_rate = db.Column(db.Integer, nullable=False)
    systolic_bp = db.Column(db.Integer, nullable=False)
    diastolic_bp = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=jakarta_now)
    updated_at = db.Column(db.DateTime, nullable=False, default=jakarta_now, onupdate=jakarta_now)

    occupation = db.relationship("Occupation", back_populates="patients")
    bmi_category = db.relationship("BmiCategory", back_populates="patients")
    prediction_history = db.relationship(
        "PredictionHistory", back_populates="patient", cascade="all, delete-orphan"
    )


class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False, index=True)
    sleep_disorder = db.Column(db.String(80), nullable=False)
    sleep_probability = db.Column(db.Float, nullable=True)
    stress_level = db.Column(db.Float, nullable=True)
    recommendation = db.Column(db.Text, nullable=True)
    model_version = db.Column(db.String(20), nullable=False)
    prediction_date = db.Column(db.DateTime, nullable=False, default=jakarta_now, index=True)
    patient_snapshot = db.Column(db.Text, nullable=True)

    patient = db.relationship("Patient", back_populates="prediction_history")

    @property
    def patient_snapshot_data(self):
        if not self.patient_snapshot:
            return {}
        try:
            return json.loads(self.patient_snapshot)
        except (TypeError, ValueError):
            return {}


class ModelMetadata(db.Model):
    """Singleton record identifying the currently active XGBoost artifacts."""

    __tablename__ = "model_metadata"

    id = db.Column(db.Integer, primary_key=True)
    active_model = db.Column(db.String(150), nullable=False)
    model_version = db.Column(db.String(20), nullable=False)
    last_training_date = db.Column(db.DateTime, nullable=True)
    training_dataset_signature = db.Column(db.String(64), nullable=True)
