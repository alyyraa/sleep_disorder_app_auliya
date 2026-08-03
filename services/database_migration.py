"""Small additive SQLite migrations for existing local databases."""

import json

from sqlalchemy import inspect, text
from sqlalchemy.orm import joinedload

from extensions import db
from models.database import ModelMetadata, Patient, PredictionHistory
from services.patient_snapshot_service import patient_snapshot
from services.training_service import training_dataset_signature


def _add_column_if_missing(table_name, column_name, definition):
    columns = {column["name"] for column in inspect(db.engine).get_columns(table_name)}
    if column_name not in columns:
        with db.engine.begin() as connection:
            connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"))


def ensure_database_schema():
    """Upgrade the existing SQLite file without replacing its data."""
    _add_column_if_missing("prediction_history", "patient_snapshot", "TEXT")
    _add_column_if_missing("model_metadata", "training_dataset_signature", "VARCHAR(64)")
    db.session.expire_all()

    histories = PredictionHistory.query.options(
        joinedload(PredictionHistory.patient).joinedload(Patient.occupation),
        joinedload(PredictionHistory.patient).joinedload(Patient.bmi_category),
    ).all()
    for history in histories:
        if not history.patient_snapshot:
            history.patient_snapshot = json.dumps(patient_snapshot(history.patient))

    metadata = db.session.get(ModelMetadata, 1)
    if metadata and not metadata.training_dataset_signature:
        metadata.training_dataset_signature = training_dataset_signature()
    db.session.commit()
