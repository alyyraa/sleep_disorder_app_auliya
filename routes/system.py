"""Authenticated information-system shell and Phase 2 module placeholders."""

from flask import Blueprint, render_template
from flask_login import login_required
from sqlalchemy import func
from sqlalchemy.orm import joinedload

from extensions import db
from models.database import ModelMetadata, Patient, PredictionHistory, TrainingDatasetRecord

system_bp = Blueprint("system", __name__)


@system_bp.get("/dashboard")
@login_required
def dashboard():
    total_predictions = PredictionHistory.query.count()
    average_stress = db.session.query(func.avg(PredictionHistory.stress_level)).scalar()
    average_sleep_duration = db.session.query(func.avg(Patient.sleep_duration)).scalar()
    average_heart_rate = db.session.query(func.avg(Patient.heart_rate)).scalar()
    most_common = (
        db.session.query(
            PredictionHistory.sleep_disorder,
            func.count(PredictionHistory.id).label("total"),
        )
        .group_by(PredictionHistory.sleep_disorder)
        .order_by(func.count(PredictionHistory.id).desc(), PredictionHistory.sleep_disorder)
        .first()
    )

    if average_stress is None:
        stress_category, stress_class = "No data", "secondary"
    elif average_stress < 4:
        stress_category, stress_class = "Low", "success"
    elif average_stress < 7:
        stress_category, stress_class = "Medium", "warning"
    else:
        stress_category, stress_class = "High", "danger"

    metrics = {
        "patients": Patient.query.count(),
        "predictions": total_predictions,
        "insomnia": PredictionHistory.query.filter_by(sleep_disorder="Insomnia").count(),
        "sleep_apnea": PredictionHistory.query.filter_by(sleep_disorder="Sleep Apnea").count(),
        "training_dataset_records": TrainingDatasetRecord.query.count(),
        "average_stress": float(average_stress) if average_stress is not None else None,
        "stress_category": stress_category,
        "stress_class": stress_class,
        "average_sleep_duration": float(average_sleep_duration) if average_sleep_duration is not None else None,
        "average_heart_rate": float(average_heart_rate) if average_heart_rate is not None else None,
        "most_common_disorder": most_common.sleep_disorder if most_common else None,
        "most_common_count": most_common.total if most_common else 0,
        "most_common_percentage": (most_common.total / total_predictions * 100) if most_common and total_predictions else None,
    }
    recent_predictions = (
        PredictionHistory.query.options(joinedload(PredictionHistory.patient))
        .order_by(PredictionHistory.prediction_date.desc())
        .limit(5)
        .all()
    )
    return render_template(
        "dashboard/index.html",
        metrics=metrics,
        model_metadata=db.session.get(ModelMetadata, 1),
        recent_predictions=recent_predictions,
    )


def pending_module(title, message):
    return render_template("system/module_pending.html", title=title, message=message)


@system_bp.get("/about")
@login_required
def about():
    return render_template("about.html")
