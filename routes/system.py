"""Authenticated information-system shell and Phase 2 module placeholders."""

from flask import Blueprint, render_template
from flask_login import login_required

from extensions import db
from models.database import ModelMetadata, Patient, PredictionHistory, TrainingDatasetRecord

system_bp = Blueprint("system", __name__)


@system_bp.get("/dashboard")
@login_required
def dashboard():
    metrics = {
        "patients": Patient.query.count(),
        "predictions": PredictionHistory.query.count(),
        "insomnia": PredictionHistory.query.filter_by(sleep_disorder="Insomnia").count(),
        "sleep_apnea": PredictionHistory.query.filter_by(sleep_disorder="Sleep Apnea").count(),
        "training_dataset_records": TrainingDatasetRecord.query.count(),
    }
    return render_template(
        "dashboard/index.html",
        metrics=metrics,
        model_metadata=db.session.get(ModelMetadata, 1),
    )


def pending_module(title, message):
    return render_template("system/module_pending.html", title=title, message=message)


@system_bp.get("/about")
@login_required
def about():
    return render_template("about.html")
