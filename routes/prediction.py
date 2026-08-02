"""Patient-based prediction and prediction-history routes."""

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required
from sqlalchemy.orm import joinedload

from extensions import db
from models.database import ModelMetadata, Patient, PredictionHistory
from services.prediction_service import predict_patient
from utils.timezone import jakarta_now

prediction_bp = Blueprint("prediction", __name__, url_prefix="/prediction")


@prediction_bp.route("/new", methods=["GET", "POST"])
@login_required
def new_prediction():
    patients = Patient.query.order_by(Patient.full_name).all()
    if request.method == "POST":
        try:
            patient_id = int(request.form.get("patient_id", ""))
        except ValueError:
            patient_id = None
        patient = db.session.get(Patient, patient_id) if patient_id else None
        if patient is None:
            flash("Select a valid patient.", "danger")
        else:
            try:
                results = predict_patient(patient)
                sleep_disorder = results["sleep_disorder"]
                probability = (results.get("disorder_probabilities") or {}).get(sleep_disorder)
                metadata = db.session.get(ModelMetadata, 1)
                history = PredictionHistory(
                    patient_id=patient.id,
                    sleep_disorder=sleep_disorder,
                    sleep_probability=probability,
                    stress_level=results.get("stress_level"),
                    recommendation="\n".join(results.get("recommendations") or []),
                    model_version=metadata.model_version if metadata else "v1",
                    prediction_date=jakarta_now(),
                )
                db.session.add(history)
                db.session.commit()
                return redirect(url_for("prediction.detail", history_id=history.id))
            except Exception as error:
                db.session.rollback()
                flash(f"Prediction failed: {error}", "danger")
    return render_template("prediction/new.html", patients=patients)


@prediction_bp.get("/history")
@login_required
def history():
    records = (
        PredictionHistory.query.options(joinedload(PredictionHistory.patient))
        .order_by(PredictionHistory.prediction_date.desc())
        .all()
    )
    return render_template("prediction/history.html", records=records)


@prediction_bp.get("/history/<int:history_id>")
@login_required
def detail(history_id):
    record = (
        PredictionHistory.query.options(
            joinedload(PredictionHistory.patient).joinedload(Patient.occupation),
            joinedload(PredictionHistory.patient).joinedload(Patient.bmi_category),
        )
        .filter_by(id=history_id)
        .first_or_404()
    )
    return render_template("prediction/detail.html", record=record)


@prediction_bp.post("/history/<int:history_id>/delete")
@login_required
def delete(history_id):
    record = db.get_or_404(PredictionHistory, history_id)
    db.session.delete(record)
    db.session.commit()
    flash("Prediction History record deleted successfully.", "success")
    return redirect(url_for("prediction.history"))
