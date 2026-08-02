"""CRUD routes for Patients, Occupations, and BMI Categories."""

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from extensions import db
from models.database import BmiCategory, Occupation, Patient

master_data_bp = Blueprint("master_data", __name__, url_prefix="/master-data")


def _patient_values():
    try:
        values = {"full_name": request.form.get("full_name", "").strip(), "gender": request.form.get("gender", ""), "age": int(request.form.get("age", "")), "occupation_id": int(request.form.get("occupation_id", "")), "sleep_duration": float(request.form.get("sleep_duration", "")), "quality_of_sleep": int(request.form.get("quality_of_sleep", "")), "physical_activity_level": int(request.form.get("physical_activity_level", "")), "daily_steps": int(request.form.get("daily_steps", "")), "bmi_category_id": int(request.form.get("bmi_category_id", "")), "heart_rate": int(request.form.get("heart_rate", "")), "systolic_bp": int(request.form.get("systolic_bp", "")), "diastolic_bp": int(request.form.get("diastolic_bp", ""))}
    except ValueError:
        return None, "All patient health values must be valid numbers."
    if not values["full_name"] or values["gender"] not in {"Male", "Female"}:
        return None, "Full name and gender are required."
    if not 1 <= values["age"] <= 120 or not 0 < values["sleep_duration"] <= 24 or not 1 <= values["quality_of_sleep"] <= 10:
        return None, "Age, sleep duration, or sleep quality is outside the allowed range."
    if not 0 <= values["physical_activity_level"] <= 100 or min(values["daily_steps"], values["heart_rate"], values["systolic_bp"], values["diastolic_bp"]) < 0:
        return None, "Activity, vital signs, and daily steps are outside the allowed range."
    if not db.session.get(Occupation, values["occupation_id"]) or not db.session.get(BmiCategory, values["bmi_category_id"]):
        return None, "Select a valid Occupation and BMI Category."
    return values, None


@master_data_bp.get("/patients")
@login_required
def patients():
    return render_template("patients/index.html", patients=Patient.query.order_by(Patient.created_at.desc()).all())


@master_data_bp.route("/patients/create", methods=["GET", "POST"])
@login_required
def patient_create():
    if request.method == "POST":
        values, error = _patient_values()
        if error:
            flash(error, "danger")
        else:
            db.session.add(Patient(**values)); db.session.commit(); flash("Patient created successfully.", "success")
            return redirect(url_for("master_data.patients"))
    return render_template("patients/form.html", patient=None, occupations=Occupation.query.order_by(Occupation.name).all(), bmi_categories=BmiCategory.query.order_by(BmiCategory.name).all())


@master_data_bp.route("/patients/<int:patient_id>/edit", methods=["GET", "POST"])
@login_required
def patient_edit(patient_id):
    patient = db.get_or_404(Patient, patient_id)
    if request.method == "POST":
        values, error = _patient_values()
        if error:
            flash(error, "danger")
        else:
            for key, value in values.items(): setattr(patient, key, value)
            db.session.commit(); flash("Patient updated successfully.", "success")
            return redirect(url_for("master_data.patients"))
    return render_template("patients/form.html", patient=patient, occupations=Occupation.query.order_by(Occupation.name).all(), bmi_categories=BmiCategory.query.order_by(BmiCategory.name).all())


@master_data_bp.post("/patients/<int:patient_id>/delete")
@login_required
def patient_delete(patient_id):
    db.session.delete(db.get_or_404(Patient, patient_id)); db.session.commit(); flash("Patient deleted successfully.", "success")
    return redirect(url_for("master_data.patients"))


def _master_routes(model, label, endpoint_prefix):
    def list_view(): return render_template("master_data/list.html", items=model.query.order_by(model.name).all(), label=label, endpoint_prefix=endpoint_prefix)
    def create_view():
        value = request.form.get("name", "").strip()
        if request.method == "POST":
            if not value: flash(f"{label} is required.", "danger")
            elif model.query.filter(func.lower(model.name) == value.lower()).first(): flash(f"{label} already exists.", "danger")
            else:
                db.session.add(model(name=value)); db.session.commit(); flash(f"{label} created successfully.", "success")
                return redirect(url_for(f"master_data.{endpoint_prefix}"))
        return render_template("master_data/form.html", item=None, label=label, endpoint_prefix=endpoint_prefix)
    def edit_view(item_id):
        item = db.get_or_404(model, item_id); value = request.form.get("name", "").strip()
        if request.method == "POST":
            if not value: flash(f"{label} is required.", "danger")
            elif model.query.filter(func.lower(model.name) == value.lower(), model.id != item.id).first(): flash(f"{label} already exists.", "danger")
            else:
                item.name = value; db.session.commit(); flash(f"{label} updated successfully.", "success")
                return redirect(url_for(f"master_data.{endpoint_prefix}"))
        return render_template("master_data/form.html", item=item, label=label, endpoint_prefix=endpoint_prefix)
    def delete_view(item_id):
        try:
            db.session.delete(db.get_or_404(model, item_id)); db.session.commit(); flash(f"{label} deleted successfully.", "success")
        except IntegrityError:
            db.session.rollback(); flash(f"{label} cannot be deleted because it is in use.", "danger")
        return redirect(url_for(f"master_data.{endpoint_prefix}"))
    list_view.__name__, create_view.__name__, edit_view.__name__, delete_view.__name__ = endpoint_prefix, f"{endpoint_prefix}_create", f"{endpoint_prefix}_edit", f"{endpoint_prefix}_delete"
    master_data_bp.add_url_rule(f"/{endpoint_prefix}", view_func=login_required(list_view))
    master_data_bp.add_url_rule(f"/{endpoint_prefix}/create", view_func=login_required(create_view), methods=["GET", "POST"])
    master_data_bp.add_url_rule(f"/{endpoint_prefix}/<int:item_id>/edit", view_func=login_required(edit_view), methods=["GET", "POST"])
    master_data_bp.add_url_rule(f"/{endpoint_prefix}/<int:item_id>/delete", view_func=login_required(delete_view), methods=["POST"])


_master_routes(Occupation, "Occupation", "occupations")
_master_routes(BmiCategory, "BMI Category", "bmi_categories")
