"""Training Dataset CRUD and CSV import routes."""

import io

import pandas as pd
from flask import Blueprint, flash, redirect, render_template, request, url_for

from extensions import db
from models.database import BmiCategory, Occupation, TrainingDatasetRecord
from utils.access import admin_required

training_dataset_bp = Blueprint("training_dataset", __name__, url_prefix="/master-data/training-dataset")

REQUIRED_COLUMNS = {"Person ID", "Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "BMI Category", "Blood Pressure", "Heart Rate", "Daily Steps", "Sleep Disorder"}


def _choices():
    return Occupation.query.order_by(Occupation.name).all(), BmiCategory.query.order_by(BmiCategory.name).all()


def _record_values(source=None):
    source = source if source is not None else request.form
    try:
        person_id = source.get("source_person_id", "")
        values = {"source_person_id": int(person_id) if str(person_id).strip() else None, "gender": str(source.get("gender", "")).strip(), "age": int(source.get("age", "")), "occupation_id": int(source.get("occupation_id", "")), "sleep_duration": float(source.get("sleep_duration", "")), "quality_of_sleep": int(source.get("quality_of_sleep", "")), "physical_activity_level": int(source.get("physical_activity_level", "")), "stress_level": float(source.get("stress_level", "")), "bmi_category_id": int(source.get("bmi_category_id", "")), "systolic_bp": int(source.get("systolic_bp", "")), "diastolic_bp": int(source.get("diastolic_bp", "")), "heart_rate": int(source.get("heart_rate", "")), "daily_steps": int(source.get("daily_steps", "")), "sleep_disorder": str(source.get("sleep_disorder", "")).strip()}
    except (TypeError, ValueError):
        return None, "All Training Dataset values must be valid numbers."
    if values["gender"] not in {"Male", "Female"} or values["sleep_disorder"] not in {"None", "Insomnia", "Sleep Apnea"}:
        return None, "Select a valid Gender and Sleep Disorder value."
    if not 1 <= values["age"] <= 120 or not 0 < values["sleep_duration"] <= 24 or not 1 <= values["quality_of_sleep"] <= 10:
        return None, "Age, sleep duration, or sleep quality is outside the allowed range."
    if not 0 <= values["physical_activity_level"] <= 100 or not 1 <= values["stress_level"] <= 10:
        return None, "Physical activity or stress level is outside the allowed range."
    if min(values["systolic_bp"], values["diastolic_bp"], values["heart_rate"], values["daily_steps"]) < 0:
        return None, "Blood pressure, heart rate, and daily steps must be non-negative."
    if not db.session.get(Occupation, values["occupation_id"]) or not db.session.get(BmiCategory, values["bmi_category_id"]):
        return None, "Select a valid Occupation and BMI Category."
    return values, None


@training_dataset_bp.get("/")
@admin_required
def index():
    records = TrainingDatasetRecord.query.order_by(TrainingDatasetRecord.id.desc()).all()
    return render_template("training_dataset/index.html", records=records)


@training_dataset_bp.route("/create", methods=["GET", "POST"])
@admin_required
def create():
    occupations, bmi_categories = _choices()
    if request.method == "POST":
        values, error = _record_values()
        if error: flash(error, "danger")
        else:
            db.session.add(TrainingDatasetRecord(**values)); db.session.commit(); flash("Training Dataset record created successfully.", "success")
            return redirect(url_for("training_dataset.index"))
    return render_template("training_dataset/form.html", record=None, occupations=occupations, bmi_categories=bmi_categories)


@training_dataset_bp.route("/<int:record_id>/edit", methods=["GET", "POST"])
@admin_required
def edit(record_id):
    record = db.get_or_404(TrainingDatasetRecord, record_id); occupations, bmi_categories = _choices()
    if request.method == "POST":
        values, error = _record_values()
        if error: flash(error, "danger")
        else:
            for key, value in values.items(): setattr(record, key, value)
            db.session.commit(); flash("Training Dataset record updated successfully.", "success")
            return redirect(url_for("training_dataset.index"))
    return render_template("training_dataset/form.html", record=record, occupations=occupations, bmi_categories=bmi_categories)


@training_dataset_bp.post("/<int:record_id>/delete")
@admin_required
def delete(record_id):
    db.session.delete(db.get_or_404(TrainingDatasetRecord, record_id)); db.session.commit(); flash("Training Dataset record deleted successfully.", "success")
    return redirect(url_for("training_dataset.index"))


@training_dataset_bp.post("/import")
@admin_required
def import_csv():
    uploaded_file = request.files.get("csv_file")
    if not uploaded_file or not uploaded_file.filename.lower().endswith(".csv"):
        flash("Select a valid CSV file.", "danger"); return redirect(url_for("training_dataset.index"))
    try:
        dataset = pd.read_csv(io.BytesIO(uploaded_file.read()))
    except Exception:
        flash("The CSV file could not be read.", "danger"); return redirect(url_for("training_dataset.index"))
    if not REQUIRED_COLUMNS.issubset(dataset.columns):
        flash("CSV columns do not match the original research dataset structure.", "danger"); return redirect(url_for("training_dataset.index"))
    occupations = {item.name: item.id for item in Occupation.query.all()}; bmi_categories = {item.name: item.id for item in BmiCategory.query.all()}; new_records = []
    try:
        for row_number, (_, row) in enumerate(dataset.iterrows(), start=2):
            systolic_bp, diastolic_bp = [int(value) for value in str(row["Blood Pressure"]).split("/", 1)]
            occupation_name, bmi_name = str(row["Occupation"]), str(row["BMI Category"])
            if occupation_name not in occupations or bmi_name not in bmi_categories: raise ValueError(f"Row {row_number}: Occupation or BMI Category is not available in Master Data.")
            payload = {"source_person_id": int(row["Person ID"]), "gender": str(row["Gender"]), "age": int(row["Age"]), "occupation_id": occupations[occupation_name], "sleep_duration": float(row["Sleep Duration"]), "quality_of_sleep": int(row["Quality of Sleep"]), "physical_activity_level": int(row["Physical Activity Level"]), "stress_level": float(row["Stress Level"]), "bmi_category_id": bmi_categories[bmi_name], "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp, "heart_rate": int(row["Heart Rate"]), "daily_steps": int(row["Daily Steps"]), "sleep_disorder": "None" if pd.isna(row["Sleep Disorder"]) else str(row["Sleep Disorder"])}
            values, error = _record_values(payload)
            if error: raise ValueError(f"Row {row_number}: {error}")
            new_records.append(TrainingDatasetRecord(**values))
        if not new_records: raise ValueError("CSV does not contain any records.")
        db.session.add_all(new_records); db.session.commit(); flash(f"Successfully imported {len(new_records)} Training Dataset records.", "success")
    except (TypeError, ValueError) as error:
        db.session.rollback(); flash(str(error), "danger")
    return redirect(url_for("training_dataset.index"))
