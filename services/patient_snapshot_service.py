"""Create immutable patient snapshots for prediction history."""


def patient_snapshot(patient):
    """Return the complete patient view used at prediction time."""
    return {
        "full_name": patient.full_name,
        "gender": patient.gender,
        "age": patient.age,
        "occupation": patient.occupation.name,
        "sleep_duration": patient.sleep_duration,
        "quality_of_sleep": patient.quality_of_sleep,
        "physical_activity_level": patient.physical_activity_level,
        "daily_steps": patient.daily_steps,
        "bmi_category": patient.bmi_category.name,
        "heart_rate": patient.heart_rate,
        "systolic_bp": patient.systolic_bp,
        "diastolic_bp": patient.diastolic_bp,
    }
