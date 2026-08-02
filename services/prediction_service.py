"""Patient-to-model adapter that preserves the existing prediction implementation."""

from models.predict_model import SleepDisorderPredictor

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = SleepDisorderPredictor()
        _predictor.load_models()
    return _predictor


def predict_patient(patient):
    """Map a Patient record to the existing predictor's exact input contract."""
    predictor = _get_predictor()
    if predictor.classifier is None or predictor.regressor is None:
        raise RuntimeError("The trained XGBoost models are not available. Train the model first.")
    input_data = {
        "Gender": patient.gender,
        "Age": patient.age,
        "Occupation": patient.occupation.name,
        "Sleep Duration": patient.sleep_duration,
        "Quality of Sleep": patient.quality_of_sleep,
        "Physical Activity Level": patient.physical_activity_level,
        "BMI Category": patient.bmi_category.name,
        "Heart Rate": patient.heart_rate,
        "Daily Steps": patient.daily_steps,
        "Systolic BP": patient.systolic_bp,
        "Diastolic BP": patient.diastolic_bp,
    }
    return predictor.make_comprehensive_prediction(input_data)


def get_feature_importance():
    """Return the active models' existing feature-importance output for the result UI."""
    return _get_predictor().get_feature_importance()
