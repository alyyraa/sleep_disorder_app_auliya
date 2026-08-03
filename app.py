import sys
import os
from pathlib import Path
from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import current_user

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.predict_model import SleepDisorderPredictor
from extensions import db, login_manager
from models.database import ModelMetadata, TrainingDatasetRecord, User
from routes.auth import auth_bp
from routes.system import system_bp
from routes.training_dataset import training_dataset_bp
from routes.master_data import master_data_bp
from routes.prediction import prediction_bp
from routes.reports import reports_bp
from routes.users import users_bp
from services.database_migration import ensure_database_schema
from services.database_seed import seed_database
from services.model_version_service import (
    activate_configured_model_version,
    activate_model_version,
    active_model_version,
    available_model_versions,
    ensure_active_model_version,
    ensure_version_dataset_archives,
    evaluation_for_version,
    train_and_register_version,
)
from services.prediction_service import reset_predictor_cache
from services.training_service import training_dataset_signature
from utils.access import admin_required

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "sleep-disorder-xgboost-app-2024")
database_path = Path(os.environ.get("DATABASE_PATH", Path(app.root_path) / "sleep_disorder.db")).expanduser().resolve()
database_path.parent.mkdir(parents=True, exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{database_path.as_posix()}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# Force browsers and reverse proxies to revalidate UI assets after deployment.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

db.init_app(app)
login_manager.init_app(app)
app.register_blueprint(auth_bp)
app.register_blueprint(system_bp)
app.register_blueprint(training_dataset_bp)
app.register_blueprint(master_data_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(reports_bp)
app.register_blueprint(users_bp)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()
    ensure_database_schema()
    seed_database()
    ensure_database_schema()
    activate_configured_model_version()
    ensure_active_model_version()
    ensure_version_dataset_archives()

# Global predictor instance
predictor = None

def get_predictor():
    """Load predictor if not already loaded"""
    global predictor
    if predictor is None:
        predictor = SleepDisorderPredictor()
        predictor.load_models()
    return predictor


@app.route('/')
def index():
    """Route visitors into the authenticated information system."""
    if current_user.is_authenticated:
        return redirect(url_for("system.dashboard"))
    return redirect(url_for("auth.login"))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction form and results"""
    flash("Manual prediction is disabled. Use New Prediction after selecting a patient record.", "info")
    return redirect(url_for("prediction.new_prediction"))

    if request.method == 'POST':
        try:
            # Parse form data — kolom sesuai dataset CSV (SEKARANG TERMASUK Occupation)
            input_data = {
                'Gender': request.form.get('gender', 'Male'),
                'Age': int(request.form.get('age', 35)),
                'Occupation': request.form.get('occupation', 'Engineer'),
                'Sleep Duration': float(request.form.get('sleep_duration', 7.5)),
                'Quality of Sleep': int(request.form.get('quality_of_sleep', 8)),
                'Physical Activity Level': int(request.form.get('physical_activity', 75)),
                'BMI Category': request.form.get('bmi_category', 'Normal'),
                'Heart Rate': int(request.form.get('heart_rate', 70)),
                'Daily Steps': int(request.form.get('daily_steps', 8000)),
                'Systolic BP': int(request.form.get('systolic_bp', 120)),
                'Diastolic BP': int(request.form.get('diastolic_bp', 80))
            }
            
            pred = get_predictor()
            
            if not pred.classifier and not pred.regressor:
                return render_template('predict.html', error="The model is not trained yet. Please train the model first.")
            
            results = pred.make_comprehensive_prediction(input_data)
            
            print(f"[DEBUG] User Input: {input_data}")
            print(f"[DEBUG] Model Output: {results['sleep_disorder']}")
            
            return render_template('result.html', results=results, input_data=input_data)
            
        except Exception as e:
            return render_template('predict.html', error=f"An error occurred: {str(e)}")
    
    return render_template('predict.html')


@app.route('/train', methods=['GET', 'POST'])
@admin_required
def train():
    """Model training and evaluation metrics page"""
    trained = False
    error = None
    notice = None
    training_record_count = TrainingDatasetRecord.query.count()
    model_metadata = db.session.get(ModelMetadata, 1)
    active_version = active_model_version(model_metadata)
    classification_results, regression_results = evaluation_for_version(active_version)
    dataset_changed = (
        model_metadata is None
        or model_metadata.training_dataset_signature != training_dataset_signature()
    )
    
    if request.method == 'POST':
        try:
            current_dataset_signature = training_dataset_signature()
            if model_metadata and model_metadata.training_dataset_signature == current_dataset_signature:
                notice = "The Training Dataset has not changed since the last successful training. The active model version and its evaluation metrics were kept."
            else:
                active_version, _, _, training_record_count = train_and_register_version(
                    current_dataset_signature
                )
                trained = True
                global predictor
                predictor = None
                reset_predictor_cache()
                model_metadata = db.session.get(ModelMetadata, 1)
                classification_results, regression_results = evaluation_for_version(active_version)
        except Exception as e:
            error = f"Training failed: {str(e)}"

    model_metadata = db.session.get(ModelMetadata, 1)
    active_version = active_model_version(model_metadata)
    dataset_changed = (
        model_metadata is None
        or model_metadata.training_dataset_signature != training_dataset_signature()
    )
    if not trained:
        classification_results, regression_results = evaluation_for_version(active_version)
    
    return render_template('train.html', 
                         classification_results=classification_results,
                         regression_results=regression_results,
                         trained=trained,
                         error=error,
                         notice=notice,
                         model_metadata=model_metadata,
                         active_version=active_version,
                         model_versions=available_model_versions(),
                         dataset_changed=dataset_changed,
                         training_record_count=training_record_count)


@app.post('/train/versions/<int:version_id>/activate')
@admin_required
def activate_trained_model(version_id):
    """Restore an archived model and its exact Training Dataset without retraining."""
    try:
        version, changed = activate_model_version(version_id)
        if changed:
            global predictor
            predictor = None
            reset_predictor_cache()
            flash(f"Model version {version.version} and its Training Dataset were restored successfully.", "success")
        else:
            flash(f"Model version {version.version} is already active.", "info")
    except Exception as error:
        flash(f"Model activation failed: {error}", "danger")
    return redirect(url_for("train"))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
# Trigger reload

# Trigger reload no occ
