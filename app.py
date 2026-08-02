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
from routes.users import users_bp
from services.database_seed import seed_database
from services.training_service import train_models_from_database
from utils.access import admin_required
from utils.timezone import jakarta_now

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "sleep-disorder-xgboost-app-2024")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{Path(app.root_path) / 'sleep_disorder.db'}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
login_manager.init_app(app)
app.register_blueprint(auth_bp)
app.register_blueprint(system_bp)
app.register_blueprint(training_dataset_bp)
app.register_blueprint(master_data_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(users_bp)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()
    seed_database()

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
                return render_template('predict.html', error="Model belum dilatih! Silakan latih model terlebih dahulu di halaman Evaluasi Model.")
            
            results = pred.make_comprehensive_prediction(input_data)
            
            print(f"[DEBUG] User Input: {input_data}")
            print(f"[DEBUG] Model Output: {results['sleep_disorder']}")
            
            return render_template('result.html', results=results, input_data=input_data)
            
        except Exception as e:
            return render_template('predict.html', error=f"Terjadi kesalahan: {str(e)}")
    
    return render_template('predict.html')


@app.route('/train', methods=['GET', 'POST'])
@admin_required
def train():
    """Model training and evaluation metrics page"""
    classification_results = None
    regression_results = None
    trained = False
    error = None
    training_record_count = TrainingDatasetRecord.query.count()
    model_metadata = db.session.get(ModelMetadata, 1)
    
    if request.method == 'POST':
        try:
            (trainer, clf_results, reg_results), training_record_count = train_models_from_database()
            if trainer is not None and clf_results is not None and reg_results is not None:
                classification_results = clf_results
                regression_results = reg_results
                trained = True
                current_version = model_metadata.model_version if model_metadata else "v0"
                try:
                    next_version = f"v{int(current_version.lstrip('v')) + 1}"
                except ValueError:
                    next_version = "v1"
                if model_metadata is None:
                    model_metadata = ModelMetadata(
                        id=1,
                        active_model="XGBoost Classifier and Regressor",
                        model_version=next_version,
                    )
                    db.session.add(model_metadata)
                else:
                    model_metadata.model_version = next_version
                model_metadata.last_training_date = jakarta_now()
                db.session.commit()
                global predictor
                predictor = None
            else:
                error = "Model training did not return complete classification and regression metrics."
        except Exception as e:
            error = f"Training failed: {str(e)}"
    
    return render_template('train.html', 
                         classification_results=classification_results,
                         regression_results=regression_results,
                         trained=trained,
                         error=error,
                         model_metadata=model_metadata,
                         training_record_count=training_record_count)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
# Trigger reload

# Trigger reload no occ
