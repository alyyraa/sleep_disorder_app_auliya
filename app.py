import sys
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.predict_model import SleepDisorderPredictor
from models.train_model import train_models_pipeline

app = Flask(__name__)
app.secret_key = 'sleep-disorder-xgboost-app-2024'

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
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction form and results"""
    if request.method == 'POST':
        try:
            # Parse form data — kolom sesuai dataset CSV (tanpa Occupation)
            input_data = {
                'Gender': request.form.get('gender', 'Male'),
                'Age': int(request.form.get('age', 35)),
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
def train():
    """Model training and evaluation metrics page"""
    classification_results = None
    regression_results = None
    trained = False
    error = None
    
    if request.method == 'POST':
        try:
            trainer, clf_results, reg_results = train_models_pipeline("data")
            if trainer is not None:
                classification_results = clf_results
                regression_results = reg_results
                trained = True
                # Reload the predictor with newly trained models
                global predictor
                predictor = None
            else:
                error = "Pelatihan model gagal! Pastikan dataset tersedia di folder data/."
        except Exception as e:
            error = f"Terjadi kesalahan saat pelatihan: {str(e)}"
    
    return render_template('train.html', 
                         classification_results=classification_results,
                         regression_results=regression_results,
                         trained=trained,
                         error=error)


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
# Trigger reload

# Trigger reload no occ
