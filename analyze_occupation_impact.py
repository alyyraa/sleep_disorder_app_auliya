from models.predict_model import SleepDisorderPredictor
import pandas as pd

# Load predictor dan cek feature importance saat ini
predictor = SleepDisorderPredictor()
predictor.load_models()

# Cek feature importance untuk occupation
print('=== FEATURE IMPORTANCE SAAT INI ===')
test_data = {
    'Gender': 'Female',
    'Age': 30,
    'Occupation': 'Nurse',
    'Sleep Duration': 6.4,
    'Quality of Sleep': 5,
    'Physical Activity Level': 40,
    'BMI Category': 'Normal Weight',
    'Heart Rate': 78,
    'Daily Steps': 4000,
    'Systolic BP': 132,
    'Diastolic BP': 87
}

results = predictor.make_comprehensive_prediction(test_data)
print('Classifier Feature Importance:')
for item in results['feature_importance']['classifier_importance']:
    print(f'  {item["feature"]}: {item["score"]:.4f} ({item["percentage"]:.1f}%)')

print()
print('Regressor Feature Importance:')
for item in results['feature_importance']['regressor_importance']:
    print(f'  {item["feature"]}: {item["score"]:.4f} ({item["percentage"]:.1f}%)')

# Cari posisi occupation
print()
print('POSISI OCCUPATION DALAM FEATURE IMPORTANCE:')
clf_occupation = next((item for item in results['feature_importance']['classifier_importance'] if item['feature'] == 'Occupation'), None)
reg_occupation = next((item for item in results['feature_importance']['regressor_importance'] if item['feature'] == 'Occupation'), None)

if clf_occupation:
    print(f'Classifier - Occupation: #{results["feature_importance"]["classifier_importance"].index(clf_occupation)+1}, score: {clf_occupation["score"]:.4f}')
if reg_occupation:
    print(f'Regressor - Occupation: #{results["feature_importance"]["regressor_importance"].index(reg_occupation)+1}, score: {reg_occupation["score"]:.4f}')