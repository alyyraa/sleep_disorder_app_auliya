import pandas as pd
from models.predict_model import SleepDisorderPredictor

# Test prediction with Occupation
predictor = SleepDisorderPredictor()
predictor.load_models()

# Test case 1: Nurse with Sleep Apnea pattern (low quality sleep, low activity)
test_nurse = {
    'Gender': 'Female',
    'Age': 30,
    'Occupation': 'Nurse',
    'Sleep Duration': 6.1,  # Similar to dataset
    'Quality of Sleep': 5,   # Low quality = Sleep Apnea pattern
    'Physical Activity Level': 40,  # Low activity
    'BMI Category': 'Normal Weight',
    'Heart Rate': 80,
    'Daily Steps': 4000,
    'Systolic BP': 132,
    'Diastolic BP': 87
}

# Test case 2: Salesperson (Insomnia pattern from dataset)
test_sales = {
    'Gender': 'Male',
    'Age': 35,
    'Occupation': 'Salesperson',
    'Sleep Duration': 6.4,  # Average from dataset
    'Quality of Sleep': 6,   # From dataset
    'Physical Activity Level': 45,  # From dataset
    'BMI Category': 'Normal',
    'Heart Rate': 72,        # From dataset
    'Daily Steps': 6000,     # From dataset
    'Systolic BP': 130,      # From dataset
    'Diastolic BP': 85       # From dataset
}

print('=== TEST PREDICTIONS WITH OCCUPATION ===')
print('\nTest Case 1 - Nurse:')
result1 = predictor.make_comprehensive_prediction(test_nurse)
print(f'Predicted Disorder: {result1["sleep_disorder"]}')
print(f'Probabilities: {result1["disorder_probabilities"]}')

print('\nTest Case 2 - Salesperson:')
result2 = predictor.make_comprehensive_prediction(test_sales)
print(f'Predicted Disorder: {result2["sleep_disorder"]}')
print(f'Probabilities: {result2["disorder_probabilities"]}')