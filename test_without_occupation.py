import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, XGBRegressor
import joblib
import os

# Load dataset
df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')

# Preprocessing seperti di utils/preprocessing.py tapi TANPA occupation
def preprocess_data_without_occupation(df):
    # Copy dataframe
    df_processed = df.copy()

    # Handle Blood Pressure
    df_processed[['Systolic BP', 'Diastolic BP']] = df_processed['Blood Pressure'].str.split('/', expand=True).astype(int)

    # Encode categorical variables (TANPA Occupation)
    le_gender = LabelEncoder()
    le_bmi = LabelEncoder()

    df_processed['Gender'] = le_gender.fit_transform(df_processed['Gender'])
    df_processed['BMI Category'] = le_bmi.fit_transform(df_processed['BMI Category'])

    # Map Sleep Disorder to numbers - handle missing values
    disorder_mapping = {'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2}
    df_processed['Sleep Disorder'] = df_processed['Sleep Disorder'].map(disorder_mapping)

    # Hanya gunakan data yang memiliki sleep disorder (Insomnia/Sleep Apnea)
    disorder_data = df_processed[df_processed['Sleep Disorder'].notna()].copy()

    # Remap classes: Insomnia = 0, Sleep Apnea = 1 (karena tidak ada None)
    disorder_data['Sleep Disorder'] = disorder_data['Sleep Disorder'].map({1: 0, 2: 1})

    # Features TANPA Occupation
    features = ['Gender', 'Age', 'Sleep Duration', 'Quality of Sleep',
               'Physical Activity Level', 'BMI Category', 'Heart Rate',
               'Daily Steps', 'Systolic BP', 'Diastolic BP']

    X = disorder_data[features]
    y_clf = disorder_data['Sleep Disorder'].astype(int)  # Convert to int
    y_reg = disorder_data['Stress Level']

    return X, y_clf, y_reg, features

print('=== MELATIH MODEL TANPA OCCUPATION ===')

# Preprocess data tanpa occupation
X, y_clf, y_reg, features = preprocess_data_without_occupation(df)

print(f'Features tanpa occupation: {features}')
print(f'Jumlah features: {len(features)} (vs 11 sebelumnya)')

# Split data
X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier tanpa occupation
print('\nMelatih XGBoost Classifier tanpa occupation...')
clf = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)
clf.fit(X_train_scaled, y_clf_train)

# Train regressor tanpa occupation
print('Melatih XGBoost Regressor tanpa occupation...')
reg = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='rmse'
)
reg.fit(X_train_scaled, y_reg_train)

# Evaluate
clf_pred = clf.predict(X_test_scaled)
clf_accuracy = accuracy_score(y_clf_test, clf_pred)
print(f'\nAccuracy Classifier tanpa occupation: {clf_accuracy:.4f}')

reg_pred = reg.predict(X_test_scaled)
reg_mae = np.mean(np.abs(reg_pred - y_reg_test))
print(f'MAE Regressor tanpa occupation: {reg_mae:.4f}')

# Bandingkan dengan model saat ini (dengan occupation)
print('\n=== MEMUAT MODEL SAAT INI (DENGAN OCCUPATION) ===')
from models.predict_model import SleepDisorderPredictor
predictor_with_occ = SleepDisorderPredictor()
predictor_with_occ.load_models()

# Test pada sample yang sama
test_data = {
    'Gender': 'Female',
    'Age': 30,
    'Occupation': 'Nurse',  # Ini akan diabaikan untuk model tanpa occupation
    'Sleep Duration': 6.4,
    'Quality of Sleep': 5,
    'Physical Activity Level': 40,
    'BMI Category': 'Normal Weight',
    'Heart Rate': 78,
    'Daily Steps': 4000,
    'Systolic BP': 132,
    'Diastolic BP': 87
}

# Preprocess test data tanpa occupation
test_df = pd.DataFrame([test_data])
test_df['Blood Pressure'] = f"{test_data['Systolic BP']}/{test_data['Diastolic BP']}"  # Add Blood Pressure column
test_df[['Systolic BP', 'Diastolic BP']] = test_df['Blood Pressure'].str.split('/', expand=True).astype(int)

# Load encoders dari model saat ini
label_encoders = joblib.load('models/label_encoders.joblib')
scaler_current = joblib.load('models/scaler.joblib')

test_processed = test_df.copy()
test_processed['Gender'] = label_encoders['Gender'].transform(test_processed['Gender'])
test_processed['BMI Category'] = label_encoders['BMI Category'].transform(test_processed['BMI Category'])
test_processed['Occupation'] = label_encoders['Occupation'].transform(test_processed['Occupation'])  # Untuk model dengan occupation

# Features dengan dan tanpa occupation
features_with_occ = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
                    'Physical Activity Level', 'BMI Category', 'Heart Rate',
                    'Daily Steps', 'Systolic BP', 'Diastolic BP']
features_without_occ = [f for f in features_with_occ if f != 'Occupation']

# Test model dengan occupation
X_test_with_occ = scaler_current.transform(test_processed[features_with_occ])
pred_with_occ = predictor_with_occ.classifier.predict(X_test_with_occ)[0]
stress_with_occ = predictor_with_occ.regressor.predict(X_test_with_occ)[0]

# Test model tanpa occupation
X_test_without_occ = scaler.transform(test_processed[features_without_occ])
pred_without_occ_raw = clf.predict(X_test_without_occ)[0]
stress_without_occ = reg.predict(X_test_without_occ)[0]

# Convert prediction back to original mapping (0=Insomnia, 1=Sleep Apnea)
pred_without_occ = 1 if pred_without_occ_raw == 0 else 2  # 0->Insomnia(1), 1->Sleep Apnea(2)

print('\n=== PERBANDINGAN PREDIKSI ===')
disorder_map = {0: 'None', 1: 'Insomnia', 2: 'Sleep Apnea'}
print(f'Dengan Occupation: {disorder_map[pred_with_occ]}, Stress: {stress_with_occ:.1f}')
print(f'Tanpa Occupation: {disorder_map[pred_without_occ]}, Stress: {stress_without_occ:.1f}')

# Feature importance tanpa occupation
print('\n=== FEATURE IMPORTANCE TANPA OCCUPATION ===')
print('Classifier:')
clf_importance = clf.feature_importances_
for feat, imp in zip(features_without_occ, clf_importance):
    print(f'  {feat}: {imp:.4f} ({imp/np.max(clf_importance)*100:.1f}%)')

print('Regressor:')
reg_importance = reg.feature_importances_
for feat, imp in zip(features_without_occ, reg_importance):
    print(f'  {feat}: {imp:.4f} ({imp/np.max(reg_importance)*100:.1f}%)')