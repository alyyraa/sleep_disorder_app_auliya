import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the sleep disorder dataset
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Clean the dataset:
    - Remove Person ID (bukan fitur)
    - Split Blood Pressure menjadi Systolic BP dan Diastolic BP
    - Handle missing values dan duplicates
    """
    df = df.copy()
    
    # Isi nilai kosong (NaN) di kolom Sleep Disorder dengan string 'None' karena 'None' di-read sebagai NaN oleh pandas
    if 'Sleep Disorder' in df.columns:
        df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values untuk kolom lain
    df = df.dropna()
    
    # Remove Person ID (bukan fitur prediksi)
    if 'Person ID' in df.columns:
        df = df.drop('Person ID', axis=1)
    
    # === Split 'Blood Pressure' (string "132/87") menjadi 2 kolom numerik ===
    if 'Blood Pressure' in df.columns:
        bp_split = df['Blood Pressure'].str.split('/', expand=True)
        df['Systolic BP'] = pd.to_numeric(bp_split[0], errors='coerce')
        df['Diastolic BP'] = pd.to_numeric(bp_split[1], errors='coerce')
        df = df.drop('Blood Pressure', axis=1)
    
    # Drop NaN setelah konversi
    df = df.dropna()
    
    return df

def encode_categorical_features(df):
    """
    Encode semua kolom kategorikal
    """
    df_encoded = df.copy()
    
    # Semua kolom kategorikal TERMASUK Occupation
    categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            print(f"[INFO] Encoded '{col}': {list(le.classes_)}")
    
    return df_encoded, label_encoders

def prepare_features_target(df_encoded):
    """
    Siapkan fitur dan target — fitur HARUS cocok dengan kolom dataset
    """
    # Semua kolom fitur TERMASUK Occupation
    feature_columns = [
        'Gender', 'Age', 'Occupation',
        'Sleep Duration', 'Quality of Sleep',
        'Physical Activity Level',
        'BMI Category',
        'Heart Rate', 'Daily Steps',
        'Systolic BP', 'Diastolic BP'
    ]
    
    # Filter hanya kolom yang ada
    available_features = [col for col in feature_columns if col in df_encoded.columns]
    
    X = df_encoded[available_features]
    
    # Target klasifikasi (Sleep Disorder)
    y_classification = df_encoded['Sleep Disorder'] if 'Sleep Disorder' in df_encoded.columns else None
    
    # Target regresi (Stress Level)
    y_regression = df_encoded['Stress Level'] if 'Stress Level' in df_encoded.columns else None
    
    print(f"[INFO] Features used ({len(available_features)}): {available_features}")
    
    return X, y_classification, y_regression, available_features

def scale_features(X_train, X_test):
    """
    Scaling menggunakan StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Bagi data training dan testing
    """
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError as e:
        if "least populated class" in str(e) or "too few" in str(e):
            print(f"Warning: Using non-stratified split: {e}")
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            raise e

def preprocess_pipeline(file_path):
    """
    Pipeline preprocessing lengkap
    """
    df = load_data(file_path)
    if df is None:
        return None
    
    print(f"[INFO] Raw dataset columns: {list(df.columns)}")
    print(f"[INFO] Raw dataset shape: {df.shape}")
    
    df_clean = clean_data(df)
    print(f"[INFO] After cleaning shape: {df_clean.shape}")
    print(f"[INFO] Cleaned columns: {list(df_clean.columns)}")
    
    df_encoded, label_encoders = encode_categorical_features(df_clean)
    
    X, y_class, y_reg, feature_names = prepare_features_target(df_encoded)
    
    return {
        'data': df_clean,
        'encoded_data': df_encoded,
        'features': X,
        'target_classification': y_class,
        'target_regression': y_reg,
        'label_encoders': label_encoders,
        'feature_names': feature_names
    }

def prepare_input_for_prediction(input_data, label_encoders, feature_names):
    """
    Siapkan input user untuk prediksi
    """
    df_input = pd.DataFrame([input_data])
    
    # Encode semua kolom kategorikal dengan encoder yang tersimpan
    for col, encoder in label_encoders.items():
        if col in df_input.columns and col != 'Sleep Disorder':
            try:
                df_input[col] = encoder.transform(df_input[col].astype(str))
            except ValueError as e:
                print(f"[WARNING] Unseen category in '{col}': {df_input[col].values}. Known: {list(encoder.classes_)}")
                df_input[col] = 0
    
    # Pilih hanya fitur yang digunakan saat training
    available = [f for f in feature_names if f in df_input.columns]
    df_input = df_input[available]
    
    return df_input