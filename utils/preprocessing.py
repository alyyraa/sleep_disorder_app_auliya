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
    Clean the dataset by handling missing values, duplicates, and feature engineering
    """
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna()
    
    # Remove Person ID if exists (not a feature)
    if 'Person ID' in df.columns:
        df = df.drop('Person ID', axis=1)
    
    # Remove Occupation column (not used as prediction feature)
    if 'Occupation' in df.columns:
        df = df.drop('Occupation', axis=1)
    
    # === CRITICAL: Split 'Blood Pressure' into 'Systolic BP' and 'Diastolic BP' ===
    if 'Blood Pressure' in df.columns:
        bp_split = df['Blood Pressure'].str.split('/', expand=True)
        df['Systolic BP'] = pd.to_numeric(bp_split[0], errors='coerce')
        df['Diastolic BP'] = pd.to_numeric(bp_split[1], errors='coerce')
        df = df.drop('Blood Pressure', axis=1)
    
    # Drop any rows with NaN values after conversion
    df = df.dropna()
    
    return df

def encode_categorical_features(df):
    """
    Encode categorical features using Label Encoding
    """
    df_encoded = df.copy()
    
    # Only encode columns that actually exist and are categorical
    categorical_columns = ['Gender', 'BMI Category', 'Sleep Disorder']
    
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
    
    return df_encoded, label_encoders

def prepare_features_target(df_encoded):
    """
    Prepare features and target variables for both classification and regression
    """
    # Features for prediction (excluding target variables)
    feature_columns = ['Age', 'Gender', 'Sleep Duration', 'Quality of Sleep', 
                      'Physical Activity Level', 'BMI Category', 'Heart Rate', 
                      'Daily Steps', 'Systolic BP', 'Diastolic BP']
    
    # Filter only existing columns
    available_features = [col for col in feature_columns if col in df_encoded.columns]
    
    X = df_encoded[available_features]
    
    # Target for classification (Sleep Disorder)
    y_classification = df_encoded['Sleep Disorder'] if 'Sleep Disorder' in df_encoded.columns else None
    
    # Target for regression (Stress Level)
    y_regression = df_encoded['Stress Level'] if 'Stress Level' in df_encoded.columns else None
    
    return X, y_classification, y_regression, available_features

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    try:
        # Try stratified split first
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError as e:
        if "least populated class" in str(e) or "too few" in str(e):
            # Fall back to non-stratified split for small datasets
            print(f"Warning: Using non-stratified split due to small dataset size: {e}")
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            raise e

def preprocess_pipeline(file_path):
    """
    Complete preprocessing pipeline
    """
    # Load data
    df = load_data(file_path)
    if df is None:
        return None
    
    # Clean data (includes Blood Pressure splitting)
    df_clean = clean_data(df)
    
    # Encode categorical features
    df_encoded, label_encoders = encode_categorical_features(df_clean)
    
    # Prepare features and targets
    X, y_class, y_reg, feature_names = prepare_features_target(df_encoded)
    
    print(f"[INFO] Features used for training: {feature_names}")
    print(f"[INFO] Total samples after cleaning: {len(X)}")
    print(f"[INFO] Label encoders created for: {list(label_encoders.keys())}")
    
    if 'BMI Category' in label_encoders:
        print(f"[INFO] BMI categories in dataset: {list(label_encoders['BMI Category'].classes_)}")
    if 'Sleep Disorder' in label_encoders:
        print(f"[INFO] Sleep Disorder classes: {list(label_encoders['Sleep Disorder'].classes_)}")
    
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
    Prepare user input for prediction
    """
    # Create DataFrame from input
    df_input = pd.DataFrame([input_data])
    
    # Encode categorical features using saved encoders
    for col, encoder in label_encoders.items():
        if col in df_input.columns and col != 'Sleep Disorder':  # Don't encode target
            try:
                df_input[col] = encoder.transform(df_input[col].astype(str))
            except ValueError as e:
                # Handle unseen categories - use the most common class (0) 
                print(f"[WARNING] Unseen category in '{col}': {df_input[col].values}. Known: {list(encoder.classes_)}. Defaulting to 0.")
                df_input[col] = 0
    
    # Select only the features used in training
    df_input = df_input[feature_names]
    
    return df_input