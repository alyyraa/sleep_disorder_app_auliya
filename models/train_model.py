import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import joblib
import os
from utils.preprocessing import preprocess_pipeline, scale_features, split_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostPredictiveModels:
    def __init__(self):
        # Hanya inisialisasi XGBoost
        self.classifier = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        self.regressor = xgb.XGBRegressor(random_state=42)
        
        self.is_classifier_trained = False
        self.is_regressor_trained = False
        
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        
    def train_classification_model(self, X_train, X_test, y_train, y_test):
        """
        Train XGBoost classifier for sleep disorder prediction
        """
        logger.info("Training XGBoost Classifier...")
        
        # Train
        self.classifier.fit(X_train, y_train)
        self.is_classifier_trained = True
        
        # Predictions
        y_pred = self.classifier.predict(X_test)
        
        # Matriks Klasifikasi (Sesuai Permintaan)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"XGB Classifier Metrics -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return results
    
    def train_regression_model(self, X_train, X_test, y_train, y_test):
        """
        Train XGBoost regressor for stress level prediction
        """
        logger.info("Training XGBoost Regressor...")
        
        # Train
        self.regressor.fit(X_train, y_train)
        self.is_regressor_trained = True
        
        # Predictions
        y_pred = self.regressor.predict(X_test)
        
        # Matriks Regresi (Sesuai Permintaan)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"XGB Regressor Metrics -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return results
    
    def save_models(self, models_dir='models'):
        """
        Save the trained models and pre-processing artifacts
        """
        os.makedirs(models_dir, exist_ok=True)
        
        if self.is_classifier_trained:
            clf_path = os.path.join(models_dir, 'xgboost_classifier.joblib')
            joblib.dump(self.classifier, clf_path)
            logger.info(f"Classifier saved: {clf_path}")
            
        if self.is_regressor_trained:
            reg_path = os.path.join(models_dir, 'xgboost_regressor.joblib')
            joblib.dump(self.regressor, reg_path)
            logger.info(f"Regressor saved: {reg_path}")
            
        if self.scaler:
            scaler_path = os.path.join(models_dir, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            
        if self.label_encoders:
            le_path = os.path.join(models_dir, 'label_encoders.joblib')
            joblib.dump(self.label_encoders, le_path)
            
        if self.feature_names:
            fn_path = os.path.join(models_dir, 'feature_names.joblib')
            joblib.dump(self.feature_names, fn_path)


def train_models_pipeline(data_path="data"):
    """
    Complete model training pipeline focused only on XGBoost
    """
    logger.info("Starting XGBoost Model Training Pipeline")
    
    # Check data file
    data_dir = pd.io.common.Path(data_path)
    csv_files = list(data_dir.glob("*.csv")) if data_dir.exists() and data_dir.is_dir() else []
    
    if not csv_files:
        if os.path.isfile(data_path) and data_path.endswith('.csv'):
            file_to_load = data_path
        else:
            logger.error("Dataset not found!")
            return None, None, None
    else:
        file_to_load = str(csv_files[0])

    model_trainer = XGBoostPredictiveModels()
    processed_data = preprocess_pipeline(file_to_load)
    
    if processed_data is None:
        logger.error("Data preprocessing failed!")
        return None, None, None
        
    X = processed_data['features']
    y_class = processed_data['target_classification']
    y_reg = processed_data['target_regression']
    
    model_trainer.label_encoders = processed_data['label_encoders']
    model_trainer.feature_names = processed_data['feature_names']
    
    classification_results = None
    regression_results = None
    
    # Train Classification if data is available
    if y_class is not None:
        X_train, X_test, y_train, y_test = split_data(X, y_class)
        X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
        
        model_trainer.scaler = scaler
        classification_results = model_trainer.train_classification_model(X_train_s, X_test_s, y_train, y_test)
        
    # Train Regression if data is available
    if y_reg is not None:
        X_train_r, X_test_r, y_train_r, y_test_r = split_data(X, y_reg)
        if model_trainer.scaler is None:
            X_train_rs, X_test_rs, scaler = scale_features(X_train_r, X_test_r)
            model_trainer.scaler = scaler
        else:
            X_train_rs = model_trainer.scaler.transform(X_train_r)
            X_test_rs = model_trainer.scaler.transform(X_test_r)
            
        regression_results = model_trainer.train_regression_model(X_train_rs, X_test_rs, y_train_r, y_test_r)

    # Save Models
    model_trainer.save_models()
    
    return model_trainer, classification_results, regression_results

if __name__ == "__main__":
    train_models_pipeline()