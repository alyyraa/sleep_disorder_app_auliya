import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import xgboost as xgb
import joblib
import os
import streamlit as st
from utils.preprocessing import preprocess_pipeline, scale_features, split_data

class SleepDisorderModels:
    def __init__(self):
        self.classification_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        }
        
        self.regression_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)
        }
        
        self.trained_classifiers = {}
        self.trained_regressors = {}
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        
    def train_classification_models(self, X_train, X_test, y_train, y_test):
        """
        Train classification models for sleep disorder prediction
        """
        classification_results = {}
        
        for name, model in self.classification_models.items():
            st.write(f"Training {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score with dynamic fold calculation
            n_samples = len(X_train)
            cv_folds = min(5, max(2, n_samples // 2))
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            
            classification_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Store trained model
            self.trained_classifiers[name] = model
            
            st.write(f"✅ {name} - Accuracy: {accuracy:.4f} (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
        
        return classification_results
    
    def train_regression_models(self, X_train, X_test, y_train, y_test):
        """
        Train regression models for stress level prediction
        """
        regression_results = {}
        
        for name, model in self.regression_models.items():
            st.write(f"Training {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score with dynamic fold calculation
            n_samples = len(X_train)
            cv_folds = min(5, max(2, n_samples // 2))
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            regression_results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_rmse': cv_rmse,
                'predictions': y_pred
            }
            
            # Store trained model
            self.trained_regressors[name] = model
            
            st.write(f"✅ {name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return regression_results
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='classification'):
        """
        Perform hyperparameter tuning for selected models
        """
        # Determine appropriate number of CV folds based on dataset size
        n_samples = len(X_train)
        cv_folds = min(3, max(2, n_samples // 2))  # Use 2-3 folds, but not more than half the samples
        
        st.write(f"Using {cv_folds}-fold cross-validation for {n_samples} samples")
        
        if model_type == 'classification':
            # Skip hyperparameter tuning if dataset is too small
            if n_samples < 10:
                st.warning("Dataset too small for hyperparameter tuning. Using default parameters.")
                return {"message": "Skipped due to small dataset"}
            
            # Random Forest hyperparameter tuning
            rf_params = {
                'n_estimators': [50, 100],  # Reduced options for small datasets
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            }
            
            rf_grid = GridSearchCV(
                RandomForestClassifier(random_state=42),
                rf_params,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1
            )
            
            rf_grid.fit(X_train, y_train)
            self.classification_models['Random Forest'] = rf_grid.best_estimator_
            
            return rf_grid.best_params_
        
        else:  # regression
            # Skip hyperparameter tuning if dataset is too small
            if n_samples < 10:
                st.warning("Dataset too small for hyperparameter tuning. Using default parameters.")
                return {"message": "Skipped due to small dataset"}
            
            # Random Forest hyperparameter tuning
            rf_params = {
                'n_estimators': [50, 100],  # Reduced options for small datasets
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            }
            
            rf_grid = GridSearchCV(
                RandomForestRegressor(random_state=42),
                rf_params,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            rf_grid.fit(X_train, y_train)
            self.regression_models['Random Forest'] = rf_grid.best_estimator_
            
            return rf_grid.best_params_
    
    def save_models(self, models_dir='models'):
        """
        Save trained models to disk
        """
        os.makedirs(models_dir, exist_ok=True)
        
        # Save classification models
        for name, model in self.trained_classifiers.items():
            filename = f"{models_dir}/{name.lower().replace(' ', '_')}_classifier.joblib"
            joblib.dump(model, filename)
            st.write(f"💾 Saved {name} classifier to {filename}")
        
        # Save regression models
        for name, model in self.trained_regressors.items():
            filename = f"{models_dir}/{name.lower().replace(' ', '_')}_regressor.joblib"
            joblib.dump(model, filename)
            st.write(f"💾 Saved {name} regressor to {filename}")
        
        # Save preprocessing objects
        if self.scaler:
            joblib.dump(self.scaler, f"{models_dir}/scaler.joblib")
            st.write(f"💾 Saved scaler to {models_dir}/scaler.joblib")
        
        if self.label_encoders:
            joblib.dump(self.label_encoders, f"{models_dir}/label_encoders.joblib")
            st.write(f"💾 Saved label encoders to {models_dir}/label_encoders.joblib")
        
        if self.feature_names:
            joblib.dump(self.feature_names, f"{models_dir}/feature_names.joblib")
            st.write(f"💾 Saved feature names to {models_dir}/feature_names.joblib")
    
    def load_models(self, models_dir='models'):
        """
        Load trained models from disk
        """
        try:
            # Load classification models
            for name in self.classification_models.keys():
                filename = f"{models_dir}/{name.lower().replace(' ', '_')}_classifier.joblib"
                if os.path.exists(filename):
                    self.trained_classifiers[name] = joblib.load(filename)
            
            # Load regression models
            for name in self.regression_models.keys():
                filename = f"{models_dir}/{name.lower().replace(' ', '_')}_regressor.joblib"
                if os.path.exists(filename):
                    self.trained_regressors[name] = joblib.load(filename)
            
            # Load preprocessing objects
            scaler_path = f"{models_dir}/scaler.joblib"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            encoders_path = f"{models_dir}/label_encoders.joblib"
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
            
            features_path = f"{models_dir}/feature_names.joblib"
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False

def train_models_pipeline(data_path, perform_tuning=False):
    """
    Complete model training pipeline
    """
    st.header("🤖 Model Training Pipeline")
    
    # Initialize model trainer
    model_trainer = SleepDisorderModels()
    
    # Preprocess data
    st.subheader("📊 Data Preprocessing")
    with st.spinner("Preprocessing data..."):
        processed_data = preprocess_pipeline(data_path)
    
    if processed_data is None:
        st.error("Failed to load and preprocess data!")
        return None
    
    st.success("✅ Data preprocessing completed!")
    
    # Extract data components
    X = processed_data['features']
    y_class = processed_data['target_classification']
    y_reg = processed_data['target_regression']
    
    # Store preprocessing objects
    model_trainer.label_encoders = processed_data['label_encoders']
    model_trainer.feature_names = processed_data['feature_names']
    
    # Split data for classification
    if y_class is not None:
        st.subheader("🎯 Classification Models Training")
        X_train_class, X_test_class, y_train_class, y_test_class = split_data(X, y_class)
        
        # Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train_class, X_test_class)
        model_trainer.scaler = scaler
        
        # Hyperparameter tuning (optional)
        if perform_tuning:
            st.write("🔧 Performing hyperparameter tuning...")
            best_params = model_trainer.hyperparameter_tuning(X_train_scaled, y_train_class, 'classification')
            st.write(f"Best parameters: {best_params}")
        
        # Train classification models
        classification_results = model_trainer.train_classification_models(
            X_train_scaled, X_test_scaled, y_train_class, y_test_class
        )
        
        # Display classification results
        st.subheader("📈 Classification Results")
        results_df = pd.DataFrame({
            'Model': list(classification_results.keys()),
            'Accuracy': [results['accuracy'] for results in classification_results.values()],
            'CV Mean': [results['cv_mean'] for results in classification_results.values()],
            'CV Std': [results['cv_std'] for results in classification_results.values()]
        })
        st.dataframe(results_df)
    
    # Split data for regression
    if y_reg is not None:
        st.subheader("📊 Regression Models Training")
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(X, y_reg)
        
        # Scale features (use same scaler if available)
        if model_trainer.scaler is None:
            X_train_scaled, X_test_scaled, scaler = scale_features(X_train_reg, X_test_reg)
            model_trainer.scaler = scaler
        else:
            X_train_scaled = model_trainer.scaler.transform(X_train_reg)
            X_test_scaled = model_trainer.scaler.transform(X_test_reg)
        
        # Hyperparameter tuning (optional)
        if perform_tuning:
            st.write("🔧 Performing hyperparameter tuning for regression...")
            best_params = model_trainer.hyperparameter_tuning(X_train_scaled, y_train_reg, 'regression')
            st.write(f"Best parameters: {best_params}")
        
        # Train regression models
        regression_results = model_trainer.train_regression_models(
            X_train_scaled, X_test_scaled, y_train_reg, y_test_reg
        )
        
        # Display regression results
        st.subheader("📈 Regression Results")
        results_df = pd.DataFrame({
            'Model': list(regression_results.keys()),
            'RMSE': [results['rmse'] for results in regression_results.values()],
            'MAE': [results['mae'] for results in regression_results.values()],
            'R²': [results['r2'] for results in regression_results.values()]
        })
        st.dataframe(results_df)
    
    # Save models
    st.subheader("💾 Saving Models")
    model_trainer.save_models()
    
    st.success("🎉 Model training completed successfully!")
    
    return model_trainer, classification_results if y_class is not None else None, regression_results if y_reg is not None else None