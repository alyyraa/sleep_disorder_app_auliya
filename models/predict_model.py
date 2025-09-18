import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st
from utils.preprocessing import prepare_input_for_prediction

class SleepDisorderPredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.classification_models = {}
        self.regression_models = {}
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.class_names = None
        
    def load_models(self):
        """
        Load all trained models and preprocessing objects
        """
        try:
            # Load classification models
            classifier_files = {
                'Logistic Regression': 'logistic_regression_classifier.joblib',
                'Random Forest': 'random_forest_classifier.joblib',
                'XGBoost': 'xgboost_classifier.joblib'
            }
            
            for name, filename in classifier_files.items():
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    self.classification_models[name] = joblib.load(filepath)
            
            # Load regression models
            regressor_files = {
                'Linear Regression': 'linear_regression_regressor.joblib',
                'Random Forest': 'random_forest_regressor.joblib'
            }
            
            for name, filename in regressor_files.items():
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    self.regression_models[name] = joblib.load(filepath)
            
            # Load preprocessing objects
            scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            encoders_path = os.path.join(self.models_dir, 'label_encoders.joblib')
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
            
            features_path = os.path.join(self.models_dir, 'feature_names.joblib')
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
            
            # Set class names for sleep disorders
            if self.label_encoders and 'Sleep Disorder' in self.label_encoders:
                self.class_names = self.label_encoders['Sleep Disorder'].classes_
            
            return len(self.classification_models) > 0 or len(self.regression_models) > 0
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def predict_sleep_disorder(self, input_data, model_name='Random Forest'):
        """
        Predict sleep disorder using specified classification model
        """
        if model_name not in self.classification_models:
            available_models = list(self.classification_models.keys())
            if available_models:
                model_name = available_models[0]
                st.warning(f"Model '{model_name}' not found. Using '{available_models[0]}' instead.")
            else:
                st.error("No classification models available!")
                return None, None
        
        try:
            # Prepare input data
            processed_input = prepare_input_for_prediction(
                input_data, self.label_encoders, self.feature_names
            )
            
            # Scale the input
            if self.scaler:
                processed_input_scaled = self.scaler.transform(processed_input)
            else:
                processed_input_scaled = processed_input
            
            # Get model
            model = self.classification_models[model_name]
            
            # Make prediction
            prediction = model.predict(processed_input_scaled)[0]
            probabilities = model.predict_proba(processed_input_scaled)[0]
            
            # Convert prediction back to original label
            if self.class_names is not None:
                predicted_class = self.class_names[prediction]
            else:
                predicted_class = prediction
            
            # Create probability dictionary
            prob_dict = {}
            if self.class_names is not None:
                for i, class_name in enumerate(self.class_names):
                    prob_dict[class_name] = probabilities[i]
            else:
                for i, prob in enumerate(probabilities):
                    prob_dict[f'Class_{i}'] = prob
            
            return predicted_class, prob_dict
            
        except Exception as e:
            st.error(f"Error making sleep disorder prediction: {e}")
            return None, None
    
    def predict_stress_level(self, input_data, model_name='Random Forest'):
        """
        Predict stress level using specified regression model
        """
        if model_name not in self.regression_models:
            available_models = list(self.regression_models.keys())
            if available_models:
                model_name = available_models[0]
                st.warning(f"Model '{model_name}' not found. Using '{available_models[0]}' instead.")
            else:
                st.error("No regression models available!")
                return None
        
        try:
            # Prepare input data
            processed_input = prepare_input_for_prediction(
                input_data, self.label_encoders, self.feature_names
            )
            
            # Scale the input
            if self.scaler:
                processed_input_scaled = self.scaler.transform(processed_input)
            else:
                processed_input_scaled = processed_input
            
            # Get model
            model = self.regression_models[model_name]
            
            # Make prediction
            prediction = model.predict(processed_input_scaled)[0]
            
            # Ensure prediction is within reasonable bounds (1-10)
            prediction = max(1, min(10, prediction))
            
            return prediction
            
        except Exception as e:
            st.error(f"Error making stress level prediction: {e}")
            return None
    
    def calculate_sleep_quality_percentage(self, disorder_probabilities):
        """
        Calculate sleep quality percentage based on disorder probabilities
        Higher probability of 'None' (no disorder) = better sleep quality
        """
        if disorder_probabilities and 'None' in disorder_probabilities:
            # Sleep quality is based on probability of having no sleep disorder
            sleep_quality = disorder_probabilities['None'] * 100
        elif disorder_probabilities:
            # If 'None' not found, use inverse of highest disorder probability
            max_disorder_prob = max([prob for key, prob in disorder_probabilities.items() if key != 'None'])
            sleep_quality = (1 - max_disorder_prob) * 100
        else:
            sleep_quality = 50  # Default neutral value
        
        return round(sleep_quality, 1)
    
    def get_health_recommendations(self, predicted_disorder, stress_level, sleep_quality):
        """
        Generate health recommendations based on predictions
        """
        recommendations = []
        
        # Sleep disorder recommendations
        if predicted_disorder == 'Sleep Apnea':
            recommendations.extend([
                "🏥 Consult a sleep specialist for sleep apnea evaluation",
                "😴 Consider using a CPAP machine if prescribed",
                "🏃‍♂️ Maintain a healthy weight through regular exercise",
                "🚫 Avoid alcohol and sedatives before bedtime"
            ])
        elif predicted_disorder == 'Insomnia':
            recommendations.extend([
                "🛏️ Establish a consistent sleep schedule",
                "📱 Limit screen time 1 hour before bed",
                "☕ Avoid caffeine after 2 PM",
                "🧘‍♀️ Practice relaxation techniques before sleep"
            ])
        else:  # None or healthy sleep
            recommendations.extend([
                "✅ Great! Continue maintaining your healthy sleep habits",
                "🛏️ Keep your consistent sleep schedule",
                "🌙 Maintain good sleep hygiene practices"
            ])
        
        # Stress level recommendations
        if stress_level >= 7:
            recommendations.extend([
                "🧘‍♀️ Practice stress management techniques (meditation, yoga)",
                "🏃‍♂️ Engage in regular physical activity",
                "👥 Consider talking to a counselor or therapist",
                "⏰ Improve time management and work-life balance"
            ])
        elif stress_level >= 5:
            recommendations.extend([
                "🌿 Try relaxation techniques like deep breathing",
                "🚶‍♀️ Take regular breaks and short walks",
                "📚 Consider stress management workshops"
            ])
        else:
            recommendations.append("😌 Your stress levels appear manageable - keep it up!")
        
        # Sleep quality recommendations
        if sleep_quality < 60:
            recommendations.extend([
                "🌡️ Optimize your bedroom temperature (60-67°F)",
                "🔇 Ensure your bedroom is quiet and dark",
                "🛏️ Invest in a comfortable mattress and pillows"
            ])
        
        return recommendations
    
    def make_comprehensive_prediction(self, input_data, classifier_name='Random Forest', regressor_name='Random Forest'):
        """
        Make comprehensive predictions including disorder, stress level, and recommendations
        """
        results = {
            'sleep_disorder': None,
            'disorder_probabilities': None,
            'stress_level': None,
            'sleep_quality_percentage': None,
            'recommendations': []
        }
        
        # Predict sleep disorder
        if self.classification_models:
            disorder, probabilities = self.predict_sleep_disorder(input_data, classifier_name)
            results['sleep_disorder'] = disorder
            results['disorder_probabilities'] = probabilities
            results['sleep_quality_percentage'] = self.calculate_sleep_quality_percentage(probabilities)
        
        # Predict stress level
        if self.regression_models:
            stress = self.predict_stress_level(input_data, regressor_name)
            results['stress_level'] = stress
        
        # Generate recommendations
        results['recommendations'] = self.get_health_recommendations(
            results['sleep_disorder'],
            results['stress_level'] if results['stress_level'] else 5,
            results['sleep_quality_percentage'] if results['sleep_quality_percentage'] else 70
        )
        
        return results
    
    def get_available_models(self):
        """
        Get lists of available classification and regression models
        """
        return {
            'classifiers': list(self.classification_models.keys()),
            'regressors': list(self.regression_models.keys())
        }
    
    def model_info(self):
        """
        Get information about loaded models
        """
        info = {
            'classification_models_loaded': len(self.classification_models),
            'regression_models_loaded': len(self.regression_models),
            'scaler_loaded': self.scaler is not None,
            'encoders_loaded': self.label_encoders is not None,
            'feature_names_loaded': self.feature_names is not None
        }
        
        if self.feature_names:
            info['feature_count'] = len(self.feature_names)
            info['features'] = self.feature_names
        
        if self.class_names is not None:
            info['sleep_disorder_classes'] = list(self.class_names)
        
        return info

def create_sample_input():
    """
    Create sample input data for testing predictions
    """
    return {
        'Age': 35,
        'Gender': 'Male',
        'Sleep Duration': 7.5,
        'Quality of Sleep': 8,
        'Physical Activity Level': 75,
        'BMI Category': 'Normal',
        'Heart Rate': 70,
        'Daily Steps': 8000,
        'Systolic BP': 120,
        'Diastolic BP': 80
    }

def test_predictor(models_dir='models'):
    """
    Test the predictor with sample data
    """
    predictor = SleepDisorderPredictor(models_dir)
    
    if predictor.load_models():
        st.success("✅ Models loaded successfully!")
        
        # Display model info
        info = predictor.model_info()
        st.write("📊 Model Information:")
        st.json(info)
        
        # Test with sample data
        sample_input = create_sample_input()
        st.write("🧪 Testing with sample data:")
        st.json(sample_input)
        
        # Make predictions
        results = predictor.make_comprehensive_prediction(sample_input)
        st.write("🔮 Prediction Results:")
        st.json(results)
        
        return predictor
    else:
        st.error("❌ Failed to load models!")
        return None