import pandas as pd
import numpy as np
import joblib
import os
import logging
from utils.preprocessing import prepare_input_for_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SleepDisorderPredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.class_names = None
        
    def load_models(self):
        """
        Load trained XGBoost models and preprocessing objects
        """
        try:
            # Load XGBoost classifier
            clf_path = os.path.join(self.models_dir, 'xgboost_classifier.joblib')
            if os.path.exists(clf_path):
                self.classifier = joblib.load(clf_path)
                logger.info("XGBoost Classifier loaded successfully")
            
            # Load XGBoost regressor
            reg_path = os.path.join(self.models_dir, 'xgboost_regressor.joblib')
            if os.path.exists(reg_path):
                self.regressor = joblib.load(reg_path)
                logger.info("XGBoost Regressor loaded successfully")
            
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
            
            return self.classifier is not None or self.regressor is not None
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_sleep_disorder(self, input_data):
        """
        Predict sleep disorder using XGBoost classifier
        """
        if self.classifier is None:
            logger.error("No XGBoost classifier available!")
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
            
            # Make prediction
            prediction = self.classifier.predict(processed_input_scaled)[0]
            probabilities = self.classifier.predict_proba(processed_input_scaled)[0]
            
            # Convert prediction back to original label
            if self.class_names is not None:
                predicted_class = self.class_names[prediction]
            else:
                predicted_class = prediction
            
            # Create probability dictionary
            prob_dict = {}
            if self.class_names is not None:
                for i, class_name in enumerate(self.class_names):
                    prob_dict[class_name] = float(probabilities[i])
            else:
                for i, prob in enumerate(probabilities):
                    prob_dict[f'Class_{i}'] = float(prob)
            
            return predicted_class, prob_dict
            
        except Exception as e:
            logger.error(f"Error making sleep disorder prediction: {e}")
            return None, None
    
    def predict_stress_level(self, input_data):
        """
        Predict stress level using XGBoost regressor
        """
        if self.regressor is None:
            logger.error("No XGBoost regressor available!")
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
            
            # Make prediction
            prediction = self.regressor.predict(processed_input_scaled)[0]
            
            # Ensure prediction is within reasonable bounds (1-10)
            prediction = max(1, min(10, prediction))
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error making stress level prediction: {e}")
            return None
    
    def calculate_sleep_quality_percentage(self, disorder_probabilities):
        """
        Calculate sleep quality percentage based on disorder probabilities
        """
        if disorder_probabilities and 'None' in disorder_probabilities:
            sleep_quality = disorder_probabilities['None'] * 100
        elif disorder_probabilities:
            max_disorder_prob = max([prob for key, prob in disorder_probabilities.items() if key != 'None'])
            sleep_quality = (1 - max_disorder_prob) * 100
        else:
            sleep_quality = 50
        
        return round(sleep_quality, 1)
    
    def get_health_recommendations(self, predicted_disorder, stress_level, sleep_quality):
        """
        Generate health recommendations based on predictions
        """
        recommendations = []
        
        if predicted_disorder == 'Sleep Apnea':
            recommendations.extend([
                "🏥 Konsultasikan dengan dokter spesialis tidur untuk evaluasi sleep apnea",
                "😴 Pertimbangkan penggunaan CPAP jika diresepkan dokter",
                "🏃‍♂️ Jaga berat badan ideal melalui olahraga teratur",
                "🚫 Hindari alkohol dan obat penenang sebelum tidur",
                "🛏️ Tidur miring, hindari posisi telentang"
            ])
        elif predicted_disorder == 'Insomnia':
            recommendations.extend([
                "🛏️ Buat jadwal tidur yang konsisten setiap hari",
                "📱 Batasi penggunaan layar 1 jam sebelum tidur",
                "☕ Hindari kafein setelah jam 2 siang",
                "🧘‍♀️ Lakukan teknik relaksasi sebelum tidur",
                "🌙 Jaga kamar tidur tetap gelap, sejuk, dan tenang"
            ])
        else:
            recommendations.extend([
                "✅ Pertahankan kebiasaan tidur sehat Anda",
                "🛏️ Jaga jadwal tidur yang konsisten",
                "🌙 Pertahankan praktik higiene tidur yang baik"
            ])
        
        if stress_level and stress_level >= 7:
            recommendations.extend([
                "🧘‍♀️ Praktikkan teknik manajemen stres (meditasi, yoga)",
                "🏃‍♂️ Lakukan aktivitas fisik secara teratur",
                "👥 Pertimbangkan untuk konsultasi dengan konselor",
                "⏰ Perbaiki manajemen waktu dan keseimbangan kerja"
            ])
        elif stress_level and stress_level >= 5:
            recommendations.extend([
                "🌿 Coba teknik relaksasi seperti pernapasan dalam",
                "🚶‍♀️ Ambil istirahat dan jalan kaki singkat secara teratur"
            ])
        else:
            recommendations.append("😌 Tingkat stres Anda terkelola dengan baik!")
        
        if sleep_quality and sleep_quality < 60:
            recommendations.extend([
                "🌡️ Optimalkan suhu kamar tidur (18-20°C)",
                "🔇 Pastikan kamar tidur tenang dan gelap",
                "🛏️ Investasikan pada kasur dan bantal yang nyaman"
            ])
        
        return recommendations
    
    def make_comprehensive_prediction(self, input_data):
        """
        Make comprehensive predictions including disorder, stress level, and recommendations
        """
        results = {
            'sleep_disorder': None,
            'disorder_probabilities': None,
            'stress_level': None,
            'sleep_quality_percentage': None,
            'recommendations': [],
            'feature_importance': {}
        }
        
        # Predict sleep disorder
        if self.classifier is not None:
            disorder, probabilities = self.predict_sleep_disorder(input_data)
            results['sleep_disorder'] = disorder
            results['disorder_probabilities'] = probabilities
            results['sleep_quality_percentage'] = self.calculate_sleep_quality_percentage(probabilities)
        
        # Predict stress level
        if self.regressor is not None:
            stress = self.predict_stress_level(input_data)
            results['stress_level'] = stress
        
        # Generate recommendations
        results['recommendations'] = self.get_health_recommendations(
            results['sleep_disorder'],
            results['stress_level'] if results['stress_level'] else 5,
            results['sleep_quality_percentage'] if results['sleep_quality_percentage'] else 70
        )
        
        # Get feature importance
        results['feature_importance'] = self.get_feature_importance()
        
        return results

    def get_feature_importance(self):
        """
        Get feature importance for both classifier and regressor models
        """
        importance_results = {
            'classifier_importance': [],
            'regressor_importance': []
        }
        
        # Get classifier feature importance
        if self.classifier is not None and self.feature_names is not None:
            clf_importance = self.classifier.feature_importances_
            max_clf_importance = max(clf_importance) if len(clf_importance) > 0 else 1.0
            
            # Create list of dictionaries for easier template access
            for i, feature in enumerate(self.feature_names):
                importance_results['classifier_importance'].append({
                    'feature': feature,
                    'score': float(clf_importance[i]),
                    'percentage': float((clf_importance[i] / max_clf_importance) * 100)
                })
            
            # Sort by importance (descending)
            importance_results['classifier_importance'].sort(key=lambda x: x['score'], reverse=True)
        
        # Get regressor feature importance
        if self.regressor is not None and self.feature_names is not None:
            reg_importance = self.regressor.feature_importances_
            max_reg_importance = max(reg_importance) if len(reg_importance) > 0 else 1.0
            
            # Create list of dictionaries for easier template access
            for i, feature in enumerate(self.feature_names):
                importance_results['regressor_importance'].append({
                    'feature': feature,
                    'score': float(reg_importance[i]),
                    'percentage': float((reg_importance[i] / max_reg_importance) * 100)
                })
            
            # Sort by importance (descending)
            importance_results['regressor_importance'].sort(key=lambda x: x['score'], reverse=True)
        
        return importance_results

    def model_info(self):
        """
        Get information about loaded models
        """
        info = {
            'classifier_loaded': self.classifier is not None,
            'regressor_loaded': self.regressor is not None,
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