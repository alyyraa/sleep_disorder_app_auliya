import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from utils.preprocessing import preprocess_pipeline
from utils.eda import create_comprehensive_eda
from models.train_model import train_models_pipeline
from models.predict_model import SleepDisorderPredictor

# Page configuration
st.set_page_config(
    page_title="Sleep Disorder Diagnosis App",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .recommendation-item {
        background-color: #f9f9f9;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        border-left: 3px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def check_data_availability():
    """
    Check if dataset is available in the data folder
    """
    data_folder = Path("data")
    csv_files = list(data_folder.glob("*.csv"))
    return len(csv_files) > 0, csv_files

def load_dataset():
    """
    Load the sleep disorder dataset
    """
    data_available, csv_files = check_data_availability()
    
    if not data_available:
        st.error("📁 No dataset found in the 'data' folder!")
        st.info("""
        Please download the dataset using the Kaggle API:
        ```bash
        kaggle datasets download -d mdsultanulislamovi/sleep-disorder-diagnosis-dataset -p data --unzip
        ```
        """)
        return None
    
    # Use the first CSV file found
    dataset_path = csv_files[0]
    
    try:
        df = pd.read_csv(dataset_path)
        return df, str(dataset_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def home_page():
    """
    Home page with app overview and dataset status
    """
    st.markdown('<h1 class="main-header">😴 Sleep Disorder Diagnosis App</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the Sleep Disorder Diagnosis Application!
    
    This application uses machine learning to analyze sleep patterns and predict:
    - **Sleep Disorders**: None, Insomnia, or Sleep Apnea
    - **Stress Levels**: Numeric prediction (1-10 scale)
    - **Sleep Quality**: Percentage based on disorder probabilities
    
    ### 🚀 Features:
    - **Data Overview**: Explore the dataset with interactive visualizations
    - **EDA**: Comprehensive exploratory data analysis
    - **Model Training**: Train multiple ML models (Logistic Regression, Random Forest, XGBoost)
    - **Predictions**: Get personalized sleep disorder and stress level predictions
    
    ### 📊 Navigation:
    Use the sidebar to navigate between different sections of the app.
    """)
    
    # Check dataset availability
    st.subheader("📁 Dataset Status")
    data_available, csv_files = check_data_availability()
    
    if data_available:
        st.success(f"✅ Dataset found: {csv_files[0].name}")
        
        # Load and display basic info
        dataset_result = load_dataset()
        if dataset_result:
            df, dataset_path = dataset_result
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicates", df.duplicated().sum())
            
            # Display column names
            st.subheader("📋 Dataset Columns")
            st.write(list(df.columns))
    else:
        st.error("❌ Dataset not found!")
        st.info("""
        **To get started:**
        1. Set up your Kaggle API credentials
        2. Run: `kaggle datasets download -d mdsultanulislamovi/sleep-disorder-diagnosis-dataset -p data --unzip`
        3. Refresh this page
        """)

def eda_page():
    """
    Exploratory Data Analysis page
    """
    st.title("🔍 Exploratory Data Analysis")
    
    dataset_result = load_dataset()
    if dataset_result is None:
        return
    
    df, dataset_path = dataset_result
    
    # Create comprehensive EDA
    create_comprehensive_eda(df)

def training_page():
    """
    Model training page
    """
    st.title("🤖 Model Training")
    
    dataset_result = load_dataset()
    if dataset_result is None:
        return
    
    df, dataset_path = dataset_result
    
    st.markdown("""
    Train machine learning models for sleep disorder classification and stress level regression.
    """)
    
    # Training options
    col1, col2 = st.columns(2)
    
    with col1:
        perform_tuning = st.checkbox("🔧 Perform Hyperparameter Tuning", 
                                   help="This will take longer but may improve model performance")
    
    with col2:
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    results = train_models_pipeline(dataset_path, perform_tuning)
                    if results:
                        st.success("🎉 Training completed successfully!")
                except Exception as e:
                    st.error(f"Training failed: {e}")
    
    # Check if models exist
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.joblib"))
        if model_files:
            st.subheader("💾 Saved Models")
            st.write(f"Found {len(model_files)} model files:")
            for model_file in model_files:
                st.write(f"• {model_file.name}")

def prediction_page():
    """
    Prediction page with input form and results
    """
    st.title("🔮 Sleep Disorder Prediction")
    
    # Check if models are available
    predictor = SleepDisorderPredictor()
    models_loaded = predictor.load_models()
    
    if not models_loaded:
        st.warning("⚠️ No trained models found! Please train models first.")
        if st.button("Go to Training Page"):
            st.session_state.page = "Train Models"
            st.rerun()
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Model selection
    available_models = predictor.get_available_models()
    
    col1, col2 = st.columns(2)
    with col1:
        if available_models['classifiers']:
            classifier_choice = st.selectbox(
                "🎯 Classification Model",
                available_models['classifiers'],
                help="Choose the model for sleep disorder prediction"
            )
        else:
            st.warning("No classification models available")
            classifier_choice = None
    
    with col2:
        if available_models['regressors']:
            regressor_choice = st.selectbox(
                "📊 Regression Model",
                available_models['regressors'],
                help="Choose the model for stress level prediction"
            )
        else:
            st.warning("No regression models available")
            regressor_choice = None
    
    # Input form
    st.subheader("📝 Patient Information")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            sleep_duration = st.number_input("Sleep Duration (hours)", min_value=3.0, max_value=12.0, value=7.5, step=0.1)
            quality_of_sleep = st.slider("Quality of Sleep (1-10)", min_value=1, max_value=10, value=8)
        
        with col2:
            physical_activity = st.slider("Physical Activity Level (0-100)", min_value=0, max_value=100, value=75)
            bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=120, value=70, step=1)
        
        with col3:
            daily_steps = st.number_input("Daily Steps", min_value=1000, max_value=20000, value=8000, step=100)
            systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120, step=1)
            diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=120, value=80, step=1)
        
        submitted = st.form_submit_button("🔮 Make Prediction", type="primary")
    
    if submitted:
        # Prepare input data
        input_data = {
            'Age': age,
            'Gender': gender,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_of_sleep,
            'Physical Activity Level': physical_activity,
            'BMI Category': bmi_category,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Systolic BP': systolic_bp,
            'Diastolic BP': diastolic_bp
        }
        
        # Make predictions
        with st.spinner("Making predictions..."):
            results = predictor.make_comprehensive_prediction(
                input_data, classifier_choice, regressor_choice
            )
        
        # Display results
        st.subheader("📊 Prediction Results")
        
        # Main predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if results['sleep_disorder']:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>😴 Sleep Disorder</h3>
                    <h2 style="color: {'#d32f2f' if results['sleep_disorder'] != 'None' else '#388e3c'}">
                        {results['sleep_disorder']}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if results['stress_level']:
                stress_color = "#d32f2f" if results['stress_level'] >= 7 else "#ff9800" if results['stress_level'] >= 5 else "#388e3c"
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>😰 Stress Level</h3>
                    <h2 style="color: {stress_color}">
                        {results['stress_level']:.1f}/10
                    </h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if results['sleep_quality_percentage']:
                quality_color = "#388e3c" if results['sleep_quality_percentage'] >= 70 else "#ff9800" if results['sleep_quality_percentage'] >= 50 else "#d32f2f"
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>💤 Sleep Quality</h3>
                    <h2 style="color: {quality_color}">
                        {results['sleep_quality_percentage']:.1f}%
                    </h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Disorder probabilities
        if results['disorder_probabilities']:
            st.subheader("📈 Sleep Disorder Probabilities")
            prob_df = pd.DataFrame([
                {'Disorder': disorder, 'Probability': f"{prob:.1%}"}
                for disorder, prob in results['disorder_probabilities'].items()
            ])
            st.dataframe(prob_df, use_container_width=True)
        
        # Recommendations
        if results['recommendations']:
            st.subheader("💡 Health Recommendations")
            for recommendation in results['recommendations']:
                st.markdown(f"""
                <div class="recommendation-item">
                    {recommendation}
                </div>
                """, unsafe_allow_html=True)

def main():
    """
    Main application function
    """
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    pages = {
        "🏠 Home": home_page,
        "🔍 EDA": eda_page,
        "🤖 Train Models": training_page,
        "🔮 Prediction": prediction_page
    }
    
    # Page selection
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"
    
    selected_page = st.sidebar.radio(
        "Select a page:",
        list(pages.keys()),
        index=list(pages.keys()).index(st.session_state.page) if st.session_state.page in pages.keys() else 0
    )
    
    st.session_state.page = selected_page
    
    # App info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📋 About
    This app predicts sleep disorders and stress levels using machine learning.
    
    **Models Used:**
    - Logistic Regression
    - Random Forest
    - XGBoost
    
    **Predictions:**
    - Sleep Disorder Classification
    - Stress Level Regression
    - Sleep Quality Assessment
    """)
    
    # Dataset download instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📥 Dataset Setup
    ```bash
    # Install Kaggle API
    pip install kaggle
    
    # Set credentials
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_key
    
    # Download dataset
    kaggle datasets download -d mdsultanulislamovi/sleep-disorder-diagnosis-dataset -p data --unzip
    ```
    """)
    
    # Run selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()