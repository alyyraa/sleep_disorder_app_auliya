<<<<<<< HEAD
# sleep_disorder_app_auliya
=======
# 😴 Sleep Disorder Diagnosis App

A comprehensive Streamlit application for predicting sleep disorders and stress levels using machine learning models trained on the Sleep Disorder Diagnosis Dataset from Kaggle.

## 🚀 Features

### 📊 Data Analysis
- **Dataset Overview**: Complete statistics and data preview
- **Exploratory Data Analysis**: Interactive visualizations and insights
- **Correlation Analysis**: Feature relationship heatmaps
- **Distribution Plots**: Key feature distributions and patterns

### 🤖 Machine Learning
- **Classification Models**: 
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- **Regression Models**:
  - Linear Regression
  - Random Forest Regressor
- **Model Evaluation**: Comprehensive metrics and cross-validation
- **Hyperparameter Tuning**: Optional GridSearchCV optimization

### 🔮 Predictions
- **Sleep Disorder Classification**: None, Insomnia, Sleep Apnea
- **Stress Level Prediction**: Numeric scale (1-10)
- **Sleep Quality Assessment**: Percentage based on disorder probabilities
- **Health Recommendations**: Personalized advice based on predictions

## 📁 Project Structure

```
sleep_disorder_app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── models/
│   ├── train_model.py         # ML model training pipeline
│   └── predict_model.py       # Prediction functions
├── utils/
│   ├── preprocessing.py       # Data cleaning and preprocessing
│   └── eda.py                 # Exploratory data analysis functions
└── data/
    └── (dataset files)        # Kaggle dataset location
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or higher (recommended for optimal compatibility)
- Kaggle account and API credentials (optional - sample data included)
- Git (optional, for cloning)

### Step 1: Clone/Download the Project
```bash
# If using git
git clone <repository-url>
cd sleep_disorder_app

# Or download and extract the project folder
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv_sleep_app

# Activate virtual environment
# On macOS/Linux:
source venv_sleep_app/bin/activate
# On Windows:
venv_sleep_app\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning library
- `xgboost>=2.0.0` - Gradient boosting framework
- `matplotlib>=3.7.0` - Plotting library
- `seaborn>=0.12.0` - Statistical data visualization
- `plotly>=5.15.0` - Interactive plotting
- `joblib>=1.3.0` - Model serialization

### Step 4: Set Up Kaggle API (Optional)

**Note**: A sample dataset is included, so Kaggle setup is optional for testing.

1. **Create Kaggle API Token**:
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Click "Create New API Token"
   - Download `kaggle.json` file

2. **Set Environment Variables**:
   ```bash
   # Option 1: Export environment variables
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   
   # Option 2: Place kaggle.json in ~/.kaggle/
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 5: Download Full Dataset (Optional)
```bash
# Download full Kaggle dataset
kaggle datasets download -d mdsultanulislamovi/sleep-disorder-diagnosis-dataset -p data --unzip

# Or use the included sample dataset for testing
# Sample data is automatically available in data/sample_sleep_data.csv
```

### Step 6: Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Home Page
- View dataset status and basic statistics
- Check if all components are properly loaded
- Navigate to different sections

### 2. Exploratory Data Analysis (EDA)
- **Dataset Overview**: Records, features, missing values, duplicates
- **Correlation Heatmap**: Feature relationships visualization
- **Distribution Plots**: Age, sleep duration, stress level distributions
- **Sleep Disorder Analysis**: Disorder distribution and patterns
- **Interactive Plots**: Customizable scatter plots
- **Key Insights**: Automated data insights

### 3. Model Training
- Select training options (with/without hyperparameter tuning)
- Train multiple models simultaneously
- View training progress and results
- Models are automatically saved for future use

**Training Results Include**:
- **Classification Metrics**: Accuracy, Cross-validation scores
- **Regression Metrics**: RMSE, MAE, R² scores
- **Model Comparison**: Side-by-side performance comparison

### 4. Prediction Interface
- **Model Selection**: Choose classification and regression models
- **Input Form**: Enter patient information
  - Age, Gender, Sleep Duration
  - Quality of Sleep, Physical Activity Level
  - BMI Category, Heart Rate, Daily Steps
  - Blood Pressure (Systolic/Diastolic)
- **Results Display**:
  - Sleep Disorder prediction with probabilities
  - Stress Level prediction (1-10 scale)
  - Sleep Quality percentage
  - Personalized health recommendations

## 🎯 Model Performance

### Classification Models (Sleep Disorder Prediction)
- **Target Classes**: None, Insomnia, Sleep Apnea
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Cross-Validation**: 5-fold CV for robust evaluation

### Regression Models (Stress Level Prediction)
- **Target Range**: 1-10 stress level scale
- **Evaluation Metrics**: RMSE, MAE, R² Score
- **Cross-Validation**: 5-fold CV with negative MSE scoring

## 🔧 Technical Details

### Recent Improvements (v2.0)

#### 🛠️ Small Dataset Handling
The application now intelligently handles small datasets with robust error handling:

- **Dynamic Cross-Validation**: Automatically adjusts CV folds based on dataset size
  ```python
  cv_folds = min(5, max(2, n_samples // 2))
  ```
- **Stratification Fallback**: Falls back to non-stratified splitting when classes have too few samples
- **Reduced Hyperparameter Grids**: Optimizes search space for small datasets to prevent overfitting
- **Warning System**: Provides clear warnings when working with limited data

#### 🎯 Cross-Validation Enhancements
- **Adaptive Folding**: Prevents "n_splits greater than n_samples" errors
- **Minimum Fold Guarantee**: Ensures at least 2-fold CV for any dataset size
- **Performance Optimization**: Reduces computational overhead for small datasets

### Data Preprocessing
- **Missing Value Handling**: Dropna approach with data quality reporting
- **Categorical Encoding**: Label encoding for categorical features with encoder persistence
- **Feature Scaling**: StandardScaler for numerical features with scaler persistence
- **Smart Train-Test Split**: 80-20 split with intelligent stratification handling
  ```python
  # Attempts stratified split, falls back to random split if needed
  try:
      return train_test_split(X, y, stratify=y, test_size=0.2)
  except ValueError:
      return train_test_split(X, y, test_size=0.2)  # Non-stratified fallback
  ```

### Model Architecture
- **Classification Pipeline**: Data Loading → Preprocessing → Scaling → Model Training → Evaluation
- **Regression Pipeline**: Data Loading → Preprocessing → Scaling → Model Training → Evaluation
- **Model Persistence**: Joblib serialization with metadata storage
- **Hyperparameter Optimization**: GridSearchCV with dynamic parameter grids

### Key Features Used
- **Demographics**: Age, Gender
- **Sleep Metrics**: Sleep Duration, Quality of Sleep
- **Health Indicators**: BMI Category, Heart Rate, Daily Steps
- **Cardiovascular**: Systolic/Diastolic Blood Pressure
- **Lifestyle**: Physical Activity Level

### Code Architecture

#### Core Modules

**`app.py`** - Main Streamlit Application
```python
# Multi-page application with navigation
# Custom CSS styling and responsive design
# Integration of all components
```

**`models/train_model.py`** - ML Training Pipeline
```python
class SleepDisorderModels:
    def __init__(self):
        # Initialize classification and regression models
        # Set up hyperparameter grids
    
    def train_classification_models(self, X_train, X_test, y_train, y_test):
        # Dynamic cross-validation with adaptive folding
        # Comprehensive evaluation metrics
    
    def hyperparameter_tuning(self, X_train, y_train, model_type):
        # Smart grid search with dataset-size awareness
        # Reduced search space for small datasets
```

**`models/predict_model.py`** - Prediction Interface
```python
class SleepDisorderPredictor:
    def load_models(self):
        # Load trained models and preprocessors
    
    def predict_sleep_disorder(self, input_data):
        # Classification with probability scores
    
    def predict_stress_level(self, input_data):
        # Regression prediction with confidence intervals
```

**`utils/preprocessing.py`** - Data Processing
```python
def split_data(X, y, test_size=0.2, random_state=42):
    # Intelligent stratification with fallback
    try:
        return train_test_split(X, y, stratify=y, test_size=test_size)
    except ValueError:
        # Handle small dataset edge cases
        return train_test_split(X, y, test_size=test_size)
```

**`utils/eda.py`** - Exploratory Data Analysis
```python
def create_comprehensive_eda(df):
    # Interactive visualizations
    # Statistical summaries
    # Correlation analysis
    # Distribution plots
```

## 🚨 Troubleshooting

### Common Issues & Solutions

#### 1. **Cross-Validation Errors** ✅ FIXED
```
ValueError: Cannot have number of splits n_splits=5 greater than the number of samples: n_samples=4
```
**Solution**: This is now automatically handled! The app uses dynamic cross-validation:
- Automatically adjusts CV folds based on dataset size
- Minimum 2-fold CV for any dataset
- Falls back gracefully for very small datasets

#### 2. **Stratification Errors** ✅ FIXED
```
ValueError: The least populated class in y has only 1 member, which is too few
```
**Solution**: Smart stratification with automatic fallback:
- Attempts stratified split first
- Falls back to random split when stratification fails
- Provides clear warnings about data limitations

#### 3. **"No dataset found" Error**
```bash
# Option 1: Use included sample data (recommended for testing)
# Sample data is automatically available at data/sample_sleep_data.csv

# Option 2: Download full Kaggle dataset
kaggle datasets download -d mdsultanulislamovi/sleep-disorder-diagnosis-dataset -p data --unzip

# Option 3: Check data directory structure
ls -la data/
```

#### 4. **Kaggle API Authentication Error**
```bash
# Check if Kaggle is installed
pip install kaggle

# Verify credentials
kaggle datasets list

# Re-set environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Or check kaggle.json location
ls ~/.kaggle/kaggle.json
```

#### 5. **Import/Dependency Errors**
```bash
# Check Python version (3.10+ recommended)
python --version

# Reinstall requirements in clean environment
pip install -r requirements.txt --upgrade

# For specific missing packages:
pip install streamlit pandas scikit-learn xgboost matplotlib seaborn plotly
```

#### 6. **Model Loading Issues**
- **First Time**: Train models using the "Train Models" page
- **Check Files**: Verify model files exist in `models/` directory
  ```bash
  ls -la models/*.joblib
  ```
- **Retrain**: If models are corrupted, retrain from the web interface

#### 7. **Memory/Performance Issues**
```bash
# Check available memory
free -h  # Linux
top      # macOS

# Reduce dataset size for testing
head -n 100 data/your_dataset.csv > data/sample_data.csv
```

#### 8. **Streamlit-Specific Issues**
```bash
# Clear Streamlit cache
streamlit cache clear

# Run with specific port
streamlit run app.py --server.port 8502

# Debug mode
streamlit run app.py --logger.level debug
```

### Performance Optimization

#### For Large Datasets (>10,000 rows)
- **EDA Sampling**: Use data sampling for faster visualization
- **Selective Hyperparameter Tuning**: Enable only for final models
- **Batch Processing**: Process data in chunks if memory limited

#### For Small Datasets (<100 rows)
- **Disable Hyperparameter Tuning**: Use default parameters
- **Increase Test Size**: Use 30-40% for test set
- **Simple Models**: Prefer Logistic Regression over complex models

#### Memory Management
- **Close Browser Tabs**: Reduce memory usage during training
- **Virtual Environment**: Use isolated environment
- **Monitor Resources**: Check CPU/memory usage during training

### Environment-Specific Issues

#### macOS
```bash
# Install Xcode command line tools (for some dependencies)
xcode-select --install

# Use Homebrew Python if system Python causes issues
brew install python@3.10
```

#### Windows
```bash
# Use PowerShell or Command Prompt
# Activate virtual environment
venv_sleep_app\Scripts\activate

# Install Visual C++ Build Tools if needed for some packages
```

#### Linux
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip

# For plotting libraries
sudo apt-get install python3-tk
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
# Add to app.py for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or run with debug flag
streamlit run app.py --logger.level debug
```

## 📚 API Documentation

### Core Classes and Methods

#### `SleepDisorderModels` (models/train_model.py)

```python
class SleepDisorderModels:
    """Main class for training sleep disorder prediction models"""
    
    def __init__(self):
        """Initialize classification and regression models with hyperparameter grids"""
        
    def train_classification_models(self, X_train, X_test, y_train, y_test):
        """Train classification models for sleep disorder prediction
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features  
            y_train (pd.Series): Training labels
            y_test (pd.Series): Testing labels
            
        Returns:
            dict: Classification results with metrics and trained models
        """
        
    def train_regression_models(self, X_train, X_test, y_train, y_test):
        """Train regression models for stress level prediction
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features
            y_train (pd.Series): Training stress levels
            y_test (pd.Series): Testing stress levels
            
        Returns:
            dict: Regression results with metrics and trained models
        """
        
    def hyperparameter_tuning(self, X_train, y_train, model_type='classification'):
        """Perform hyperparameter tuning with dynamic grid sizing
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
            model_type (str): 'classification' or 'regression'
            
        Returns:
            dict: Best parameters for each model
        """
```

#### `SleepDisorderPredictor` (models/predict_model.py)

```python
class SleepDisorderPredictor:
    """Class for making predictions using trained models"""
    
    def __init__(self):
        """Initialize predictor with model loading capabilities"""
        
    def load_models(self):
        """Load all trained models and preprocessors
        
        Returns:
            bool: True if models loaded successfully
        """
        
    def predict_sleep_disorder(self, input_data):
        """Predict sleep disorder classification
        
        Args:
            input_data (dict): Patient data dictionary
            
        Returns:
            tuple: (prediction, probabilities, confidence)
        """
        
    def predict_stress_level(self, input_data):
        """Predict stress level (1-10 scale)
        
        Args:
            input_data (dict): Patient data dictionary
            
        Returns:
            float: Predicted stress level
        """
        
    def get_health_recommendations(self, disorder_pred, stress_level, input_data):
        """Generate personalized health recommendations
        
        Args:
            disorder_pred (str): Predicted sleep disorder
            stress_level (float): Predicted stress level
            input_data (dict): Patient data
            
        Returns:
            list: List of health recommendations
        """
```

#### Utility Functions (utils/preprocessing.py)

```python
def load_data(file_path):
    """Load and validate dataset
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset or None if error
    """
    
def clean_data(df):
    """Clean dataset by removing missing values and duplicates
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    
def encode_categorical_features(df):
    """Encode categorical features using label encoding
    
    Args:
        df (pd.DataFrame): Dataset with categorical features
        
    Returns:
        tuple: (encoded_df, label_encoders_dict)
    """
    
def split_data(X, y, test_size=0.2, random_state=42):
    """Smart data splitting with stratification fallback
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion for test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
```

#### EDA Functions (utils/eda.py)

```python
def create_comprehensive_eda(df):
    """Create comprehensive exploratory data analysis
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        
    Returns:
        None: Displays Streamlit visualizations
    """
    
def plot_correlation_heatmap(df):
    """Create correlation heatmap for numerical features
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    
def plot_distribution_analysis(df):
    """Create distribution plots for key features
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        None: Displays multiple distribution plots
    """
```

### Input Data Format

#### Required Input Fields

```python
input_data = {
    'Age': int,                    # Age in years (20-80)
    'Gender': str,                 # 'Male' or 'Female'
    'Sleep Duration': float,       # Hours of sleep (4.0-10.0)
    'Quality of Sleep': int,       # Scale 1-10
    'Physical Activity Level': int, # Minutes per day (0-90)
    'BMI Category': str,           # 'Underweight', 'Normal', 'Overweight', 'Obese'
    'Heart Rate': int,             # BPM (65-90)
    'Daily Steps': int,            # Steps per day (3000-15000)
    'Systolic BP': int,            # mmHg (110-140)
    'Diastolic BP': int            # mmHg (70-95)
}
```

#### Output Format

```python
# Classification Output
{
    'prediction': str,              # 'None', 'Insomnia', 'Sleep Apnea'
    'probabilities': dict,          # {'None': 0.7, 'Insomnia': 0.2, 'Sleep Apnea': 0.1}
    'confidence': float,            # 0.0-1.0
    'sleep_quality_percentage': int # 0-100%
}

# Regression Output
{
    'stress_level': float,          # 1.0-10.0
    'confidence_interval': tuple   # (lower_bound, upper_bound)
}

# Recommendations Output
[
    "Get 7-9 hours of sleep nightly",
    "Maintain regular exercise routine",
    "Consider stress management techniques"
]
```

## 🚀 Deployment Guide

### Local Development

```bash
# Clone repository
git clone <repository-url>
cd sleep_disorder_app

# Set up environment
python -m venv venv_sleep_app
source venv_sleep_app/bin/activate  # Linux/macOS
# venv_sleep_app\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Production Deployment

#### Option 1: Streamlit Cloud

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub repository
   - Set main file path: `app.py`
   - Deploy automatically

#### Option 2: Docker Deployment

**Create Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and Run**:
```bash
# Build image
docker build -t sleep-disorder-app .

# Run container
docker run -p 8501:8501 sleep-disorder-app
```

#### Option 3: Heroku Deployment

**Create Procfile**:
```
web: sh setup.sh && streamlit run app.py
```

**Create setup.sh**:
```bash
mkdir -p ~/.streamlit/

echo "\n\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\n\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

**Deploy**:
```bash
heroku create your-app-name
git push heroku main
```

#### Option 4: AWS EC2 Deployment

```bash
# Launch EC2 instance (Ubuntu 20.04)
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv nginx

# Clone and setup application
git clone <your-repo>
cd sleep_disorder_app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with PM2 for process management
npm install -g pm2
pm2 start "streamlit run app.py" --name sleep-app

# Configure nginx reverse proxy
sudo nano /etc/nginx/sites-available/sleep-app
```

### Environment Variables

**For Production**:
```bash
# .env file
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Performance Optimization for Production

```python
# Add to app.py for production
import streamlit as st

# Cache configuration
st.set_page_config(
    page_title="Sleep Disorder App",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enable caching for expensive operations
@st.cache_data
def load_and_process_data(file_path):
    # Cached data loading
    pass

@st.cache_resource
def load_trained_models():
    # Cached model loading
    pass
```

## 📊 Dataset Information

**Source**: [Sleep Disorder Diagnosis Dataset](https://www.kaggle.com/datasets/mdsultanulislamovi/sleep-disorder-diagnosis-dataset)

**Features**:
- **Demographics**: Age, Gender
- **Sleep Metrics**: Sleep Duration, Quality of Sleep
- **Health Indicators**: BMI Category, Heart Rate, Blood Pressure
- **Lifestyle**: Physical Activity Level, Daily Steps
- **Targets**: Sleep Disorder, Stress Level

**Dataset Statistics**:
- **Records**: ~400 patient records
- **Features**: 11 input features + 2 target variables
- **Missing Values**: Handled automatically
- **Data Types**: Mixed (numerical and categorical)
- **Target Classes**: 3 sleep disorder categories
- **Stress Range**: 1-10 scale

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset provided by [mdsultanulislamovi](https://www.kaggle.com/mdsultanulislamovi) on Kaggle
- Built with [Streamlit](https://streamlit.io/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/)

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the dataset and model requirements
3. Ensure all dependencies are properly installed
4. Verify Kaggle API credentials are correctly set

---

**Happy Sleep Analysis! 😴💤**
>>>>>>> f139165 (Clean repo with .gitignore applied)
