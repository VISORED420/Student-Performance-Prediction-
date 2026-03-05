# Student Math Score Predictor

An end-to-end machine learning project that predicts student math exam scores based on demographic and academic features. Built with a modular ML pipeline and served through a Flask web application.

## Overview

The project trains and evaluates 7 regression models on a dataset of 1,000 student records, automatically selects the best performer via GridSearchCV, and exposes predictions through a simple web form.

**Input features:** gender, race/ethnicity, parental education level, lunch type, test preparation course, reading score, writing score

**Target:** math score

## Tech Stack

- **ML:** scikit-learn, XGBoost, CatBoost
- **Data:** pandas, numpy, seaborn, matplotlib
- **Web:** Flask
- **Deployment:** AWS Elastic Beanstalk, Azure App Service

## Project Structure

```
mlproject/
├── app.py                        # Flask web application
├── setup.py                      # Package configuration
├── requirements.txt              # Dependencies
├── src/
│   ├── exception.py              # Custom exception handling
│   ├── logger.py                 # Logging setup
│   ├── utils.py                  # Helper functions (save/load objects, model evaluation)
│   ├── components/
│   │   ├── data_ingestion.py     # Read data and train/test split
│   │   ├── data_transformation.py# Feature preprocessing pipeline
│   │   └── model_trainer.py      # Model training and selection
│   └── pipeline/
│       ├── train_pipeline.py     # Training orchestration
│       └── predict_pipeline.py   # Inference pipeline with CustomData class
├── notebook/
│   ├── 1. EDA STUDENT PERFORMANCE .ipynb
│   ├── 2. MODEL TRAINING.ipynb
│   └── data/stud.csv            # Raw dataset
├── templates/
│   ├── index.html                # Landing page
│   └── home.html                 # Prediction form
├── artifacts/                    # Generated model and data artifacts
├── .ebextensions/                # AWS EB config
└── .github/workflows/            # Azure CI/CD pipeline
```

## ML Pipeline

### 1. Data Ingestion
Reads the raw dataset, saves a copy, and splits into 80/20 train/test sets.

### 2. Data Transformation
- **Numerical features** (reading_score, writing_score): median imputation + standard scaling
- **Categorical features** (gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course): mode imputation + one-hot encoding + standard scaling

Saves the fitted preprocessor as `artifacts/preprocessor.pkl`.

### 3. Model Training
Evaluates 7 regression models with hyperparameter tuning (GridSearchCV, 3-fold CV):

| Model | Key Hyperparameters |
|-------|-------------------|
| Random Forest | n_estimators, max_depth, max_features |
| Decision Tree | criterion, max_depth, max_features |
| Gradient Boosting | learning_rate, n_estimators, subsample |
| Linear Regression | - |
| XGBoost | learning_rate, n_estimators |
| CatBoost | depth, learning_rate, iterations |
| AdaBoost | learning_rate, n_estimators |

The best model (by R² score, minimum threshold 0.6) is saved to `artifacts/model.pkl`.

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd mlproject

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python src/components/data_ingestion.py
```

This runs the full pipeline: ingestion → transformation → training. Artifacts are saved to `artifacts/`.

### Run the Web App

```bash
python app.py
```

Navigate to `http://localhost:5000/predictdata`, fill in the student details form, and get a predicted math score.

## Deployment

### AWS Elastic Beanstalk
Configured via `.ebextensions/python.config` with WSGI path pointing to the Flask app.

### Azure App Service
GitHub Actions workflow (`.github/workflows/`) triggers on push to `main`:
1. Sets up Python 3.7
2. Installs dependencies
3. Deploys to Azure Web App
