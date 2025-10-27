# Advanced Machine Learning — Healthcare Analytics Portfolio

Overview
--------
This project demonstrates an end-to-end machine learning pipeline for diabetes risk prediction, with emphasis on clinical validity and production deployment (AWS SageMaker). The repository contains data preparation, model development, validation, and deployment artifacts along with a lightweight FastAPI inference example.

Goals
-----
- Build robust predictive models to identify patients at high risk of diabetes.
- Provide clinically-interpretable risk scores and decision support.
- Deploy a production-ready inference service (SageMaker / FastAPI).
- Implement monitoring for model performance and data drift.

Key results (representative)
----------------------------
These are the reported metrics for the champion model (Logistic Regression) on the Pima Indians Diabetes dataset (768 samples):
- Accuracy: 84.2%
- Precision: 79.1%
- Recall (Sensitivity): 86.4%
- F1-score: 82.6%
- AUC-ROC: 0.847

Project scope
-------------
- Data: Pima Indians Diabetes Database (public dataset for demonstration)
- Tasks: binary classification (diabetes vs. no diabetes), feature engineering, model selection, cross-validation, clinical-style validation, production deployment
- Primary technologies: Python, pandas, NumPy, scikit-learn, XGBoost, FastAPI, AWS SageMaker, Docker

Repository structure
--------------------
05-Machine-Learning/
- diabetes-predictive-analytics.ipynb      — exploratory analysis and modeling notebook
- src/
  - data_preprocessing.py                  — data cleaning & feature engineering
  - model_training.py                      — training, CV, and model persistence
  - clinical_validation.py                 — validation and subgroup analysis
  - deployment/
    - sagemaker_deploy.py                  — scripts to package and deploy to SageMaker
    - api_endpoints.py                     — FastAPI inference server
    - monitoring.py                        — drift detection and metrics logging
- data/
  - pima-diabetes.csv
  - preprocessed/
- models/
  - logistic_regression.pkl
  - random_forest.pkl
  - gradient_boosting.pkl
- requirements.txt
- README.md

Getting started
---------------
1. Create a Python environment and install dependencies:
   pip install -r requirements.txt

2. Train a local model (example):
   python src/model_training.py --config config/diabetes_model.yaml

3. Run the local FastAPI server for inference:
   uvicorn src.deployment.api_endpoints:app --reload --port 8000

4. Deploy to AWS SageMaker (example):
   python src/deployment/sagemaker_deploy.py --env production

Example: feature engineering helper
-----------------------------------
Here is a compact, safe example of domain-specific feature engineering used in the project:

```python
import pandas as pd

def create_medical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-specific feature engineering:
    - BMI categories
    - Glucose risk levels
    - Age groups
    """
    df = df.copy()
    # BMI categorization
    df['bmi_category'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 25, 30, float('inf')],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )

    # Glucose risk levels (example cutoffs)
    df['glucose_risk'] = pd.cut(
        df['Glucose'],
        bins=[0, 100, 126, float('inf')],
        labels=['Normal', 'Prediabetic', 'Diabetic']
    )

    # Age groups
    df['age_group'] = pd.cut(
        df['Age'],
        bins=[0, 30, 45, 60, float('inf')],
        labels=['Young', 'Adult', 'Middle-aged', 'Senior']
    )

    return df
```

Modeling & validation notes
--------------------------
- Use stratified k-fold cross-validation for reliable performance estimation on class-imbalanced data.
- Track metrics that matter for clinical deployment: sensitivity (recall), specificity, AUC, and false negative rate.
- Perform subgroup analysis (age, BMI, sex, etc.) to evaluate fairness and clinical robustness.
- Keep a reproducible pipeline (preprocessing + feature selection + model) so production inputs match training.

Production deployment
---------------------
- FastAPI provides a minimal example inference API (src/deployment/api_endpoints.py).
- SageMaker deployment scripts package the model and dependencies and create a real-time endpoint.
- Monitor endpoints for latency, throughput, prediction distributions, and data drift. Trigger retraining when drift exceeds thresholds.
- Use secure storage and encryption for PHI; follow HIPAA, GDPR, and local regulations for clinical data.

Monitoring & governance
-----------------------
- Log prediction metadata, latencies, and high-risk counts.
- Implement drift detection (statistical tests on feature distributions) and performance alerts.
- Version models and data pipelines (MLflow or similar) for reproducibility and audit trails.
- Include clinical oversight (human-in-the-loop) for high-risk predictions.

Privacy, compliance & clinical validation
----------------------------------------
- This project is intended for demonstration and research. Any real-world clinical use requires:
  - Institutional review board (IRB) approval and patient consent where applicable.
  - Data handling that complies with HIPAA / GDPR and local laws.
  - Prospective clinical validation before clinical deployment.
  - Documentation required for regulatory pathways (e.g., SaMD guidance).

Future enhancements
-------------------
- Ensemble methods and stacking for improved performance
- Explainability: SHAP / LIME integration for per-prediction explanations
- Federated learning to train across multiple institutions without centralizing PHI
- Continuous learning pipeline with safe retraining and clinical review

How to contribute
-----------------
- Improve data preprocessing, feature engineering, and hyperparameter tuning.
- Add unit tests for preprocessing and prediction endpoints.
- Harden the deployment pipeline and add CI/CD for model packaging and testing.
- Add E2E examples and clearer configs for SageMaker runs.

Contact
-------
Joseph Bidias  
Email: rodabeck777@gmail.com  
Phone: (214) 886-3785

Notes & disclaimer
------------------
- The provided metrics come from experiments on a public dataset for demonstration. Do not use these models in production clinical settings without appropriate validation and compliance checks.
- This README was revised to remove formatting errors and inconsistent fragments from the original document. For any additional changes or to retain specific text from the prior version, open a pull request or an issue in the repository.
