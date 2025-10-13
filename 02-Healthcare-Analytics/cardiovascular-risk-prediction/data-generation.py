import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_cardiovascular_dataset(n_patients=10000):
    """Generate comprehensive synthetic cardiovascular patient dataset"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Demographics
    patient_ids = [f"CV_{str(i).zfill(6)}" for i in range(1, n_patients + 1)]
    ages = np.random.normal(65, 15, n_patients).astype(int)
    ages = np.clip(ages, 18, 95)
    
    genders = np.random.choice(['M', 'F'], n_patients, p=[0.52, 0.48])
    races = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                            n_patients, p=[0.6, 0.15, 0.15, 0.08, 0.02])
    
    # Risk factors with realistic correlations
    diabetes = np.random.binomial(1, 0.25, n_patients)
    hypertension = np.random.binomial(1, 0.45, n_patients)
    smoking = np.random.binomial(1, 0.20, n_patients)
    family_history = np.random.binomial(1, 0.35, n_patients)
    
    # Clinical measurements with correlations
    bmi_base = np.random.normal(28, 6, n_patients)
    bmi = bmi_base + diabetes * 3 + hypertension * 2
    bmi = np.clip(bmi, 15, 50)
    
    # Blood pressure
    systolic_bp = np.random.normal(130, 20, n_patients) + hypertension * 25 + ages * 0.3
    diastolic_bp = systolic_bp * 0.6 + np.random.normal(0, 5, n_patients)
    systolic_bp = np.clip(systolic_bp, 90, 200)
    diastolic_bp = np.clip(diastolic_bp, 50, 120)
    
    # Cholesterol levels
    total_cholesterol = np.random.normal(200, 40, n_patients) + ages * 0.5
    ldl_cholesterol = total_cholesterol * 0.6 + np.random.normal(0, 15, n_patients)
    hdl_cholesterol = np.random.normal(50, 15, n_patients) - diabetes * 8
    triglycerides = np.random.normal(150, 60, n_patients) + diabetes * 50
    
    # Lab values
    hba1c = np.random.normal(5.7, 1.2, n_patients) + diabetes * 2.5
    creatinine = np.random.normal(1.0, 0.3, n_patients) + ages * 0.01
    ejection_fraction = np.random.normal(60, 10, n_patients)
    
    # Medications
    ace_inhibitor = np.random.binomial(1, 0.4, n_patients)
    beta_blocker = np.random.binomial(1, 0.35, n_patients)
    statin = np.random.binomial(1, 0.45, n_patients)
    aspirin = np.random.binomial(1, 0.50, n_patients)
    
    # Calculate risk scores
    framingham_risk = calculate_framingham_risk(ages, genders, systolic_bp, 
                                               total_cholesterol, hdl_cholesterol, 
                                               smoking, diabetes, hypertension)
    
    # Outcomes based on risk factors
    outcome_prob = (0.05 + 
                   diabetes * 0.15 + 
                   hypertension * 0.10 + 
                   smoking * 0.12 + 
                   family_history * 0.08 + 
                   (ages - 40) * 0.002 +
                   framingham_risk * 0.01)
    
    cardiovascular_event = np.random.binomial(1, outcome_prob, n_patients)
    
    # Time to event (in days from baseline)
    days_to_event = np.random.exponential(365 * 2, n_patients)
    days_to_event = np.where(cardiovascular_event == 1, 
                           np.minimum(days_to_event, 365 * 3), 
                           365 * 5)
    
    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': ages,
        'gender': genders,
        'race': races,
        'bmi': np.round(bmi, 1),
        'systolic_bp': np.round(systolic_bp, 0),
        'diastolic_bp': np.round(diastolic_bp, 0),
        'total_cholesterol': np.round(total_cholesterol, 0),
        'ldl_cholesterol': np.round(ldl_cholesterol, 0),
        'hdl_cholesterol': np.round(hdl_cholesterol, 0),
        'triglycerides': np.round(triglycerides, 0),
        'hba1c': np.round(hba1c, 1),
        'creatinine': np.round(creatinine, 2),
        'ejection_fraction': np.round(ejection_fraction, 0),
        'diabetes': diabetes,
        'hypertension': hypertension,
        'smoking': smoking,
        'family_history_cad': family_history,
        'ace_inhibitor': ace_inhibitor,
        'beta_blocker': beta_blocker,
        'statin': statin,
        'aspirin': aspirin,
        'framingham_risk_score': np.round(framingham_risk, 1),
        'cardiovascular_event': cardiovascular_event,
        'days_to_event': np.round(days_to_event, 0).astype(int),
        'enrollment_date': pd.date_range('2020-01-01', periods=n_patients, freq='D')[:n_patients]
    })
    
    return df

def calculate_framingham_risk(ages, genders, sbp, total_chol, hdl_chol, smoking, diabetes, hypertension):
    """Calculate Framingham Risk Score"""
    risk_scores = []
    
    for i in range(len(ages)):
        age = ages[i]
        gender = genders[i]
        
        # Age points
        if gender == 'M':
            if age < 35: age_points = -9
            elif age < 40: age_points = -4
            elif age < 45: age_points = 0
            elif age < 50: age_points = 3
            elif age < 55: age_points = 6
            elif age < 60: age_points = 8
            elif age < 65: age_points = 10
            elif age < 70: age_points = 11
            elif age < 75: age_points = 12
            else: age_points = 13
        else:  # Female
            if age < 35: age_points = -7
            elif age < 40: age_points = -3
            elif age < 45: age_points = 0
            elif age < 50: age_points = 3
            elif age < 55: age_points = 6
            elif age < 60: age_points = 8
            elif age < 65: age_points = 10
            elif age < 70: age_points = 12
            elif age < 75: age_points = 14
            else: age_points = 16
        
        # HDL points
        if hdl_chol[i] >= 60: hdl_points = -1
        elif hdl_chol[i] >= 50: hdl_points = 0
        elif hdl_chol[i] >= 40: hdl_points = 1
        else: hdl_points = 2
        
        # Total cholesterol points
        if total_chol[i] < 160: chol_points = 0
        elif total_chol[i] < 200: chol_points = 4
        elif total_chol[i] < 240: chol_points = 7
        elif total_chol[i] < 280: chol_points = 9
        else: chol_points = 11
        
        # Smoking points
        smoke_points = 8 if smoking[i] else 0
        
        # Blood pressure points
        if sbp[i] < 120: bp_points = 0
        elif sbp[i] < 130: bp_points = 0
        elif sbp[i] < 140: bp_points = 1
        elif sbp[i] < 160: bp_points = 1
        else: bp_points = 2
        
        # Diabetes points
        diabetes_points = 6 if diabetes[i] else 0
        
        total_points = age_points + hdl_points + chol_points + smoke_points + bp_points + diabetes_points
        
        # Convert to 10-year risk percentage
        if gender == 'M':
            if total_points < 0: risk = 1
            elif total_points < 5: risk = 1
            elif total_points < 7: risk = 2
            elif total_points < 9: risk = 5
            elif total_points < 11: risk = 6
            elif total_points < 13: risk = 10
            elif total_points < 15: risk = 13
            elif total_points < 17: risk = 20
            else: risk = 30
        else:  # Female
            if total_points < 9: risk = 1
            elif total_points < 13: risk = 1
            elif total_points < 15: risk = 2
            elif total_points < 17: risk = 5
            elif total_points < 19: risk = 8
            elif total_points < 21: risk = 11
            elif total_points < 23: risk = 15
            elif total_points < 25: risk = 20
            else: risk = 25
        
        risk_scores.append(risk)
    
    return np.array(risk_scores)

if __name__ == "__main__":
    print("Generating cardiovascular dataset...")
    df = generate_cardiovascular_dataset(10000)
    
    # Save dataset
    df.to_csv('data/cardiovascular_patients.csv', index=False)
    print(f"Dataset generated with {len(df)} patients")
    print(f"Cardiovascular event rate: {df['cardiovascular_event'].mean():.2%}")
    print("Dataset saved to data/cardiovascular_patients.csv")
    print("\nFirst 5 rows:")
    print(df.head())
