import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class CardiovascularRiskAnalyzer:
    def __init__(self, data_path='data/cardiovascular_patients.csv'):
        try:
            self.df = pd.read_csv(data_path)
        except FileNotFoundError:
            print("Data file not found. Please run data_generation.py first.")
            return
        
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def exploratory_analysis(self):
        """Comprehensive exploratory data analysis with visualizations"""
        print("=== CARDIOVASCULAR RISK ANALYSIS REPORT ===")
        print(f"Dataset Overview: {len(self.df)} patients")
        print(f"Cardiovascular Event Rate: {self.df['cardiovascular_event'].mean():.2%}")
        
        # Basic statistics
        print("\nKey Statistics:")
        print(f"Average Age: {self.df['age'].mean():.1f} years")
        print(f"Male Patients: {(self.df['gender'] == 'M').mean():.1%}")
        print(f"Diabetes Prevalence: {self.df['diabetes'].mean():.1%}")
        print(f"Hypertension Prevalence: {self.df['hypertension'].mean():.1%}")
        
        # Risk factor analysis
        risk_factors = ['diabetes', 'hypertension', 'smoking', 'family_history_cad']
        print("\nRisk Factor Analysis:")
        
        for factor in risk_factors:
            with_factor = self.df[self.df[factor] == 1]['cardiovascular_event'].mean()
            without_factor = self.df[self.df[factor] == 0]['cardiovascular_event'].mean()
            relative_risk = with_factor / without_factor if without_factor > 0 else 0
            
            print(f"{factor.replace('_', ' ').title()}:")
            print(f"  - With factor: {with_factor:.1%}")
            print(f"  - Without factor: {without_factor:.1%}")
            print(f"  - Relative Risk: {relative_risk:.2f}")
        
        return self.df
    
    def feature_engineering(self):
        """Advanced feature engineering for cardiovascular risk prediction"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Create new features
        self.df['pulse_pressure'] = self.df['systolic_bp'] - self.df['diastolic_bp']
        self.df['cholesterol_ratio'] = self.df['total_cholesterol'] / self.df['hdl_cholesterol']
        
        # Risk stratification
        self.df['risk_category'] = pd.cut(
            self.df['framingham_risk_score'],
            bins=[0, 5, 10, 20, 100],
            labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
        )
        
        print("New features created:")
        print("- Pulse pressure")
        print("- Cholesterol ratio")
        print("- Risk categories")
        
        return self.df
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n=== MODEL TRAINING ===")
        
        # Prepare features
        categorical_cols = ['gender', 'race']
        numerical_cols = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'total_cholesterol',
                         'ldl_cholesterol', 'hdl_cholesterol', 'triglycerides', 'hba1c',
                         'creatinine', 'ejection_fraction', 'pulse_pressure', 'cholesterol_ratio']
        binary_cols = ['diabetes', 'hypertension', 'smoking', 'family_history_cad',
                      'ace_inhibitor', 'beta_blocker', 'statin', 'aspirin']
        
        # Encode categorical variables
        df_encoded = self.df.copy()
        le = LabelEncoder()
        
        for col in categorical_cols:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        # Prepare feature matrix
        feature_cols = numerical_cols + binary_cols + categorical_cols
        X = df_encoded[feature_cols]
        y = df_encoded['cardiovascular_event']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Model configurations
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        results = []
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(model, X_train if name != 'Logistic Regression' else X_train_scaled, 
                                      y_train, cv=5, scoring='roc_auc')
            
            results.append({
                'Model': name,
                'AUC Score': auc_score,
                'CV Mean AUC': cv_scores.mean(),
                'CV Std AUC': cv_scores.std()
            })
            
            # Store model
            self.models[name] = model
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[name] = importance
        
        # Results summary
        results_df = pd.DataFrame(results)
        print("\nModel Performance Summary:")
        print(results_df.round(4))
        
        # Best model analysis
        best_model_name = results_df.loc[results_df['AUC Score'].idxmax(), 'Model']
        print(f"\nBest performing model: {best_model_name}")
        
        return results_df, X_test, y_test, feature_cols
    
    def generate_insights(self):
        """Generate clinical insights and recommendations"""
        print("\n=== CLINICAL INSIGHTS ===")
        
        # High-risk patient analysis
        high_risk_patients = self.df[self.df['framingham_risk_score'] > 15]
        
        print(f"High-Risk Patient Analysis (n={len(high_risk_patients)}):")
        print(f"- Average Age: {high_risk_patients['age'].mean():.1f} years")
        print(f"- Male Gender: {(high_risk_patients['gender'] == 'M').mean():.1%}")
        print(f"- Diabetes: {high_risk_patients['diabetes'].mean():.1%}")
        print(f"- Hypertension: {high_risk_patients['hypertension'].mean():.1%}")
        print(f"- Current Smoking: {high_risk_patients['smoking'].mean():.1%}")
        
        # Treatment gaps
        print(f"\nTreatment Gaps in High-Risk Patients:")
        print(f"- Not on Statin: {(1 - high_risk_patients['statin']).mean():.1%}")
        print(f"- Not on ACE Inhibitor: {(1 - high_risk_patients['ace_inhibitor']).mean():.1%}")
        print(f"- Not on Aspirin: {(1 - high_risk_patients['aspirin']).mean():.1%}")
        
        return high_risk_patients

def main():
    """Main execution pipeline"""
    print("Starting Cardiovascular Risk Analysis Pipeline...")
    
    # Initialize analyzer
    analyzer = CardiovascularRiskAnalyzer()
    
    # Run analysis pipeline
    print("\n1. Exploratory Data Analysis")
    dataset = analyzer.exploratory_analysis()
    
    print("\n2. Feature Engineering")
    enhanced_df = analyzer.feature_engineering()
    
    print("\n3. Model Training and Evaluation")
    model_results, X_test, y_test, feature_cols = analyzer.train_models()
    
    print("\n4. Clinical Insights Generation")
    insights = analyzer.generate_insights()
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Key Findings:")
    print(f"- Dataset: {len(analyzer.df)} patients")
    print(f"- Event Rate: {analyzer.df['cardiovascular_event'].mean():.2%}")
    print(f"- Best Model AUC: {model_results['AUC Score'].max():.3f}")
    print(f"- High-Risk Patients: {(analyzer.df['framingham_risk_score'] > 15).sum()}")
    
    # Save results
    model_results.to_csv('reports/model_performance.csv', index=False)
    insights.to_csv('reports/high_risk_patients.csv', index=False)
    
    print("\nResults saved to reports/ directory")
    return analyzer, model_results

if __name__ == "__main__":
    analyzer, results = main()
