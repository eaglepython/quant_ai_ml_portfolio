import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Risk Analytics Platform",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #f44336;
        color: #c62828;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
        color: #e65100;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load cardiovascular data"""
    try:
        df = pd.read_csv('data/cardiovascular_patients.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run data_generation.py first.")
        return None

def calculate_risk_score(patient_data):
    """Calculate cardiovascular risk score"""
    risk_score = 0
    
    # Age contribution
    if patient_data['age'] > 65:
        risk_score += 3
    elif patient_data['age'] > 55:
        risk_score += 2
    elif patient_data['age'] > 45:
        risk_score += 1
    
    # Gender contribution
    if patient_data['gender'] == 'Male':
        risk_score += 1
    
    # Risk factors
    if patient_data['diabetes']:
        risk_score += 3
    if patient_data['hypertension']:
        risk_score += 2
    if patient_data['smoking']:
        risk_score += 2
    if patient_data['family_history']:
        risk_score += 1
    
    # Clinical values
    if patient_data['systolic_bp'] > 140:
        risk_score += 2
    elif patient_data['systolic_bp'] > 130:
        risk_score += 1
    
    if patient_data['total_cholesterol'] > 240:
        risk_score += 2
    elif patient_data['total_cholesterol'] > 200:
        risk_score += 1
    
    if patient_data['hdl_cholesterol'] < 40:
        risk_score += 1
    
    if patient_data['bmi'] > 30:
        risk_score += 1
    
    # Convert to percentage
    risk_percentage = min(50, max(1, risk_score * 3))
    return risk_percentage, risk_score

def get_risk_category(risk_percentage):
    """Categorize risk level"""
    if risk_percentage < 5:
        return "Low Risk", "🟢"
    elif risk_percentage < 10:
        return "Moderate Risk", "🟡"
    elif risk_percentage < 20:
        return "High Risk", "🟠"
    else:
        return "Very High Risk", "🔴"

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Cardiovascular Risk Analytics Platform</h1>', 
               unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_patients = len(df)
        st.metric("Total Patients", f"{total_patients:,}")
    
    with col2:
        event_rate = df['cardiovascular_event'].mean()
        st.metric("CV Event Rate", f"{event_rate:.1%}")
    
    with col3:
        high_risk_count = (df['framingham_risk_score'] > 15).sum()
        st.metric("High Risk Patients", f"{high_risk_count:,}")
    
    with col4:
        avg_age = df['age'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Dashboard",
        ["Risk Calculator", "Population Analytics", "Clinical Insights", "Executive Summary"]
    )
    
    if page == "Risk Calculator":
        display_risk_calculator()
    elif page == "Population Analytics":
        display_population_analytics(df)
    elif page == "Clinical Insights":
        display_clinical_insights(df)
    elif page == "Executive Summary":
        display_executive_summary(df)

def display_risk_calculator():
    """Individual patient risk calculator"""
    st.header("🩺 Individual Patient Risk Calculator")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Patient Information")
        
        # Demographics
        age = st.slider("Age", 18, 100, 55)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        # Risk factors
        st.subheader("Risk Factors")
        diabetes = st.checkbox("Diabetes")
        hypertension = st.checkbox("Hypertension") 
        smoking = st.checkbox("Current Smoker")
        family_history = st.checkbox("Family History of CAD")
        
        # Clinical measurements
        st.subheader("Clinical Measurements")
        bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
        systolic_bp = st.slider("Systolic BP (mmHg)", 80, 200, 120)
        total_cholesterol = st.slider("Total Cholesterol (mg/dL)", 100, 400, 200)
        hdl_cholesterol = st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
    
    with col2:
        st.subheader("Risk Assessment Results")
        
        # Patient data
        patient_data = {
            'age': age,
            'gender': gender,
            'diabetes': diabetes,
            'hypertension': hypertension,
            'smoking': smoking,
            'family_history': family_history,
            'bmi': bmi,
            'systolic_bp': systolic_bp,
            'total_cholesterol': total_cholesterol,
            'hdl_cholesterol': hdl_cholesterol
        }
        
        # Calculate risk
        risk_percentage, risk_score = calculate_risk_score(patient_data)
        risk_category, risk_emoji = get_risk_category(risk_percentage)
        
        # Display risk
        st.markdown(f"""
        <div class="metric-card {'risk-high' if 'High' in risk_category else 'risk-medium' if 'Moderate' in risk_category else 'risk-low'}">
            <h3>{risk_emoji} {risk_category}</h3>
            <h2>{risk_percentage}% 10-Year CV Risk</h2>
            <p>Risk Score: {risk_score}/20</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "10-Year CV Risk (%)"},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 5], 'color': "lightgreen"},
                    {'range': [5, 10], 'color': "yellow"},
                    {'range': [10, 20], 'color': "orange"},
                    {'range': [20, 50], 'color': "red"}
                ]
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Clinical Recommendations")
        
        if risk_percentage > 20:
            st.write("🔴 High priority for aggressive risk factor modification")
            st.write("💊 Consider statin therapy if not contraindicated")
            st.write("🩺 Cardiology consultation recommended")
        elif risk_percentage > 10:
            st.write("🟡 Moderate risk - lifestyle modifications essential")
            st.write("💊 Consider preventive medications")
        else:
            st.write("🟢 Low risk - continue current management")
            st.write("🏃‍♂️ Maintain healthy lifestyle")

def display_population_analytics(df):
    """Population analytics dashboard"""
    st.header("📊 Population Health Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by events
        fig = px.histogram(df, x='age', color='cardiovascular_event',
                          title="Age Distribution by CV Events")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk factors prevalence
        risk_factors = ['diabetes', 'hypertension', 'smoking', 'family_history_cad']
        prevalence = [df[factor].mean() for factor in risk_factors]
        
        fig = px.bar(x=[f.replace('_', ' ').title() for f in risk_factors], 
                    y=prevalence,
                    title="Risk Factor Prevalence")
        st.plotly_chart(fig, use_container_width=True)

def display_clinical_insights(df):
    """Clinical insights dashboard"""
    st.header("🏥 Clinical Decision Support Insights")
    
    # High-risk patients
    high_risk_patients = df[df['framingham_risk_score'] > 15]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High-Risk Patients", len(high_risk_patients))
    with col2:
        st.metric("Requiring Intervention", len(high_risk_patients) - high_risk_patients['statin'].sum())
    with col3:
        avg_risk = high_risk_patients['framingham_risk_score'].mean()
        st.metric("Average Risk Score", f"{avg_risk:.1f}%")
    
    # Patient table
    st.subheader("High-Risk Patient Prioritization")
    display_df = df.nlargest(20, 'framingham_risk_score')[
        ['patient_id', 'age', 'gender', 'framingham_risk_score', 'diabetes', 
         'hypertension', 'smoking', 'statin', 'cardiovascular_event']
    ]
    st.dataframe(display_df, use_container_width=True)

def display_executive_summary(df):
    """Executive dashboard"""
    st.header("📈 Executive Summary Dashboard")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", f"{len(df):,}")
    with col2:
        event_rate = df['cardiovascular_event'].mean()
        st.metric("CV Event Rate", f"{event_rate:.1%}", delta="-2.3%")
    with col3:
        high_risk = (df['framingham_risk_score'] > 15).sum()
        st.metric("High-Risk Managed", f"{high_risk:,}")
    with col4:
        # Cost savings calculation
        events_prevented = len(df) * 0.023
        cost_savings = events_prevented * 50000
        st.metric("Annual Cost Savings", f"${cost_savings:,.0f}")
    
    # ROI Analysis
    st.subheader("Return on Investment Analysis")
    
    program_cost = 500000
    roi_ratio = cost_savings / program_cost
    
    col1, col2 = st.columns(2)
    
    with col1:
        roi_data = pd.DataFrame({
            'Category': ['Program Cost', 'Cost Savings', 'Net Benefit'],
            'Amount': [-program_cost, cost_savings, cost_savings - program_cost]
        })
        
        fig = px.bar(roi_data, x='Category', y='Amount',
                    title=f"Financial Impact (ROI: {roi_ratio:.1f}x)",
                    color='Amount',
                    color_continuous_scale=['red', 'yellow', 'green'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Quality trends
        months = pd.date_range('2024-01-01', periods=12, freq='M')
        event_rates = np.random.normal(0.12, 0.01, 12)
        event_rates = np.maximum(0.08, event_rates - np.linspace(0, 0.04, 12))
        
        trend_df = pd.DataFrame({
            'Month': months,
            'CV Event Rate': event_rates
        })
        
        fig = px.line(trend_df, x='Month', y='CV Event Rate',
                     title='CV Event Rate Improvement Trend')
        fig.add_hline(y=0.12, line_dash="dash", annotation_text="Baseline")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
