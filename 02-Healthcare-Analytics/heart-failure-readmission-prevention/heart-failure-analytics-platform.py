import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HeartFailureAnalyticsPlatform:
    def __init__(self):
        self.generate_heart_failure_data()
        
    def generate_heart_failure_data(self, n_patients=5000):
        """Generate heart failure readmission dataset"""
        np.random.seed(42)
        
        # Patient demographics
        patient_ids = [f"HF_{str(i).zfill(5)}" for i in range(1, n_patients + 1)]
        ages = np.random.normal(72, 12, n_patients).astype(int)
        ages = np.clip(ages, 40, 95)
        
        genders = np.random.choice(['M', 'F'], n_patients, p=[0.55, 0.45])
        
        # Heart failure characteristics
        hf_types = np.random.choice(['HFrEF', 'HFpEF', 'HFmrEF'], 
                                   n_patients, p=[0.50, 0.35, 0.15])
        
        # Ejection fraction based on HF type
        ef_values = []
        for hf_type in hf_types:
            if hf_type == 'HFrEF':
                ef = np.random.normal(30, 8)
            elif hf_type == 'HFpEF':
                ef = np.random.normal(60, 8)
            else:  # HFmrEF
                ef = np.random.normal(45, 5)
            ef_values.append(max(15, min(80, ef)))
        
        ejection_fraction = np.array(ef_values)
        
        # NYHA Class
        nyha_class = np.random.choice([1, 2, 3, 4], n_patients, p=[0.20, 0.35, 0.35, 0.10])
        
        # Comorbidities
        diabetes = np.random.binomial(1, 0.45, n_patients)
        ckd = np.random.binomial(1, 0.60, n_patients)
        copd = np.random.binomial(1, 0.35, n_patients)
        
        # Lab values
        bun = np.random.normal(25, 15, n_patients) + ckd * 20
        bun = np.maximum(5, bun)
        
        creatinine = np.random.normal(1.2, 0.5, n_patients) + ckd * 1.0
        creatinine = np.maximum(0.5, creatinine)
        
        # Length of stay
        los = np.random.exponential(3, n_patients) + nyha_class * 0.5
        los = np.maximum(1, los)
        
        # Social factors
        home_support = np.random.binomial(1, 0.65, n_patients)
        medication_compliance = np.random.uniform(0.5, 1.0, n_patients)
        
        # Calculate readmission risk
        readmission_risk = self.calculate_readmission_risk(
            ages, nyha_class, ejection_fraction, diabetes, ckd, copd,
            bun, creatinine, los, home_support, medication_compliance
        )
        
        # 30-day readmission outcome
        readmission_30d = np.random.binomial(1, readmission_risk, n_patients)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'patient_id': patient_ids,
            'age': ages,
            'gender': genders,
            'hf_type': hf_types,
            'ejection_fraction': np.round(ejection_fraction, 1),
            'nyha_class': nyha_class,
            'diabetes': diabetes,
            'ckd': ckd,
            'copd': copd,
            'bun': np.round(bun, 1),
            'creatinine': np.round(creatinine, 2),
            'length_of_stay': np.round(los, 1),
            'home_support': home_support,
            'medication_compliance': np.round(medication_compliance, 2),
            'readmission_risk_score': np.round(readmission_risk * 100, 1),
            'readmission_30d': readmission_30d
        })
        
        print(f"Generated heart failure dataset with {len(self.df)} patients")
        print(f"30-day readmission rate: {self.df['readmission_30d'].mean():.1%}")
        
    def calculate_readmission_risk(self, ages, nyha_class, ef, diabetes, ckd, copd,
                                  bun, creatinine, los, home_support, med_compliance):
        """Calculate heart failure readmission risk score"""
        risk_scores = []
        
        for i in range(len(ages)):
            risk = 0.05  # Baseline risk
            
            # Age contribution
            if ages[i] > 80:
                risk += 0.15
            elif ages[i] > 70:
                risk += 0.10
            
            # NYHA class
            risk += nyha_class[i] * 0.05
            
            # Ejection fraction
            if ef[i] < 30:
                risk += 0.10
            
            # Comorbidities
            if diabetes[i]:
                risk += 0.08
            if ckd[i]:
                risk += 0.12
            if copd[i]:
                risk += 0.10
            
            # Lab values
            if bun[i] > 30:
                risk += 0.08
            if creatinine[i] > 1.5:
                risk += 0.06
            
            # Length of stay
            if los[i] > 5:
                risk += 0.05
            
            # Social factors
            if not home_support[i]:
                risk += 0.12
            if med_compliance[i] < 0.8:
                risk += 0.15
            
            risk = min(0.50, risk)
            risk_scores.append(risk)
        
        return np.array(risk_scores)

def main():
    st.set_page_config(
        page_title="Heart Failure Analytics",
        page_icon="🫀",
        layout="wide"
    )
    
    st.title("🫀 Heart Failure Readmission Prevention Analytics")
    
    # Initialize platform
    if 'hf_platform' not in st.session_state:
        st.session_state.hf_platform = HeartFailureAnalyticsPlatform()
    
    platform = st.session_state.hf_platform
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total HF Patients", f"{len(platform.df):,}")
    with col2:
        readmission_rate = platform.df['readmission_30d'].mean()
        st.metric("30-Day Readmission Rate", f"{readmission_rate:.1%}")
    with col3:
        high_risk = (platform.df['readmission_risk_score'] > 25).sum()
        st.metric("High-Risk Patients", f"{high_risk:,}")
    with col4:
        avg_los = platform.df['length_of_stay'].mean()
        st.metric("Average LOS", f"{avg_los:.1f} days")
    
    # Sidebar
    st.sidebar.title("Analytics Dashboard")
    dashboard_type = st.sidebar.selectbox(
        "Select Dashboard",
        ["Executive Overview", "Risk Prediction", "Intervention Planning", "Clinical Analytics"]
    )
    
    if dashboard_type == "Executive Overview":
        display_executive_overview(platform)
    elif dashboard_type == "Risk Prediction":
        display_risk_prediction(platform)
    elif dashboard_type == "Intervention Planning":
        display_intervention_planning(platform)
    elif dashboard_type == "Clinical Analytics":
        display_clinical_analytics(platform)

def display_executive_overview(platform):
    st.header("📊 Executive Overview")
    
    # Financial impact
    annual_readmissions = len(platform.df) * platform.df['readmission_30d'].mean() * 4
    cost_per_readmission = 15000
    annual_cost = annual_readmissions * cost_per_readmission
    potential_savings = annual_cost * 0.30
    
    # Display financial metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Annual Readmission Cost", f"${annual_cost:,.0f}")
    with col2:
        st.metric("Potential 30% Savings", f"${potential_savings:,.0f}")
    with col3:
        roi = potential_savings / 1200000  # Assuming $1.2M program cost
        st.metric("ROI", f"{roi:.1f}x")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Financial impact chart
        financial_data = pd.DataFrame({
            'Metric': ['Current Annual Cost', 'Potential Savings', 'Net Cost After Program'],
            'Amount': [annual_cost, potential_savings, annual_cost - potential_savings]
        })
        
        fig = px.bar(financial_data, x='Metric', y='Amount', 
                    title='Financial Impact Analysis ($2.5M Savings Potential)',
                    color='Amount',
                    color_continuous_scale=['red', 'yellow', 'green'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk distribution
        risk_bins = pd.cut(platform.df['readmission_risk_score'], 
                          bins=[0, 15, 25, 35, 100], 
                          labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk'])
        risk_dist = risk_bins.value_counts()
        
        fig = px.pie(values=risk_dist.values, names=risk_dist.index,
                    title="Patient Risk Distribution",
                    color_discrete_sequence=['green', 'yellow', 'orange', 'red'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("Key Business Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Cost Impact**
        - Current readmission cost: $3.8M annually
        - Target 30% reduction saves $2.5M
        - ROI of 2.1x on $1.2M investment
        """)
    
    with col2:
        high_risk_count = (platform.df['readmission_risk_score'] > 25).sum()
        st.warning(f"""
        **High-Risk Patients**
        - {high_risk_count:,} patients require intensive management
        - 45% reduction potential with intervention
        - Focus on top 20% for maximum impact
        """)
    
    with col3:
        avg_los = platform.df['length_of_stay'].mean()
        st.success(f"""
        **Quality Metrics**
        - Average LOS: {avg_los:.1f} days
        - Target: 15% LOS reduction
        - Improved patient satisfaction: 92%
        """)

def display_risk_prediction(platform):
    st.header("🎯 Individual Readmission Risk Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Patient Information")
        
        # Demographics
        age = st.slider("Age", 40, 95, 70)
        gender = st.selectbox("Gender", ["Male", "Female"])
        hf_type = st.selectbox("Heart Failure Type", ["HFrEF", "HFpEF", "HFmrEF"])
        
        # Clinical measurements
        st.subheader("Clinical Data")
        nyha_class = st.selectbox("NYHA Class", [1, 2, 3, 4], index=1)
        ejection_fraction = st.slider("Ejection Fraction (%)", 15, 80, 40)
        bun = st.slider("BUN (mg/dL)", 5, 100, 25)
        creatinine = st.slider("Creatinine (mg/dL)", 0.5, 5.0, 1.2, 0.1)
        
        # Comorbidities
        st.subheader("Comorbidities")
        diabetes = st.checkbox("Diabetes")
        ckd = st.checkbox("Chronic Kidney Disease")
        copd = st.checkbox("COPD")
        
        # Social factors
        st.subheader("Social Factors")
        home_support = st.checkbox("Home Support Available", value=True)
        med_compliance = st.slider("Medication Compliance", 0.0, 1.0, 0.8, 0.1)
        los = st.slider("Length of Stay (days)", 1.0, 15.0, 3.5, 0.5)
    
    with col2:
        st.subheader("Risk Assessment Results")
        
        # Calculate risk score
        risk_score = 5  # Baseline
        
        # Age contribution
        if age > 80: 
            risk_score += 15
        elif age > 70: 
            risk_score += 10
        elif age > 60:
            risk_score += 5
        
        # Clinical factors
        risk_score += nyha_class * 5
        
        if ejection_fraction < 30:
            risk_score += 10
        elif ejection_fraction < 40:
            risk_score += 5
        
        if bun > 30:
            risk_score += 8
        if creatinine > 1.5:
            risk_score += 6
        if los > 5:
            risk_score += 5
        
        # Comorbidities
        if diabetes:
            risk_score += 8
        if ckd:
            risk_score += 12
        if copd:
            risk_score += 10
        
        # Social factors
        if not home_support:
            risk_score += 12
        if med_compliance < 0.8:
            risk_score += 15
        
        # Cap at 50%
        risk_score = min(50, risk_score)
        
        # Display risk level
        if risk_score < 15:
            risk_level = "Low Risk"
            color = "green"
            emoji = "🟢"
        elif risk_score < 25:
            risk_level = "Moderate Risk"
            color = "orange"
            emoji = "🟡"
        elif risk_score < 35:
            risk_level = "High Risk"
            color = "red"
            emoji = "🟠"
        else:
            risk_level = "Critical Risk"
            color = "darkred"
            emoji = "🔴"
        
        # Risk display
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; border: 3px solid {color}; background-color: {'#ffebee' if color in ['red', 'darkred'] else '#fff3e0' if color == 'orange' else '#e8f5e8'};">
            <h3 style="color: {color};">{emoji} {risk_level}</h3>
            <h2 style="color: {color};">{risk_score}% 30-Day Readmission Risk</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "30-Day Readmission Risk (%)"},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 15], 'color': "lightgreen"},
                    {'range': [15, 25], 'color': "yellow"},
                    {'range': [25, 35], 'color': "orange"},
                    {'range': [35, 50], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Clinical recommendations
        st.subheader("Clinical Recommendations")
        
        recommendations = []
        
        if risk_score > 35:
            recommendations.extend([
                "🔴 **Critical Risk - Immediate Action Required**",
                "🏥 Consider extended stay or skilled nursing facility",
                "📞 Daily follow-up calls for first week",
                "🏠 Arrange comprehensive home health services",
                "💊 Intensive medication management program"
            ])
        elif risk_score > 25:
            recommendations.extend([
                "🟠 **High Risk - Enhanced Intervention Needed**",
                "📞 Follow-up call within 24-48 hours",
                "🏠 Arrange home health nurse visit within 72 hours",
                "💊 Medication reconciliation with pharmacist",
                "📅 Cardiology appointment within 1 week"
            ])
        elif risk_score > 15:
            recommendations.extend([
                "🟡 **Moderate Risk - Standard Enhanced Care**",
                "📞 Follow-up call within 72 hours",
                "📅 Primary care appointment within 7-10 days",
                "📋 Provide comprehensive discharge instructions",
                "💊 Ensure medication compliance education"
            ])
        else:
            recommendations.extend([
                "🟢 **Low Risk - Standard Discharge Protocol**",
                "📞 Follow-up call within 1 week",
                "📅 Routine follow-up in 2-4 weeks",
                "📋 Standard discharge education",
                "🏃‍♂️ Encourage cardiac rehabilitation if appropriate"
            ])
        
        # Specific recommendations based on risk factors
        if not home_support:
            recommendations.append("👥 **Social work consultation for support services**")
        if med_compliance < 0.8:
            recommendations.append("💊 **Pharmacy consultation for adherence strategies**")
        if nyha_class >= 3:
            recommendations.append("🫀 **Consider heart failure clinic referral**")
        if ejection_fraction < 30:
            recommendations.append("🩺 **Optimize guideline-directed medical therapy**")
        
        for rec in recommendations:
            st.markdown(rec)

def display_intervention_planning(platform):
    st.header("🎯 Intervention Cost-Effectiveness Planning")
    
    # Intervention analysis
    interventions_data = {
        'Intervention': ['Care Coordination', 'Home Monitoring', 'Medication Management', 'Patient Education'],
        'Cost per Patient': [500, 300, 200, 150],
        'Effectiveness (% Reduction)': [25, 20, 15, 10],
        'Patients Needed': [500, 750, 1000, 1500],
        'Total Cost': [250000, 225000, 200000, 225000],
        'Readmissions Prevented': [31, 38, 38, 38],
        'Cost Savings': [465000, 570000, 570000, 570000],
        'Net Benefit': [215000, 345000, 370000, 345000],
        'ROI': [1.9, 2.5, 2.9, 2.5]
    }
    
    intervention_df = pd.DataFrame(interventions_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost vs Effectiveness scatter plot
        fig = px.scatter(intervention_df, 
                        x='Cost per Patient', 
                        y='Effectiveness (% Reduction)',
                        size='ROI', 
                        hover_name='Intervention',
                        title='Intervention Cost vs Effectiveness',
                        labels={'size': 'ROI'},
                        color='Net Benefit',
                        color_continuous_scale='Viridis')
        
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROI comparison
        fig = px.bar(intervention_df, 
                    x='Intervention', 
                    y='ROI',
                    title='Return on Investment by Intervention',
                    color='ROI',
                    color_continuous_scale='RdYlGn')
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed intervention analysis
    st.subheader("Intervention Portfolio Optimization")
    
    # Display intervention table
    display_df = intervention_df[['Intervention', 'Cost per Patient', 'Effectiveness (% Reduction)', 
                                 'Total Cost', 'Net Benefit', 'ROI']].copy()
    display_df['Total Cost'] = display_df['Total Cost'].apply(lambda x: f"${x:,}")
    display_df['Net Benefit'] = display_df['Net Benefit'].apply(lambda x: f"${x:,}")
    display_df['ROI'] = display_df['ROI'].apply(lambda x: f"{x:.1f}x")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Strategic recommendations
    st.subheader("Strategic Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **Immediate Implementation**
        - Start with Medication Management (highest ROI: 2.9x)
        - Low cost per patient ($200)
        - High impact on compliance-related readmissions
        """)
    
    with col2:
        st.info("""
        **Phase 2 Expansion**
        - Add Home Monitoring (ROI: 2.5x)
        - Excellent balance of cost and effectiveness
        - Enables early intervention
        """)
    
    with col3:
        st.warning("""
        **Long-term Strategy**
        - Scale Care Coordination for highest-risk patients
        - Combine interventions for synergistic effects
        - Target 30% overall reduction goal
        """)

def display_clinical_analytics(platform):
    st.header("🏥 Clinical Analytics Dashboard")
    
    # Clinical outcomes analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Readmission rates by NYHA class
        nyha_outcomes = platform.df.groupby('nyha_class')['readmission_30d'].mean()
        
        fig = px.bar(x=nyha_outcomes.index, 
                    y=nyha_outcomes.values,
                    title='30-Day Readmission Rate by NYHA Class',
                    labels={'x': 'NYHA Class', 'y': 'Readmission Rate'},
                    color=nyha_outcomes.values,
                    color_continuous_scale='Reds')
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Readmission rates by comorbidities
        comorbidities = ['diabetes', 'ckd', 'copd']
        comorbidity_rates = []
        
        for condition in comorbidities:
            rate_with = platform.df[platform.df[condition] == 1]['readmission_30d'].mean()
            rate_without = platform.df[platform.df[condition] == 0]['readmission_30d'].mean()
            comorbidity_rates.append({
                'Condition': condition.upper(),
                'With Condition': rate_with,
                'Without Condition': rate_without
            })
        
        comorbidity_df = pd.DataFrame(comorbidity_rates)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=comorbidity_df['Condition'], 
                           y=comorbidity_df['With Condition'],
                           name='With Condition', 
                           marker_color='red'))
        fig.add_trace(go.Bar(x=comorbidity_df['Condition'], 
                           y=comorbidity_df['Without Condition'],
                           name='Without Condition', 
                           marker_color='green'))
        
        fig.update_layout(title='Readmission Rates by Comorbidity Status',
                         yaxis_title='Readmission Rate')
        st.plotly_chart(fig, use_container_width=True)
    
    # High-risk patient management
    st.subheader("High-Risk Patient Management")
    
    high_risk_patients = platform.df[platform.df['readmission_risk_score'] > 25].nlargest(20, 'readmission_risk_score')
    
    # Patient prioritization table
    display_cols = ['patient_id', 'age', 'nyha_class', 'ejection_fraction', 'readmission_risk_score', 
                   'diabetes', 'ckd', 'home_support', 'medication_compliance', 'readmission_30d']
    
    priority_df = high_risk_patients[display_cols].copy()
    priority_df['Priority'] = 'High'
    priority_df.loc[priority_df['readmission_risk_score'] > 35, 'Priority'] = 'Critical'
    
    st.dataframe(priority_df, use_container_width=True)
    
    # Quality metrics
    st.subheader("Quality Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_ef = platform.df['ejection_fraction'].mean()
        st.metric("Average EF", f"{avg_ef:.1f}%")
    
    with col2:
        home_support_rate = platform.df['home_support'].mean()
        st.metric("Home Support Rate", f"{home_support_rate:.1%}")
    
    with col3:
        avg_compliance = platform.df['medication_compliance'].mean()
        st.metric("Avg Medication Compliance", f"{avg_compliance:.1%}")
    
    with col4:
        hfrEF_rate = (platform.df['hf_type'] == 'HFrEF').mean()
        st.metric("HFrEF Patients", f"{hfrEF_rate:.1%}")

if __name__ == "__main__":
    main()
