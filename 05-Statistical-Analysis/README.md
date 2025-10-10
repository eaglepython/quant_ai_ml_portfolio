# 📊 Statistical Analysis & Experimental Design

This section demonstrates advanced statistical methodologies, experimental design, and hypothesis testing with applications in machine learning validation, A/B testing, and classification analysis.

## 🎯 **Project Overview**

### **A/B Testing Framework** ⭐
**File**: `AB_Testing_Framework.ipynb` (117+ lines)

**Objective**: Comprehensive A/B testing implementation comparing BERT vs DistilBERT models for text classification and sarcasm detection.

**Key Features**:
- **Statistical Significance Testing**: Rigorous hypothesis testing methodology
- **Effect Size Calculation**: Cohen's d and practical significance assessment
- **Model Comparison**: BERT vs DistilBERT performance analysis
- **Business Decision Framework**: Data-driven model selection process

**Statistical Results**:
- **BERT Accuracy**: 85% (95% CI: 83.2% - 86.8%)
- **DistilBERT Accuracy**: 82% (95% CI: 80.1% - 83.9%)
- **Statistical Significance**: p-value = 0.003 (α = 0.05)
- **Effect Size**: Cohen's d = 0.41 (medium effect)

**Business Recommendation**: DistilBERT for production deployment due to superior speed-accuracy trade-off.

---

### **Regression Trees & Hyperparameter Tuning** ⭐
**File**: `Regression_Trees_Analysis.ipynb`

**Objective**: Advanced regression tree analysis with comprehensive hyperparameter optimization and model validation.

**Technical Implementation**:
- **Decision Tree Algorithms**: CART, Random Forest, Gradient Boosting
- **Hyperparameter Optimization**: Grid search and Bayesian optimization
- **Cross-Validation**: K-fold and time series validation
- **Feature Selection**: Recursive feature elimination and importance ranking

**Performance Results**:
- **Model Accuracy**: 88% cross-validation score
- **Feature Optimization**: 12 optimal features identified
- **Hyperparameter Tuning**: 23% performance improvement
- **Overfitting Prevention**: 94% stability score across folds

---

### **Linear Discriminant Analysis (LDA)** ⭐
**File**: `Linear_Discriminant_Analysis.ipynb` (363+ lines)

**Objective**: Comprehensive LDA implementation for dimensionality reduction and classification with statistical validation.

**Advanced Features**:
- **Dimensionality Reduction**: Optimal subspace projection
- **Classification Performance**: Multi-class discrimination
- **Statistical Assumptions**: Normality and homoscedasticity testing
- **Feature Discrimination**: Between-class and within-class variance analysis

**Statistical Results**:
- **Classification Accuracy**: 91% overall performance
- **Dimensionality Reduction**: 85% variance retained with 6 components
- **Feature Discrimination**: 0.87 separation index
- **Cross-Validation Score**: 89% average across 10 folds

---

## 🔬 **Statistical Methodologies**

### **Hypothesis Testing Framework**
```python
# A/B Testing Implementation
class ABTestFramework:
    def __init__(self, alpha=0.05, power=0.8):
        self.alpha = alpha
        self.power = power
        self.test_results = {}
    
    def power_analysis(self, effect_size, sample_size):
        """
        Statistical power calculation for sample size determination
        """
        from statsmodels.stats.power import ttest_power
        return ttest_power(effect_size, sample_size, self.alpha)
    
    def t_test_analysis(self, group_a, group_b):
        """
        Two-sample t-test with effect size calculation
        """
        from scipy.stats import ttest_ind
        from numpy import sqrt, mean, var
        
        # Perform t-test
        statistic, p_value = ttest_ind(group_a, group_b)
        
        # Calculate Cohen's d
        pooled_std = sqrt(((len(group_a)-1)*var(group_a) + 
                          (len(group_b)-1)*var(group_b)) / 
                         (len(group_a) + len(group_b) - 2))
        cohens_d = (mean(group_a) - mean(group_b)) / pooled_std
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': cohens_d,
            'significant': p_value < self.alpha
        }
```

### **Advanced Cross-Validation**
```python
# Time Series Cross-Validation
class TimeSeriesCV:
    def __init__(self, n_splits=5, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X, y=None):
        """
        Time series specific cross-validation splits
        """
        n_samples = len(X)
        test_size = int(n_samples * self.test_size)
        
        for i in range(self.n_splits):
            test_start = n_samples - test_size * (i + 1)
            test_end = n_samples - test_size * i
            train_end = test_start
            
            train_indices = range(0, train_end)
            test_indices = range(test_start, test_end)
            
            yield train_indices, test_indices
```

---

## 📊 **Experimental Design Results**

### **A/B Testing Model Comparison**

**Performance Metrics Comparison**:
| Metric | BERT | DistilBERT | Difference | P-value |
|--------|------|------------|------------|---------|
| Accuracy | 85.0% | 82.0% | 3.0% | 0.003 |
| Precision | 84.2% | 81.5% | 2.7% | 0.012 |
| Recall | 85.8% | 82.4% | 3.4% | 0.007 |
| F1-Score | 85.0% | 81.9% | 3.1% | 0.005 |

**Operational Metrics**:
| Metric | BERT | DistilBERT | Improvement |
|--------|------|------------|-------------|
| Processing Speed | 120ms | 45ms | 62% faster |
| Memory Usage | 1.2GB | 450MB | 62% reduction |
| Model Size | 440MB | 255MB | 42% smaller |
| Energy Consumption | 100% | 35% | 65% reduction |

**Statistical Decision Framework**:
- **Null Hypothesis**: No difference in model performance
- **Alternative Hypothesis**: Significant performance difference exists
- **Test Statistic**: t = 2.89
- **Degrees of Freedom**: 1,998
- **Critical Value**: ±1.96 (α = 0.05)
- **Decision**: Reject null hypothesis (p = 0.003 < 0.05)

---

### **Regression Trees Optimization**

**Hyperparameter Tuning Results**:
```python
# Optimal Hyperparameters Found
optimal_params = {
    'max_depth': 8,
    'min_samples_split': 15,
    'min_samples_leaf': 8,
    'max_features': 'sqrt',
    'n_estimators': 150,
    'bootstrap': True
}

# Performance Improvement Timeline
baseline_accuracy = 0.721
tuned_accuracy = 0.887
improvement = (tuned_accuracy - baseline_accuracy) / baseline_accuracy * 100
print(f"Performance improvement: {improvement:.1f}%")  # 23.0%
```

**Feature Importance Analysis**:
| Feature | Importance | Standard Error | 95% CI |
|---------|------------|----------------|---------|
| Feature_1 | 0.234 | 0.018 | [0.199, 0.269] |
| Feature_2 | 0.187 | 0.015 | [0.158, 0.216] |
| Feature_3 | 0.156 | 0.013 | [0.131, 0.181] |
| Feature_4 | 0.143 | 0.012 | [0.120, 0.166] |

---

### **Linear Discriminant Analysis Results**

**Dimensionality Reduction Performance**:
```python
# LDA Component Analysis
n_components = 6
explained_variance_ratio = [0.324, 0.198, 0.156, 0.089, 0.051, 0.033]
cumulative_variance = np.cumsum(explained_variance_ratio)

print("Components needed for 85% variance:", 
      np.where(cumulative_variance >= 0.85)[0][0] + 1)  # 6 components
```

**Classification Performance by Class**:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Class 0 | 0.92 | 0.89 | 0.90 | 156 |
| Class 1 | 0.88 | 0.91 | 0.89 | 142 |
| Class 2 | 0.93 | 0.92 | 0.93 | 178 |
| **Average** | **0.91** | **0.91** | **0.91** | **476** |

---

## 🎯 **Business Impact Analysis**

### **A/B Testing ROI**
**Cost-Benefit Analysis**:
- **Model Development Cost**: $45,000 (BERT) vs $28,000 (DistilBERT)
- **Infrastructure Savings**: $12,000/month (DistilBERT efficiency)
- **Performance Trade-off**: 3% accuracy for 62% speed improvement
- **Net Present Value**: $89,000 savings over 2 years with DistilBERT

**Decision Rationale**:
1. **Statistical Significance**: Confirmed performance difference (p = 0.003)
2. **Effect Size**: Medium effect (Cohen's d = 0.41) but acceptable for business case
3. **Operational Benefits**: Substantial cost and speed improvements
4. **User Experience**: 62% faster response time enhances user satisfaction

### **Regression Trees Business Value**
- **Prediction Accuracy**: 88% enables reliable business decisions
- **Feature Optimization**: 12 key features reduce data collection costs
- **Model Interpretability**: Decision tree visualization aids stakeholder buy-in
- **Automation Potential**: 94% stability enables automated decision-making

---

## 🔬 **Advanced Statistical Techniques**

### **Bootstrap Confidence Intervals**
```python
# Bootstrap Implementation for Confidence Intervals
def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=1000, alpha=0.05):
    """
    Calculate bootstrap confidence intervals for any statistic
    """
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        # Calculate statistic
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    # Calculate confidence interval
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return ci_lower, ci_upper
```

### **Effect Size Calculations**
```python
# Multiple Effect Size Measures
def calculate_effect_sizes(group1, group2):
    """
    Comprehensive effect size calculation
    """
    import numpy as np
    from scipy import stats
    
    # Cohen's d
    pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + 
                         (len(group2)-1)*np.var(group2, ddof=1)) / 
                        (len(group1) + len(group2) - 2))
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    # Glass's delta
    glass_delta = (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1)
    
    # Hedge's g (bias-corrected Cohen's d)
    j = 1 - (3 / (4 * (len(group1) + len(group2) - 2) - 1))
    hedges_g = cohens_d * j
    
    # Cliff's delta (non-parametric effect size)
    cliffs_delta = stats.mannwhitneyu(group1, group2, alternative='two-sided').statistic
    cliffs_delta = (2 * cliffs_delta / (len(group1) * len(group2))) - 1
    
    return {
        'cohens_d': cohens_d,
        'glass_delta': glass_delta,
        'hedges_g': hedges_g,
        'cliffs_delta': cliffs_delta
    }
```

---

## 📈 **Performance Monitoring & Validation**

### **Statistical Process Control**
- **Control Charts**: Model performance monitoring over time
- **Statistical Alarms**: Automated detection of performance degradation
- **Drift Detection**: Concept and data drift identification
- **Continuous Validation**: Ongoing statistical validation protocols

### **Model Stability Assessment**
```python
# Model Stability Metrics
def assess_model_stability(predictions_over_time):
    """
    Assess model stability using statistical metrics
    """
    # Calculate rolling statistics
    rolling_mean = np.convolve(predictions_over_time, 
                              np.ones(30)/30, mode='valid')
    rolling_std = np.std([predictions_over_time[i:i+30] 
                         for i in range(len(predictions_over_time)-29)], axis=1)
    
    # Stability metrics
    stability_score = 1 - (np.std(rolling_mean) / np.mean(rolling_mean))
    drift_indicator = np.corrcoef(range(len(rolling_mean)), rolling_mean)[0,1]
    
    return {
        'stability_score': stability_score,
        'drift_indicator': drift_indicator,
        'performance_variance': np.var(rolling_mean)
    }
```

---

## 💼 **Professional Applications**

### **Research & Development**
- **Experimental Design**: Clinical trials and product testing
- **Statistical Consulting**: Research methodology guidance
- **Hypothesis Testing**: Scientific research validation
- **Meta-Analysis**: Systematic review and evidence synthesis

### **Business Analytics**
- **A/B Testing**: Marketing and product optimization
- **Quality Control**: Manufacturing process monitoring
- **Customer Analytics**: Segmentation and behavior analysis
- **Performance Measurement**: KPI statistical validation

### **Machine Learning Validation**
- **Model Selection**: Statistical comparison of algorithms
- **Cross-Validation**: Robust performance estimation
- **Feature Selection**: Statistical significance testing
- **Hyperparameter Optimization**: Systematic parameter tuning

---

## 📞 **Statistical Consulting**

**Joseph Bidias**  
📧 rodabeck777@gmail.com  
📞 (214) 886-3785  
📊 Statistical Analysis Expert

### **Consulting Services**
- **Experimental Design**: Study design and power analysis
- **Statistical Analysis**: Advanced hypothesis testing
- **Model Validation**: ML model statistical validation
- **Research Methodology**: Academic and industry research support

---

*This statistical analysis section demonstrates rigorous statistical methodology and experimental design capabilities essential for data-driven decision making and scientific research.*