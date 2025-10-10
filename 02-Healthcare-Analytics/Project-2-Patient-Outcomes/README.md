# Patient Outcomes & Cost Optimization Model

## **Objective**
This project examines how hospital staffing levels impact **patient outcomes** and **hospital costs**, using **Random Forest Regression**.

## **Dataset Used**
- `data3.csv`: Contains hospital staffing and productivity data.
- `data1.xlsx`: Contains hospital financial and operational costs.

## **Methodology**
1. **Data Cleaning & Preprocessing:**
   - Used **fuzzy matching** to align hospital names across datasets.
   - Converted relevant columns to numeric values.
   - Removed missing values.

2. **Data Analysis & Visualization:**
   - Examined staffing levels vs. hospital costs.
   - Feature importance analysis.
   - Scatter plot of **Actual vs. Predicted Total Drug Costs**.

3. **Predictive Modeling (Random Forest Regression):**
   - Built a **Random Forest model** to predict **Total Drug Costs** from **staffing levels**.
   - Evaluated model performance using **error distribution**.

## **Results**
- Successfully matched **425 hospitals** across datasets.
- **Higher productive hours correlated with lower costs.**
- **Model accurately predicts total drug costs.**
- **Productive hours per adjusted patient day is the most important feature.**

## **Visualizations**
- **Scatter Plot:** Actual vs. Predicted Total Drug Costs.
- **Error Distribution Plot:** Analyzing prediction accuracy.
- **Feature Importance Plot:** Identifying key drivers of costs.

## **How to Run the Code**
### 1. Install Dependencies

pip install pandas numpy matplotlib seaborn scikit-learn fuzzywuzzy openpyxl rapidfuzz

###2. Run the Python Script(bash)

python patient_outcomes.py


##Key Takeaways

✅ Hospitals with higher staffing efficiency have lower costs.
✅ The model successfully predicts total drug costs based on staffing levels.
✅ Feature importance analysis helps optimize hospital resource allocation.




### **Next Steps**
1. **Expand the model to include more cost variables** (e.g., patient severity, hospital size).
2. **Deploy the model into a dashboard** for real-time cost predictions.
3. **Explore other machine learning techniques** like **XGBoost** for better performance.



