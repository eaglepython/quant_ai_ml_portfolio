

# Healthcare Financial Performance Dashboard

## **Objective**
This project analyzes hospital financial performance by visualizing revenue trends and building a predictive model using **Linear Regression**.

## **Dataset Used**
- `data2.csv`: Contains hospital revenue and system information.
- `data1.xlsx`: Contains hospital financial and operational costs.

## **Methodology**
1. **Data Cleaning & Preprocessing:**
   - Fuzzy matching was used to align hospital names across datasets.
   - Missing values were removed.
   - Revenue columns were converted to numerical values.

2. **Data Analysis & Visualization:**
   - Distribution of total and net revenue.
   - Correlation analysis between financial variables.
   - Scatter plot of **Actual vs Predicted Net Revenue**.

3. **Predictive Modeling (Linear Regression):**
   - Built a **Linear Regression** model to predict **Net Revenue** based on **Total Revenue**.
   - Evaluated model performance using residual analysis.

## **Results**
- **Successfully matched 6 hospitals** across datasets.
- **Strong correlation** between total revenue and net revenue.
- **Linear Regression model provides accurate predictions**, with some room for improvement.

## **Visualizations**
- **Revenue Distribution:** Histogram of hospital revenues.
- **Correlation Heatmap:** Relationships between revenue, drug costs, and overhead labor costs.
- **Scatter Plot:** Actual vs. Predicted Net Revenue.
- **Residual Plot:** Analyzing prediction errors.

## **How to Run the Code**
### 1. Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn fuzzywuzzy openpyxl





Key Takeaways

✅ Hospitals with higher total revenue tend to have higher net revenue.
✅ The model effectively predicts net revenue but can be improved with additional cost-related variables.
✅ Correlation analysis helps identify which cost factors impact revenue the most.



## **Next Steps**
1. **Apply Feature Engineering**  
   - Incorporate additional hospital operational metrics like **patient volume** or **service mix** to improve revenue predictions.

2. **Deploy the Model**  
   - Convert the model into a **web application** using **Streamlit** for an interactive dashboard.

3. **Expand Analysis to Other Projects**  
   - Integrate this financial model with **hospital staffing optimization** to improve cost efficiency.



