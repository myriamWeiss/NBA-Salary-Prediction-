# NBA Salary Prediction – Regression & Regularization Analysis

## Executive Summary

This project analyzes the relationship between NBA player performance metrics 
and salary using linear regression models.

OLS, Ridge, and Lasso regression were implemented and compared 
to evaluate the impact of multicollinearity and regularization 
on predictive performance.

The final model explains ~43% of salary variance, suggesting that 
while performance is a key driver of compensation, 
non-performance factors play a substantial role.

---

## Project Objectives

- Perform structured Exploratory Data Analysis (EDA)
- Apply log transformation to address skewness
- Detect and analyze multicollinearity
- Compare OLS, Ridge, and Lasso regression
- Evaluate model performance using test data
- Translate modeling results into business insights

---

## Methodology

1. Data cleaning and preprocessing  
2. Log transformation of salary  
3. Correlation and multicollinearity analysis  
4. Train-test split (80/20)  
5. Feature scaling  
6. Model training and cross-validation  
7. Performance comparison (MSE, R²)

---

## Model Performance

| Model | MSE | R² |
|-------|------|------|
| OLS | 0.953 | 0.427 |
| Ridge | 0.952 | 0.427 |
| Lasso | 0.952 | 0.427 |

Regularization provided marginal improvement over OLS, 
indicating that multicollinearity exists but does not severely 
impact generalization.

---

## Key Insights

- Points scored (PTS) is the strongest salary predictor.
- Salary is influenced by multiple performance metrics.
- Extreme contracts are likely driven by non-performance factors.
- Regularization stabilizes coefficients but does not dramatically improve accuracy.

---

## Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## Author

Myriam – Industrial Engineering & Management  
Focused on data analysis and applied machine learning.
