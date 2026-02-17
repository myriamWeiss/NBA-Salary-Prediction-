# NBA-Salary-Prediction-
NBA Salary Prediction Using Linear &amp; Regularized Regression Models

This project builds and compares multiple regression models to explain and predict NBA players' annual salaries based on performance statistics.

## Objective

To model the relationship between player performance metrics and salary, and evaluate different regression approaches to determine which provides the best predictive performance.

The target variable is:
- Log-transformed annual salary

---

## Dataset

The dataset includes NBA players across multiple seasons with features such as:

- Minutes Played (MIN)
- Field Goals Made (FGM)
- 3-Point Field Goals (3PM)
- Free Throws Made (FTM)
- Free Throw Percentage (FT%)
- Rebounds (REB)
- Assists (AST)
- Steals (STL)
- Blocks (BLK)
- Turnovers (TO)

Text-based variables (player, team, position) were removed.

---

## Preprocessing

- Missing values removed
- Log transformation applied to salary
- Feature correlation analysis performed
- Train/Test split (80/20)
- Feature scaling applied for regularized models

---

## Models Implemented

### 1️ OLS (Ordinary Least Squares)
- Forward Selection based on Test MSE
- Final model selected using lowest validation error
- Performance:
  - MSE ≈ 0.99
  - R² ≈ 0.40

### 2️ PCR (Principal Component Regression)
- PCA applied before regression
- Cross-validation used to select number of components
- Performance similar to OLS

### 3️ PLS (Partial Least Squares)
- Components optimized via Cross-Validation
- Reduced dimensionality without loss of performance

### 4️ Ridge Regression
- Regularization parameter (alpha) explored
- Coefficient shrinkage analyzed

### 5️ Lasso Regression
- L1 regularization
- Feature selection via coefficient shrinkage

---

## Architecture

The project includes a modular regression framework:

- Abstract base class for regression models
- Separate Ridge and Lasso implementations
- Flexible alpha selection pipeline
- Cross-validation evaluation

---

## Key Insights

- Salary distribution is heavily right-skewed → log transformation improved model assumptions.
- Offensive contribution and playing time strongly correlate with salary.
- Regularization did not significantly improve predictive performance compared to OLS.
- Dimensionality reduction (PCR/PLS) did not outperform full-feature models.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Statsmodels
- Matplotlib / Seaborn


