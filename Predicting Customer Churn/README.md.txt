# Customer Churn Prediction

This project predicts whether a customer will cancel their subscription (churn) in the upcoming billing cycle, allowing for targeted interventions.

## 1. Overview
- **Data Source**: `synthetic_churn_data.csv` (generated or provided).
- **Goal**: Build a robust model to anticipate churn and inform business decisions.

## 2. Steps
1. **Data Inspection & EDA**: Check data structure, visualize distributions, correlation with churn.
2. **Feature Engineering**: Convert categorical features, create new metrics (e.g., usage × tier).
3. **Modeling**: Logistic Regression (baseline), Random Forest / XGBoost (advanced).
4. **Evaluation**: AUC, Precision-Recall, feature importances, SHAP values for deeper insight.

## 3. Repository Structure
- `data/`: Contains the churn dataset.
- `notebooks/`: Jupyter notebooks for EDA, modeling, and evaluation.
- `scripts/`: Modular Python files for data preprocessing and the main pipeline.
- `requirements.txt`: Lists needed Python packages (pandas, numpy, scikit-learn, matplotlib, etc.).

## 4. Usage
1. Clone or download this repo.
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt