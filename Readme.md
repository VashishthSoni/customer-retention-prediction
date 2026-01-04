# Customer Churn Prediction (Supervised ML)

## Problem Statement
The goal of this project is to predict whether a customer will churn (leave the service) using historical customer data. This is a binary classification problem where correctly identifying churn customers is critical for business retention strategies.

---

## Dataset
- Telco Customer Churn Dataset
- ~7,000 customer records
- Mix of categorical and numerical features
- Target variable: `Churn` (Yes / No)

---

## Machine Learning Pipeline
1. Data loading and inspection
2. Handling missing values (`TotalCharges`)
3. Encoding categorical variables using One-Hot Encoding
4. Train–test split with stratification
5. Feature scaling using StandardScaler
6. Model training
7. Model evaluation and comparison

---

## Models Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Due to class imbalance, **Recall and F1-score** were prioritized over accuracy.

---

## Final Model Selection
Logistic Regression was selected as the final model because it achieved higher recall compared to other models. In churn prediction, missing a churn customer is more costly than incorrectly flagging a non-churn customer. A custom decision threshold was also applied to further improve recall.

---

## How to Run
```bash
pip install -r requirements.txt
python churn_model.py
