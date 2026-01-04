import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Evaluation Functions
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n--- {model_name} ---")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


def collect_results(y_true, y_pred, model_name, results):
    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    })


df = pd.read_csv("Telco-Customer-Churn.csv")


# Data Preprocessing
# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop non-predictive column
df.drop(columns=["customerID"], inplace=True)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Feature / Target Split
X = df.drop("Churn", axis=1)
y = df["Churn"]


# Train–Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling (Standard Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Model Training
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)


# Model Evaluation
evaluate_model(y_test, y_pred_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_knn, "KNN")
evaluate_model(y_test, y_pred_dt, "Decision Tree")


# Model Comparison
results = []
collect_results(y_test, y_pred_lr, "Logistic Regression", results)
collect_results(y_test, y_pred_knn, "KNN", results)
collect_results(y_test, y_pred_dt, "Decision Tree", results)

results_df = pd.DataFrame(results)
print("\nModel Comparison (sorted by Recall):\n")
print(results_df.sort_values(by="Recall", ascending=False))


# Visualization (Matplotlib)
# Recall comparison
plt.figure()
plt.bar(results_df["Model"], results_df["Recall"])
plt.xlabel("Model")
plt.ylabel("Recall")
plt.title("Recall Comparison Across Models")
plt.ylim(0, 1)
plt.show()

# F1 score comparison
plt.figure()
plt.bar(results_df["Model"], results_df["F1 Score"])
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.title("F1 Score Comparison Across Models")
plt.ylim(0, 1)
plt.show()