"""
Research Prototype: Predictive Maintenance

Purpose:
Quickly experiment with a Random Forest model.
This is exploratory code, not the production pipeline.
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# 1. Load processed dataset
data = pd.read_csv("data/processed/processed_predictive_maintenance.csv")

features = [
    "metric1", "metric2", "metric3", "metric4",
    "metric5", "metric6", "metric7", "metric9",
    "day_of_week", "month", "day_of_month",
]

X = data[features]
y = data["failure"]

# 2. Split data for an initial experiment
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 3. Train a Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
)

model.fit(X_train, y_train)

# 4. Evaluate the experiment
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Research Prototype Results")
print("--------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(
    "PR-AUC:",
    average_precision_score(y_test, probabilities),
)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))