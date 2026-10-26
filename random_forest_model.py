import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------
# 1. Load cleaned dataset
# --------------------------------

data = pd.read_csv("data/heart_disease_cleaned.csv")


# --------------------------------
# 2. Separate features and target
# --------------------------------

X = data.drop("num", axis=1)

y = data["num"]


# --------------------------------
# 3. Train/Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# 4. Create Random Forest model
# --------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------
# 5. Train the model
# --------------------------------

rf_model.fit(X_train, y_train)


# --------------------------------
# 6. Make predictions
# --------------------------------

y_pred = rf_model.predict(X_test)


# --------------------------------
# 7. Calculate accuracy
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Model Accuracy:")
print(f"{accuracy * 100:.2f}%")


# --------------------------------
# 8. Classification Report
# --------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------
# 9. Confusion Matrix
# --------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------
# 10. Display Confusion Matrix
# --------------------------------

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()