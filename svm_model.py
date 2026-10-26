import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
# 2. Separate X and y
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
# 4. Feature Scaling
# --------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# --------------------------------
# 5. Create SVM model
# --------------------------------

svm_model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)


# --------------------------------
# 6. Train the model
# --------------------------------

svm_model.fit(X_train_scaled, y_train)


# --------------------------------
# 7. Make predictions
# --------------------------------

y_pred = svm_model.predict(X_test_scaled)


# --------------------------------
# 8. Calculate accuracy
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nSVM Model Accuracy:")
print(f"{accuracy * 100:.2f}%")


# --------------------------------
# 9. Classification Report
# --------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------
# 10. Confusion Matrix
# --------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------
# 11. Display Confusion Matrix
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

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()