import os
import joblib


import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================
# 1. LOAD DATASET
# ============================================

data = pd.read_csv("data/heart_disease_cleaned.csv")


# ============================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================

X = data.drop("num", axis=1)

y = data["num"]


# ============================================
# 3. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================
# 4. CREATE BASE MODELS
# ============================================

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])


knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)


# ============================================
# 5. CREATE LIST OF BASE MODELS
# ============================================

estimators = [
    ("svm", svm_model),
    ("knn", knn_model),
    ("rf", rf_model),
    ("gb", gb_model)
]
# ============================================
# 6. INDIVIDUAL MODEL ACCURACY
# ============================================

# Train individual models
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Predictions
svm_prediction = svm_model.predict(X_test)
knn_prediction = knn_model.predict(X_test)
rf_prediction = rf_model.predict(X_test)
gb_prediction = gb_model.predict(X_test)

# Calculate accuracies automatically
svm_accuracy = accuracy_score(y_test, svm_prediction)
knn_accuracy = accuracy_score(y_test, knn_prediction)
rf_accuracy = accuracy_score(y_test, rf_prediction)
gb_accuracy = accuracy_score(y_test, gb_prediction)

print("\n============================================")
print("       INDIVIDUAL MODEL ACCURACIES")
print("============================================")

print(f"SVM Accuracy              : {svm_accuracy * 100:.2f}%")
print(f"k-NN Accuracy             : {knn_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy    : {rf_accuracy * 100:.2f}%")
print(f"Gradient Boosting Accuracy: {gb_accuracy * 100:.2f}%")


# ============================================
# 6. VOTING ENSEMBLE
# ============================================

voting_model = VotingClassifier(
    estimators=estimators,
    voting="hard"
)
# ==========================================
# ============================================
# 7. TRAIN VOTING MODEL
# ============================================

voting_model.fit(X_train, y_train)



os.makedirs("models", exist_ok=True)
joblib.dump(voting_model, "models/voting_model.joblib")
joblib.dump(list(X.columns), "models/feature_columns.joblib")

print("Voting model saved successfully!")


# ============================================
# 8. VOTING PREDICTION
# ============================================

voting_prediction = voting_model.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_prediction)
voting_cm = confusion_matrix(y_test, voting_prediction)


# ============================================
# 9. PATIENT PREDICTION
# ============================================

print("\n==========================================")
print("       HEART DISEASE PREDICTION")
print("==========================================")

print("\nEnter Patient Details:")

age = float(input("Age: "))
sex = float(input("Sex (0 = Female, 1 = Male): "))
cp = float(input("Chest Pain Type (0-3): "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol: "))
fbs = float(input("Fasting Blood Sugar (0/1): "))
restecg = float(input("Resting ECG (0-2): "))
thalach = float(input("Maximum Heart Rate: "))
exang = float(input("Exercise Induced Angina (0/1): "))
oldpeak = float(input("Oldpeak: "))
slope = float(input("Slope (0-2): "))
ca = float(input("Number of Major Vessels (0-4): "))
thal = float(input("Thalassemia (0-3): "))


# Create patient data using the SAME feature names
patient_data = pd.DataFrame([[
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]], columns=X.columns)


# Predict using trained Voting Ensemble
prediction = voting_model.predict(patient_data)[0]


print("\n==========================================")
print("              PREDICTION")
print("==========================================")

if prediction == 1:
    print("⚠ HEART DISEASE PREDICTED")
else:
    print("✓ NO HEART DISEASE PREDICTED")

print("==========================================")
# ============================================
# 10. STACKING ENSEMBLE
# ============================================

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)


# ============================================
# 11. TRAIN STACKING MODEL
# ============================================

stacking_model.fit(X_train, y_train)


# ============================================
# 12. STACKING PREDICTION
# ============================================

stacking_prediction = stacking_model.predict(X_test)


# ============================================
# 13. STACKING ACCURACY
# ============================================

stacking_accuracy = accuracy_score(
    y_test,
    stacking_prediction
)


print("\n====================================")
print("STACKING ENSEMBLE")
print("====================================")

print("\nStacking Ensemble Accuracy:")
print(f"{stacking_accuracy * 100:.2f}%")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        stacking_prediction
    )
)


stacking_cm = confusion_matrix(
    y_test,
    stacking_prediction
)


print("\nConfusion Matrix:")

print(stacking_cm)


# ============================================
# 14. CONFUSION MATRIX - VOTING
# ============================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    voting_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title("Voting Ensemble Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.show()


# ============================================
# 15. CONFUSION MATRIX - STACKING
# ============================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    stacking_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title("Stacking Ensemble Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.show()


# ============================================
# 16. MODEL COMPARISON
# ============================================

print("\n====================================")
print("MODEL COMPARISON")
print("====================================")

print("\n============================================")
print("        MODEL ACCURACY COMPARISON")
print("============================================")

print(f"SVM Accuracy              : {svm_accuracy * 100:.2f}%")
print(f"k-NN Accuracy             : {knn_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy    : {rf_accuracy * 100:.2f}%")
print(f"Gradient Boosting Accuracy: {gb_accuracy * 100:.2f}%")
print(f"Voting Ensemble Accuracy  : {voting_accuracy * 100:.2f}%")
print(f"Stacking Ensemble Accuracy: {stacking_accuracy * 100:.2f}%")

print("============================================")