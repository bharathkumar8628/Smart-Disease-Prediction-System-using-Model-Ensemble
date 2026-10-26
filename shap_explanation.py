import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import shap


# ============================================
# 1. LOAD DATASET
# ============================================

data = pd.read_csv("data/heart_disease_cleaned.csv")


# ============================================
# 2. FEATURES AND TARGET
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
# 4. TRAIN RANDOM FOREST
# ============================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================
# 5. CREATE SHAP EXPLAINER
# ============================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)


# ============================================
# 6. HANDLE SHAP OUTPUT
# ============================================

print("SHAP values shape:")

try:
    print(shap_values.shape)
except:
    print("SHAP returned a list")


# For newer SHAP versions
if hasattr(shap_values, "ndim") and shap_values.ndim == 3:

    # Select class 1 = disease
    shap_values = shap_values[:, :, 1]


# For older SHAP versions
elif isinstance(shap_values, list):

    shap_values = shap_values[1]


# ============================================
# 7. SHAP SUMMARY PLOT
# ============================================

print("Generating SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.title(
    "SHAP Feature Importance - Random Forest"
)

plt.tight_layout()

plt.show()