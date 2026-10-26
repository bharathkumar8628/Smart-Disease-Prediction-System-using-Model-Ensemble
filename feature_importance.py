import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
# 5. GET FEATURE IMPORTANCE
# ============================================

importance = model.feature_importances_


# ============================================
# 6. CREATE DATAFRAME
# ============================================

feature_data = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})


# ============================================
# 7. SORT FEATURES
# ============================================

feature_data = feature_data.sort_values(
    by="Importance",
    ascending=False
)


print("\nFeature Importance:")
print(feature_data)


# ============================================
# 8. PLOT GRAPH
# ============================================

plt.figure(figsize=(10, 6))

plt.bar(
    feature_data["Feature"],
    feature_data["Importance"]
)

plt.title(
    "Feature Importance - Random Forest"
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Importance"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()