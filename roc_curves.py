import pandas as pd

import matplotlib.pyplot as plt

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

from sklearn.metrics import roc_curve, roc_auc_score


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
# 4. CREATE MODELS
# ============================================

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ))
])


knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5
    ))
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
# 5. BASE ESTIMATORS
# ============================================

estimators = [
    ("svm", svm_model),
    ("knn", knn_model),
    ("rf", rf_model),
    ("gb", gb_model)
]


# ============================================
# 6. VOTING ENSEMBLE
# ============================================

voting_model = VotingClassifier(
    estimators=estimators,
    voting="soft"
)


# ============================================
# 7. STACKING ENSEMBLE
# ============================================

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(
        max_iter=1000
    ),
    cv=5
)


# ============================================
# 8. STORE ALL MODELS
# ============================================

models = {
    "SVM": svm_model,
    "k-NN": knn_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Voting Ensemble": voting_model,
    "Stacking Ensemble": stacking_model
}


# ============================================
# 9. CREATE ROC GRAPH
# ============================================

plt.figure(figsize=(10, 7))


# ============================================
# 10. TRAIN EACH MODEL AND CALCULATE ROC
# ============================================

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    probability = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probability
    )

    auc_score = roc_auc_score(
        y_test,
        probability
    )

    print(
        f"{name} AUC: {auc_score:.4f}"
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc_score:.3f})"
    )


# ============================================
# 11. RANDOM CLASSIFIER LINE
# ============================================

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


# ============================================
# 12. GRAPH SETTINGS
# ============================================

plt.title(
    "ROC Curves - Smart Disease Prediction System"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend(
    loc="lower right"
)

plt.grid(True)

plt.tight_layout()

plt.show()