"""
Heart Disease Model Training and Serialization Script
Trains and saves all classifiers and ensemble models for the Heart Disease dataset,
including performance metrics, ROC curves, feature importances, and SHAP explainers.
"""

import os
import warnings
import joblib
import numpy as np
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
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import shap

# Suppress warnings
warnings.filterwarnings("ignore")

def train_and_save_heart_disease_models():
    csv_path = "data/heart_disease_cleaned.csv"
    target_col = "num"
    dataset_name = "Heart Disease"
    
    print(f"\n==========================================")
    print(f"Training models for {dataset_name} ({csv_path})")
    print(f"==========================================")
    
    df = pd.read_csv(csv_path)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    feature_names = list(X.columns)
    
    # Train / Test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Base models
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42))
    ])
    
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    estimators = [
        ("svm", svm_pipeline),
        ("knn", knn_pipeline),
        ("rf", rf_model),
        ("gb", gb_model)
    ]
    
    # Ensemble Models
    voting_model = VotingClassifier(estimators=estimators, voting="soft")
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
    
    all_models = {
        "Voting Ensemble": voting_model,
        "Stacking Ensemble": stacking_model,
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "Support Vector Machine (SVM)": svm_pipeline,
        "k-Nearest Neighbors (k-NN)": knn_pipeline
    }
    
    metrics = {}
    roc_data = {}
    conf_matrices = {}
    reports = {}
    
    for name, model in all_models.items():
        print(f"-> Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        }
        
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_data[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": auc
        }
        
        conf_matrices[name] = cm.tolist()
        reports[name] = report
        
        print(f"   Accuracy: {acc*100:.2f}% | AUC: {auc:.4f} | F1: {f1:.4f}")
    
    # Feature Importance (Random Forest and Gradient Boosting)
    rf_importance = dict(zip(feature_names, rf_model.feature_importances_.tolist()))
    gb_importance = dict(zip(feature_names, gb_model.feature_importances_.tolist()))
    
    # SHAP Explainer on Random Forest
    print("-> Computing SHAP values on Random Forest...")
    explainer = shap.TreeExplainer(rf_model)
    shap_vals_raw = explainer.shap_values(X_test)
    
    if hasattr(shap_vals_raw, "ndim") and shap_vals_raw.ndim == 3:
        shap_vals = shap_vals_raw[:, :, 1]
    elif isinstance(shap_vals_raw, list):
        shap_vals = shap_vals_raw[1]
    else:
        shap_vals = shap_vals_raw
    
    bundle = {
        "dataset_name": dataset_name,
        "feature_names": feature_names,
        "target_col": target_col,
        "models": all_models,
        "metrics": metrics,
        "roc_data": roc_data,
        "conf_matrices": conf_matrices,
        "reports": reports,
        "rf_importance": rf_importance,
        "gb_importance": gb_importance,
        "shap_values": shap_vals,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "rf_explainer": explainer
    }
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, "models/heart_disease_bundle.joblib")
    joblib.dump(voting_model, "models/voting_model.joblib")
    joblib.dump(feature_names, "models/feature_columns.joblib")
    print("Heart Disease models and artifacts saved successfully!")
    return bundle

def main():
    train_and_save_heart_disease_models()

if __name__ == "__main__":
    main()
