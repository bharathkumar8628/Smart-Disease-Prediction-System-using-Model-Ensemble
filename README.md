# 🫀 Heart Disease Prediction System

A Clinical Machine Learning and Explainable AI (XAI) Web Platform for coronary heart disease diagnosis, risk profiling, and biomarker interpretation.

---

## 🌟 Key Features

- **Clinical Medical Dataset**:
  - UCI Cleveland Heart Disease Dataset (13 physiological & cardiac biomarkers, 299 patient records)
- **Comprehensive Classifiers**:
  - Support Vector Machine (SVM) with RBF Kernel & Probability Calibration
  - k-Nearest Neighbors (k-NN) with StandardScaler Pipeline
  - Random Forest Classifier (100 Estimators)
  - Gradient Boosting Classifier (Learning rate 0.1, max depth 3)
- **Advanced Ensemble Learning**:
  - **Soft Voting Ensemble**: Consensus probability aggregation across all 4 base estimators
  - **Stacking Ensemble**: Meta-classification using Logistic Regression
- **Clinical Model Evaluation**:
  - Accuracy, Precision, Recall, F1-Score, and ROC-AUC Benchmarking
  - Multi-model Receiver Operating Characteristic (ROC) curves with AUC
  - Interactive Confusion Matrix inspector with Sensitivity and Specificity metrics
- **Explainable AI (XAI)**:
  - Global Feature Importance (Random Forest MDI & Gradient Boosting)
  - SHAP (SHapley Additive exPlanations) summary rankings identifying critical biomarkers (`ca`, `thal`, `cp`, `oldpeak`)
- **Interactive Web Interface**:
  - Single Patient Clinical Predictor with live risk gauge and multi-model consensus
  - Quick-Fill presets (Healthy, High-Risk, Borderline)
  - Batch CSV patient record screening with CSV export
  - Interactive Exploratory Data Analysis (EDA) and correlation heatmaps

---

## 🚀 How to Run the Web Application

### Option 1: Double-click
Double-click `run_app.bat` in the project root directory.

### Option 2: Terminal / PowerShell
```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Launch the Streamlit application
streamlit run src/app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 📁 Project Structure

```
Smart Disease Prediction/
│
├── data/
│   └── heart_disease_cleaned.csv     # Cleaned UCI Heart Disease dataset
│
├── models/
│   ├── heart_disease_bundle.joblib   # Trained models, metrics & SHAP artifacts
│   ├── voting_model.joblib           # Pre-saved Voting Ensemble
│   └── feature_columns.joblib        # Feature schema
│
├── src/
│   ├── app.py                        # Streamlit Web Application
│   ├── train_models.py               # Automated pipeline training & serialization
│   ├── ensemble_model.py             # Terminal ensemble script (Voting & Stacking)
│   ├── roc_curves.py                 # Terminal ROC curve plotting
│   ├── feature_importance.py         # Terminal Random Forest feature importance
│   ├── shap_explanation.py           # Terminal SHAP summary script
│   ├── svm_model.py                  # Terminal SVM classifier
│   ├── knn_model.py                  # Terminal k-NN classifier
│   ├── random_forest_model.py        # Terminal Random Forest classifier
│   ├── gradient_boosting_model.py    # Terminal Gradient Boosting classifier
│   ├── preprocess.py                 # UCI dataset downloader & cleaner
│   └── eda.py                        # Terminal Exploratory Data Analysis
│
├── run_app.bat                       # 1-click launcher for Windows
└── README.md                         # Documentation
```

---

## 📊 Model Performance Highlights

| Model | Accuracy | ROC-AUC | F1-Score |
| :--- | :---: | :---: | :---: |
| **k-Nearest Neighbors (k-NN)** | **88.33%** | 0.9492 | **0.8679** |
| **Voting Ensemble (Soft)** | **85.00%** | **0.9509** | **0.8302** |
| **Support Vector Machine (SVM)** | **85.00%** | **0.9542** | **0.8302** |
| **Random Forest** | **85.00%** | 0.9408 | **0.8302** |
| **Stacking Ensemble** | 83.33% | **0.9542** | 0.8077 |
| **Gradient Boosting** | 80.00% | 0.9007 | 0.7778 |

---

## 🔬 Explainable AI (SHAP) Insights

The SHAP framework explains which physiological signals dictate heart disease risk:
1. **Coronary Blood Flow (`ca`)**: Number of major vessels colored by fluoroscopy is the most decisive predictor.
2. **Reversible Perfusion Defect (`thal`)**: Thalassemia status indicates myocardial ischemia.
3. **Exercise ST Depression (`oldpeak`)**: Directly reflects cardiac stress tolerance.
4. **Chest Pain Classification (`cp`)**: Distinguishes typical angina from asymptomatic presentations.
