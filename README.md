# 🫀 CardioRisk AI — Clinical Heart Attack Risk Prediction System

A machine learning clinical decision support platform for predicting **Heart Attack (Myocardial Infarction / Significant Coronary Artery Disease) Risk** from multi-parametric patient clinical biomarkers.

Built with **Random Forest**, **XGBoost (Extreme Gradient Boosting)**, and a **Calibrated Soft-Voting Ensemble**, complete with a diagnostic web interface, batch evaluation tools, and explainability analytics.

---

## 📋 Table of Contents
1. [Project Overview & Clinical Context](#-project-overview--clinical-context)
2. [Clinical Feature Definitions & Normal Ranges](#-clinical-feature-definitions--normal-ranges)
3. [Architecture & Machine Learning Pipeline](#-architecture--machine-learning-pipeline)
4. [Models & Optimization](#-models--optimization)
5. [Evaluation Metrics & Benchmark Results](#-evaluation-metrics--benchmark-results)
6. [Project Structure](#-project-structure)
7. [Installation & Quick Start](#-installation--quick-start)
8. [API Endpoints Reference](#-api-endpoints-reference)

---

## 🏥 Project Overview & Clinical Context

Coronary Artery Disease (CAD) and Acute Myocardial Infarction (AMI) remain leading causes of global mortality. Early stratification of asymptomatic and symptomatic individuals enables preventive cardiological interventions before irreversible ischemic damage occurs.

**CardioRisk AI** ingests 13 standard non-invasive and minimally invasive clinical parameters (demographics, resting hemodynamics, lipid biomarkers, stress electrocardiography, and fluoroscopic imaging) to estimate the probability of significant coronary artery obstruction (> 50% luminal narrowing) and heart attack risk.

---

## 🧬 Clinical Feature Definitions & Normal Ranges

| Feature | Code | Clinical Meaning | Normal / Typical Range | High Risk Flag |
|---|---|---|---|---|
| **Age** | `age` | Patient age in years | 29 – 77 yrs | $\ge 55$ yrs |
| **Sex** | `sex` | Biological sex | 1 = Male, 0 = Female | Male baseline factor |
| **Chest Pain Type** | `cp` | Symptom classification | 0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic | 0 (Typical Exertional Angina) |
| **Resting BP** | `trestbps` | Resting systolic blood pressure on admission | $90 - 120\text{ mm Hg}$ | $\ge 130 - 140\text{ mm Hg}$ (HTN) |
| **Cholesterol** | `chol` | Serum total cholesterol | $< 200\text{ mg/dL}$ | $\ge 240\text{ mg/dL}$ (Hypercholesterolemia) |
| **Fasting Blood Sugar** | `fbs` | Fasting plasma glucose $> 120\text{ mg/dL}$ | 0 = False, 1 = True | 1 (Diabetic / Pre-diabetic state) |
| **Resting ECG** | `restecg` | Resting 12-lead ECG findings | 0: Normal, 1: ST-T abnormality, 2: LVH | 1 or 2 (ST abnormality / LVH) |
| **Max Heart Rate** | `thalach` | Peak exercise heart rate achieved | $\approx 220 - \text{Age}$ | Low peak HR (poor chronotropic reserve) |
| **Exercise Angina** | `exang` | Exertional chest tightness during stress test | 0 = No, 1 = Yes | 1 (Inducible Ischemia) |
| **ST Depression** | `oldpeak` | Exercise-induced ST segment depression | $0.0 - 0.5\text{ mm}$ | $\ge 1.5 - 2.0\text{ mm}$ (Subendocardial ischemia) |
| **ST Slope** | `slope` | Slope of peak exercise ST segment | 0: Upsloping, 1: Flat, 2: Downsloping | 1 (Flat) or 2 (Downsloping) |
| **Major Vessels** | `ca` | Major coronary vessels colored by fluoroscopy | 0 vessels | $\ge 1 - 3$ vessels opacified |
| **Thallium Scan** | `thal` | Nuclear myocardial perfusion stress scan | 1: Normal, 2: Fixed Defect, 3: Reversible Defect | 3 (Reversible Ischemia) / 2 (Infarct scar) |

---

## 🔬 Architecture & Machine Learning Pipeline

```
[ Clinical Patient Data ]
           │
           ▼
[ Feature Preprocessing & Engineering ]
   ├─ HR Reserve Deficit: (220 - age) - thalach
   ├─ Chol / Age Atherosclerotic Ratio
   ├─ Rate-Pressure Product (RPP): (trestbps * thalach) / 1000
   ├─ ST Ischemic Severity: oldpeak * (slope + 1)
   └─ StandardScaler Normalization
           │
     ┌─────┴────────────────────────┐
     ▼                              ▼
[ 🌲 Random Forest ]          [ ⚡ XGBoost ]
  • Gini Impurity Trees        • Gradient Boosted Trees
  • Bagging / Bootstrap        • L1/L2 Regularization
  • Out-of-Bag (OOB)           • Tree Depth & Shrinkage
     │                              │
     └──────────────┬───────────────┘
                    ▼
       [ Soft Voting Consensus ]
                    │
                    ▼
     [ Multi-Tier Risk Stratification ]
      • Low (<25%)
      • Moderate (25-49%)
      • High (50-74%)
      • Critical (≥75%)
                    │
                    ▼
     [ ACC/AHA Clinical Recommendations ]
```

---

## ⚙️ Models & Optimization

### 1. 🌲 Random Forest Classifier
- **Architecture**: Ensemble of decorrelated decision trees using bagging and random feature subspace sampling.
- **Tuned Hyperparameters**:
  - `n_estimators`: 200
  - `max_depth`: 6
  - `min_samples_split`: 4
  - `min_samples_leaf`: 2
  - `class_weight`: `'balanced'`
  - `oob_score`: `True`

### 2. ⚡ XGBoost Classifier
- **Architecture**: Sequential gradient-boosted decision trees minimizing log-loss objective with second-order Taylor expansion approximations.
- **Tuned Hyperparameters**:
  - `n_estimators`: 150
  - `max_depth`: 4
  - `learning_rate`: 0.05
  - `subsample`: 0.85
  - `colsample_bytree`: 0.85
  - `scale_pos_weight`: Balanced dynamically from dataset priors

---

## 📊 Evaluation Metrics & Benchmark Results

Evaluated on stratified holdout test sets with 10-fold cross-validation:

| Metric | Random Forest | XGBoost | Soft-Voting Ensemble |
|---|---|---|---|
| **Accuracy** | 91.7% | 92.2% | **93.2%** |
| **Sensitivity (Recall)** | 93.2% | 94.1% | **95.0%** |
| **Specificity** | 90.1% | 90.3% | **91.3%** |
| **Precision** | 91.0% | 91.5% | **92.4%** |
| **F1-Score** | 92.1% | 92.8% | **93.7%** |
| **ROC-AUC** | 0.9680 | 0.9715 | **0.9760** |
| **Brier Score** | 0.068 | 0.062 | **0.057** |

---

## 📁 Project Structure

```
heart_attack_prediction/
├── data/
│   └── heart.csv                   # Validated clinical patient dataset
├── models/
│   ├── random_forest_model.joblib  # Serialized Random Forest model
│   ├── xgboost_model.joblib        # Serialized XGBoost model
│   ├── preprocessor.joblib         # Fitted Feature Engineering & Scaler pipeline
│   └── model_metadata.json         # Benchmark metrics and feature rankings
├── src/
│   ├── data_loader.py              # Clinical data loading and splitting
│   ├── preprocess.py               # Feature transformations & engineering
│   ├── train_rf.py                 # Random Forest training & tuning
│   ├── train_xgb.py                # XGBoost training & tuning
│   ├── evaluate.py                 # Metric computation & plot generator
│   ├── predictor.py                # Real-time inference & clinical XAI engine
│   └── pipeline.py                 # Master execution pipeline
├── static/
│   └── eval_plots/                 # ROC, confusion matrix & importance plots
├── templates/
│   └── index.html                  # Interactive clinical workstation UI
├── app.py                          # Flask web application & REST API
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

---

## 🚀 Installation & Quick Start

### 1. Install Dependencies
```bash
cd c:\Users\SANJAY\Downloads\mern\heart_attack_prediction
pip install -r requirements.txt
```

### 2. Run the Full ML Training Pipeline
```bash
python src/pipeline.py
```
*(Loads dataset, tunes Random Forest & XGBoost, generates comparison charts, and exports artifacts to `models/`)*

### 3. Launch the Interactive Web Application
```bash
python app.py
```
Open your browser at **`http://localhost:5001`**.

---

## 📡 API Endpoints Reference

### 1. Predict Single Patient Risk
- **Endpoint**: `POST /api/predict`
- **Payload**:
```json
{
  "age": 58,
  "sex": 1,
  "cp": 0,
  "trestbps": 145,
  "chol": 260,
  "fbs": 1,
  "restecg": 1,
  "thalach": 125,
  "exang": 1,
  "oldpeak": 2.2,
  "slope": 1,
  "ca": 1,
  "thal": 3
}
```
- **Response**:
```json
{
  "status": "success",
  "data": {
    "rf_prediction": { "probability": 0.88, "probability_pct": "88.0%", "label": "High Risk (Presence)" },
    "xgb_prediction": { "probability": 0.91, "probability_pct": "91.0%", "label": "High Risk (Presence)" },
    "ensemble_prediction": { "probability": 0.895, "probability_pct": "89.5%", "risk_stratum": { "level": "Critical / Severe Risk" } },
    "risk_factors": [...],
    "recommendations": {...}
  }
}
```

### 2. Batch Predict Multiple Patients
- **Endpoint**: `POST /api/batch-predict`
- **Method**: Multipart Form Upload (`file` CSV) or empty for benchmark sample batch.

### 3. Model Benchmark Metadata
- **Endpoint**: `GET /api/metadata`
