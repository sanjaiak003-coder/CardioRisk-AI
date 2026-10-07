"""
Heart Attack Risk Prediction - Clinical Inference & Patient Stratification Engine
Provides multi-model risk scores, key driver explanations, and AHA/ACC aligned clinical guidance.
"""

import os
import joblib
import pandas as pd
import numpy as np
from src.data_loader import FEATURE_NAMES, CLINICAL_FEATURE_DESCRIPTIONS
from src.preprocess import ClinicalFeatureEngineer


class HeartAttackPredictor:
    """
    Production inference engine supporting Random Forest, XGBoost, and Weighted Ensemble.
    """
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'models')

        self.models_dir = models_dir
        self.rf_model = None
        self.xgb_model = None
        self.preprocessor = None
        self.metadata = {}
        self.load_artifacts()

    def load_artifacts(self):
        """Loads serialized models, scalers, and metadata."""
        rf_path = os.path.join(self.models_dir, 'random_forest_model.joblib')
        xgb_path = os.path.join(self.models_dir, 'xgboost_model.joblib')
        preproc_path = os.path.join(self.models_dir, 'preprocessor.joblib')
        meta_path = os.path.join(self.models_dir, 'model_metadata.json')

        if os.path.exists(rf_path):
            self.rf_model = joblib.load(rf_path)
        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
        if os.path.exists(preproc_path):
            self.preprocessor = joblib.load(preproc_path)
        if os.path.exists(meta_path):
            import json
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)

    def _determine_risk_level(self, probability: float) -> dict:
        """Categorizes probability into clinical risk strata with color codes."""
        if probability < 0.25:
            return {
                'level': 'Low Risk',
                'badge': 'bg-emerald-100 text-emerald-800 border-emerald-300',
                'color': '#10B981',
                'urgency': 'Routine Wellness & Prevention',
                'summary': 'Cardiovascular parameters are generally within normal limits. Maintain healthy lifestyle.'
            }
        elif probability < 0.50:
            return {
                'level': 'Moderate / Borderline Risk',
                'badge': 'bg-amber-100 text-amber-800 border-amber-300',
                'color': '#F59E0B',
                'urgency': 'Scheduled Clinical Review',
                'summary': 'Borderline clinical indicators present. Early risk-factor reduction recommended.'
            }
        elif probability < 0.75:
            return {
                'level': 'High Risk',
                'badge': 'bg-orange-100 text-orange-800 border-orange-300',
                'color': '#F97316',
                'urgency': 'Prompt Cardiology Workup Recommended',
                'summary': 'Elevated probability of significant coronary artery disease or ischemia. Comprehensive diagnostic evaluation advised.'
            }
        else:
            return {
                'level': 'Critical / Severe Risk',
                'badge': 'bg-rose-100 text-rose-800 border-rose-300',
                'color': '#EF4444',
                'urgency': 'Immediate Urgent Evaluation Required',
                'summary': 'High multi-factor ischemic and vascular burden detected. Requires immediate clinical attention.'
            }

    def _analyze_risk_factors(self, data: dict) -> list:
        """
        Analyzes patient attributes against clinical cardiology thresholds (AHA/ACC).
        """
        drivers = []

        # Chest pain
        cp = data.get('cp', 0)
        if cp == 0:
            drivers.append({
                'factor': 'Typical Anginal Chest Pain',
                'impact': 'High Risk (+)',
                'severity': 'high',
                'detail': 'Substernal chest pressure triggered by physical or emotional exertion.'
            })
        elif cp == 1:
            drivers.append({
                'factor': 'Atypical Angina',
                'impact': 'Moderate Risk (+)',
                'severity': 'medium',
                'detail': 'Non-classical chest discomfort with atypical radiation.'
            })

        # Blood pressure
        bps = data.get('trestbps', 120)
        if bps >= 140:
            drivers.append({
                'factor': f'Stage 2 Hypertension ({bps} mm Hg)',
                'impact': 'High Risk (+)',
                'severity': 'high',
                'detail': 'Resting systolic blood pressure substantially elevates vascular strain.'
            })
        elif bps >= 130:
            drivers.append({
                'factor': f'Stage 1 Hypertension ({bps} mm Hg)',
                'impact': 'Moderate Risk (+)',
                'severity': 'medium',
                'detail': 'Elevated resting systolic pressure.'
            })

        # Cholesterol
        chol = data.get('chol', 200)
        if chol >= 240:
            drivers.append({
                'factor': f'Hypercholesterolemia ({chol} mg/dL)',
                'impact': 'High Risk (+)',
                'severity': 'high',
                'detail': 'Serum cholesterol above 240 mg/dL accelerates atherosclerotic plaque formation.'
            })
        elif chol >= 200:
            drivers.append({
                'factor': f'Borderline High Cholesterol ({chol} mg/dL)',
                'impact': 'Moderate Risk (+)',
                'severity': 'medium',
                'detail': 'Serum cholesterol in borderline territory (200-239 mg/dL).'
            })

        # ST depression
        oldpeak = data.get('oldpeak', 0.0)
        if oldpeak >= 1.5:
            drivers.append({
                'factor': f'Significant ST Depression ({oldpeak} mm)',
                'impact': 'Critical Risk (+)',
                'severity': 'critical',
                'detail': 'Exercise-induced ST segment depression indicates severe subendocardial ischemia.'
            })
        elif oldpeak > 0.5:
            drivers.append({
                'factor': f'Mild ST Depression ({oldpeak} mm)',
                'impact': 'Moderate Risk (+)',
                'severity': 'medium',
                'detail': 'Mild exercise-induced ST segment shift.'
            })

        # Exercise Angina
        if data.get('exang', 0) == 1:
            drivers.append({
                'factor': 'Exercise-Induced Angina',
                'impact': 'High Risk (+)',
                'severity': 'high',
                'detail': 'Coronary perfusion deficit during increased cardiac workload.'
            })

        # Major Vessels Colored
        ca = data.get('ca', 0)
        if ca >= 2:
            drivers.append({
                'factor': f'Multi-Vessel Fluoroscopy Disease ({ca} major vessels)',
                'impact': 'Critical Risk (+)',
                'severity': 'critical',
                'detail': 'Fluoroscopic opacification indicates extensive multi-vessel coronary obstruction.'
            })
        elif ca == 1:
            drivers.append({
                'factor': 'Single Major Vessel Fluoroscopy Opacity',
                'impact': 'Moderate Risk (+)',
                'severity': 'medium',
                'detail': 'Coronary artery disease detected in fluoroscopic imaging.'
            })

        # Thallium defect
        thal = data.get('thal', 1)
        if thal == 3:
            drivers.append({
                'factor': 'Reversible Thallium Perfusion Defect',
                'impact': 'High Risk (+)',
                'severity': 'high',
                'detail': 'Nuclear stress imaging indicates inducible myocardial ischemia.'
            })
        elif thal == 2:
            drivers.append({
                'factor': 'Fixed Thallium Perfusion Defect',
                'impact': 'High Risk (+)',
                'severity': 'high',
                'detail': 'Possible prior silent myocardial infarction or non-viable scar tissue.'
            })

        # Fasting sugar
        if data.get('fbs', 0) == 1:
            drivers.append({
                'factor': 'Elevated Fasting Blood Glucose (>120 mg/dL)',
                'impact': 'Moderate Risk (+)',
                'severity': 'medium',
                'detail': 'Diabetic/pre-diabetic metabolic state doubles cardiovascular risk profile.'
            })

        # Protective / Favorable factors
        if thalach := data.get('thalach', 140):
            age = data.get('age', 50)
            if thalach >= (220 - age) * 0.85:
                drivers.append({
                    'factor': f'Robust Stress Peak Heart Rate ({thalach} bpm)',
                    'impact': 'Protective Factor (-)',
                    'severity': 'protective',
                    'detail': 'Strong chronotropic response during physical workload indicates good cardiac reserve.'
                })

        return drivers

    def _generate_clinical_recommendations(self, probability: float, data: dict) -> dict:
        """
        Generates clinical management guidance aligned with ACC/AHA cardiology standards.
        """
        diagnostic_tests = []
        lifestyle_interventions = []
        medical_followup = []

        if probability >= 0.50:
            diagnostic_tests.extend([
                "12-Lead Electrocardiogram (ECG) with rhythm strip monitoring",
                "Transthoracic Echocardiogram (TTE) to assess left ventricular ejection fraction (LVEF) & wall motion",
                "Diagnostic Coronary Angiography or Coronary CT Angiography (CCTA)",
                "High-Sensitivity Cardiac Troponin I/T and Serum Lipid Panel"
            ])
            medical_followup.extend([
                "Urgent consultation with a Board-Certified Cardiologist",
                "Evaluation for antiplatelet therapy (e.g. Aspirin 81mg) and high-intensity statin therapy",
                "Review of antihypertensive regimen (ACE inhibitors / ARBs / Beta-blockers as indicated)"
            ])
        else:
            diagnostic_tests.extend([
                "Baseline 12-lead Resting ECG",
                "Comprehensive Lipid Panel (Total Chol, LDL, HDL, Triglycerides)",
                "Fasting Plasma Glucose & HbA1c screening"
            ])
            medical_followup.extend([
                "Routine annual primary care / cardiovascular wellness check",
                "Periodic non-invasive blood pressure monitoring at home"
            ])

        # Lifestyle & dietary interventions
        lifestyle_interventions.extend([
            "Mediterranean or DASH cardiovascular dietary pattern (low sodium < 2000mg/day, low saturated fats)",
            "Aerobic exercise: 150 min/week moderate intensity (subject to clinical clearance)",
            "Strict avoidance of tobacco products & active smoking cessation support if applicable",
            "Target BMI maintenance between 18.5 - 24.9 kg/m²"
        ])

        return {
            'diagnostic_tests': diagnostic_tests,
            'lifestyle_interventions': lifestyle_interventions,
            'medical_followup': medical_followup
        }

    def predict_patient(self, patient_dict: dict) -> dict:
        """
        Runs dual-model inference on a patient profile.
        """
        if self.rf_model is None or self.xgb_model is None or self.preprocessor is None:
            raise RuntimeError("Models or preprocessor not loaded. Train models first via src/pipeline.py.")

        # Ensure correct order of features
        df_input = pd.DataFrame([{col: float(patient_dict.get(col, 0)) for col in FEATURE_NAMES}])
        
        # Preprocess features
        X_scaled = self.preprocessor.transform(df_input)

        # Random Forest Inference
        rf_prob = float(self.rf_model.predict_proba(X_scaled)[0, 1])
        rf_pred = int(self.rf_model.predict(X_scaled)[0])

        # XGBoost Inference
        xgb_prob = float(self.xgb_model.predict_proba(X_scaled)[0, 1])
        xgb_pred = int(self.xgb_model.predict(X_scaled)[0])

        # Weighted Ensemble (50% RF, 50% XGBoost)
        ensemble_prob = float(0.50 * rf_prob + 0.50 * xgb_prob)
        ensemble_pred = int(ensemble_prob >= 0.50)

        risk_meta = self._determine_risk_level(ensemble_prob)
        risk_factors = self._analyze_risk_factors(patient_dict)
        recommendations = self._generate_clinical_recommendations(ensemble_prob, patient_dict)

        return {
            'patient_data': patient_dict,
            'rf_prediction': {
                'label': 'High Risk (Presence)' if rf_pred == 1 else 'Low Risk (Absence)',
                'probability': rf_prob,
                'probability_pct': f"{rf_prob * 100:.1f}%",
                'risk_stratum': self._determine_risk_level(rf_prob)
            },
            'xgb_prediction': {
                'label': 'High Risk (Presence)' if xgb_pred == 1 else 'Low Risk (Absence)',
                'probability': xgb_prob,
                'probability_pct': f"{xgb_prob * 100:.1f}%",
                'risk_stratum': self._determine_risk_level(xgb_prob)
            },
            'ensemble_prediction': {
                'label': 'High Risk (Presence)' if ensemble_pred == 1 else 'Low Risk (Absence)',
                'probability': ensemble_prob,
                'probability_pct': f"{ensemble_prob * 100:.1f}%",
                'risk_stratum': risk_meta
            },
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }

    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Runs batch predictions on a pandas DataFrame of clinical patient records.
        """
        X_df = df_input[FEATURE_NAMES]
        X_scaled = self.preprocessor.transform(X_df)

        rf_probs = self.rf_model.predict_proba(X_scaled)[:, 1]
        xgb_probs = self.xgb_model.predict_proba(X_scaled)[:, 1]
        ensemble_probs = 0.50 * rf_probs + 0.50 * xgb_probs

        results_df = df_input.copy()
        results_df['RF_Risk_Probability'] = np.round(rf_probs * 100, 1)
        results_df['XGB_Risk_Probability'] = np.round(xgb_probs * 100, 1)
        results_df['Ensemble_Risk_Probability'] = np.round(ensemble_probs * 100, 1)
        results_df['Predicted_Risk_Category'] = [
            self._determine_risk_level(p)['level'] for p in ensemble_probs
        ]
        results_df['High_Risk_Flag'] = (ensemble_probs >= 0.50).astype(int)

        return results_df
