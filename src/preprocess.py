"""
Heart Attack Risk Prediction - Data Preprocessing & Feature Engineering
Builds robust clinical feature transformations and scikit-learn preprocessing pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw clinical metrics into derived cardiovascular risk indicators:
    1. HR Reserve Deficit: (220 - age) - thalach (Myocardial stress deficit)
    2. Chol-to-Age Ratio: chol / age (Atherosclerotic burden per lifetime year)
    3. Rate Pressure Product (RPP): (trestbps * thalach) / 1000 (Cardiac workload index)
    4. ST Ischemia Severity: oldpeak * (slope + 1) (Exercise-induced ischemic burden)
    5. Major Clinical Flag Count: Count of active risk thresholds (HTN, Dyslipidemia, Angina, etc.)
    """
    def __init__(self, include_engineered: bool = True):
        self.include_engineered = include_engineered
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=[
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ])
        else:
            df = X.copy()

        if not self.include_engineered:
            self.feature_names_out_ = list(df.columns)
            return df

        # 1. Heart rate reserve deficit (Stress test deficit)
        max_pred_hr = 220.0 - df['age']
        df['hr_reserve_deficit'] = np.maximum(0, max_pred_hr - df['thalach'])

        # 2. Cholesterol to Age ratio (Atherosclerosis velocity)
        df['chol_age_ratio'] = df['chol'] / np.maximum(df['age'], 1.0)

        # 3. Rate Pressure Product / 1000 (Myocardial oxygen consumption)
        df['rate_pressure_product'] = (df['trestbps'] * df['thalach']) / 1000.0

        # 4. ST Ischemic Severity Index
        df['st_ischemia_index'] = df['oldpeak'] * (df['slope'] + 1.0)

        # 5. Composite Clinical Risk Count
        risk_flags = (
            (df['age'] >= 55).astype(int) +
            (df['sex'] == 1).astype(int) +
            (df['trestbps'] >= 140).astype(int) +
            (df['chol'] >= 240).astype(int) +
            (df['fbs'] == 1).astype(int) +
            (df['exang'] == 1).astype(int) +
            (df['ca'] >= 1).astype(int) +
            (df['oldpeak'] >= 1.5).astype(int)
        )
        df['clinical_risk_count'] = risk_flags

        self.feature_names_out_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


def create_preprocessing_pipeline(include_engineered: bool = True) -> Pipeline:
    """
    Creates full preprocessing pipeline including clinical feature engineering and standard scaling.
    """
    pipeline = Pipeline([
        ('feature_engineer', ClinicalFeatureEngineer(include_engineered=include_engineered)),
        ('scaler', StandardScaler())
    ])
    return pipeline


def save_preprocessor(preprocessor, filepath: str):
    """Saves fitted preprocessor to disk."""
    joblib.dump(preprocessor, filepath)
    print(f"Preprocessor saved to: {filepath}")


def load_preprocessor(filepath: str):
    """Loads fitted preprocessor from disk."""
    return joblib.load(filepath)
