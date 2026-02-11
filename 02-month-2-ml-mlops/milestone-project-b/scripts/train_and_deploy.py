"""
Milestone B: ML Microservice - Training Pipeline
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                         '01-month-1-data-analytics', 'week-01-tools-setup',
                         'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')


def train_model():
    """Complete training pipeline."""
    print("=" * 60)
    print("Milestone B: Training Pipeline")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                   'PhoneService', 'MultipleLines', 'InternetService',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies',
                   'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    X = df.drop(['Churn', 'customerID'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessing
    print("[2/4] Creating pipeline...")
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical)
    ])
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    # Training
    print("[3/4] Training with GridSearchCV...")
    param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [10, 20]}
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"    Best F1: {grid.best_score_:.4f}")
    
    # Evaluation
    print("[4/4] Evaluating...")
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"    Test F1: {f1:.4f}")
    
    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, 'churn_model.pkl'))
    print("\n[OK] Model saved to models/churn_model.pkl")
    
    return model


if __name__ == "__main__":
    train_model()
