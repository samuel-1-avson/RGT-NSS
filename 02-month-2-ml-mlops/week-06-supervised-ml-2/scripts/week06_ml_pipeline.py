"""
Week 6: ML Pipeline Script

This script demonstrates building a production-ready ML pipeline with:
- Complete preprocessing using ColumnTransformer
- Hyperparameter tuning with GridSearchCV
- Permutation importance analysis
- Model serialization with joblib

Usage:
    python week06_ml_pipeline.py

Output:
    - best_pipeline.pkl: Serialized best model
    - grid_search_results.csv: Hyperparameter tuning results
    - permutation_importance.csv: Feature importance scores
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import json
from datetime import datetime
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OUTPUT_DIR = Path('../outputs')
DATA_DIR = Path('../data')


def load_data():
    """
    Load the Telco Customer Churn dataset.
    
    Returns:
        pd.DataFrame: Raw dataset
    """
    print("📊 Loading dataset...")
    
    # Try local file first, then URL
    local_path = DATA_DIR / 'Telco-Customer-Churn.csv'
    
    if local_path.exists():
        df = pd.read_csv(local_path)
    else:
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
        # Save for future use
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_path, index=False)
    
    print(f"   Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def preprocess_data(df):
    """
    Clean and prepare data for modeling.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("🔧 Preprocessing data...")
    
    # Create a copy
    df = df.copy()
    
    # Handle TotalCharges - convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    print(f"   Preprocessing complete: {len(df)} rows")
    return df


def get_feature_types(df):
    """
    Identify numeric and categorical features.
    
    Args:
        df: DataFrame
        
    Returns:
        tuple: (numeric_features, categorical_features)
    """
    # Exclude target
    feature_cols = [col for col in df.columns if col != 'Churn']
    
    numeric_features = df[feature_cols].select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()
    
    categorical_features = df[feature_cols].select_dtypes(
        include=['object', 'category']
    ).columns.tolist()
    
    print(f"   Numeric features: {len(numeric_features)}")
    print(f"   Categorical features: {len(categorical_features)}")
    
    return numeric_features, categorical_features


def create_preprocessor(numeric_features, categorical_features):
    """
    Create ColumnTransformer for preprocessing.
    
    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    print("⚙️  Creating preprocessor...")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
             categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def create_pipeline(preprocessor, classifier):
    """
    Create full ML pipeline with preprocessing and classifier.
    
    Args:
        preprocessor: ColumnTransformer
        classifier: Estimator
        
    Returns:
        Pipeline: Complete ML pipeline
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline


def get_param_grid(classifier_name):
    """
    Get hyperparameter grid for specified classifier.
    
    Args:
        classifier_name: Name of the classifier
        
    Returns:
        dict: Parameter grid for GridSearchCV
    """
    param_grids = {
        'RandomForest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        },
        'GradientBoosting': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__min_samples_split': [2, 5]
        },
        'LogisticRegression': {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__class_weight': [None, 'balanced']
        }
    }
    
    return param_grids.get(classifier_name, {})


def train_and_tune(X_train, y_train, preprocessor, classifier_name):
    """
    Train and tune hyperparameters using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: ColumnTransformer
        classifier_name: Name of classifier to use
        
    Returns:
        GridSearchCV: Fitted grid search object
    """
    print(f"\n🔍 Training and tuning {classifier_name}...")
    
    # Select classifier
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    }
    
    classifier = classifiers[classifier_name]
    pipeline = create_pipeline(preprocessor, classifier)
    param_grid = get_param_grid(classifier_name)
    
    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"   ✓ Best CV Score: {grid_search.best_score_:.4f}")
    print(f"   ✓ Best Parameters: {grid_search.best_params_}")
    
    return grid_search


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    return metrics


def calculate_permutation_importance(model, X_test, y_test, feature_names):
    """
    Calculate permutation importance for feature interpretability.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        
    Returns:
        pd.DataFrame: Feature importance scores
    """
    print("\n📊 Calculating permutation importance...")
    
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring='f1'
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df


def save_results(grid_search, metrics, importance_df, output_dir):
    """
    Save all results to files.
    
    Args:
        grid_search: Fitted GridSearchCV
        metrics: Evaluation metrics dict
        importance_df: Feature importance DataFrame
        output_dir: Directory to save files
    """
    print("\n💾 Saving results...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best pipeline
    pipeline_path = output_dir / 'best_pipeline.pkl'
    joblib.dump(grid_search.best_estimator_, pipeline_path)
    print(f"   ✓ Saved pipeline to {pipeline_path}")
    
    # Save grid search results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_path = output_dir / 'grid_search_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"   ✓ Saved grid search results to {results_path}")
    
    # Save feature importance
    importance_path = output_dir / 'permutation_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"   ✓ Saved feature importance to {importance_path}")
    
    # Save metrics
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Saved test metrics to {metrics_path}")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'best_params': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_),
        'test_metrics': metrics,
        'cv_folds': CV_FOLDS,
        'random_state': RANDOM_STATE
    }
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Saved metadata to {metadata_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("WEEK 6: ML PIPELINE WITH HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Get feature types
    numeric_features, categorical_features = get_feature_types(df)
    
    # Split data
    print("\n✂️  Splitting data...")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Create preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Train and tune models
    models_to_train = ['RandomForest', 'GradientBoosting']
    results = {}
    best_models = {}
    
    for model_name in models_to_train:
        grid_search = train_and_tune(X_train, y_train, preprocessor, model_name)
        
        # Evaluate on test set
        metrics = evaluate_model(grid_search.best_estimator_, X_test, y_test)
        results[model_name] = metrics
        best_models[model_name] = grid_search
        
        print(f"\n   Test Set Performance:")
        for metric, value in metrics.items():
            print(f"      {metric}: {value:.4f}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_grid_search = best_models[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   F1 Score: {best_metrics['f1']:.4f}")
    
    # Calculate permutation importance
    # Get feature names after preprocessing
    preprocessor_fitted = best_grid_search.best_estimator_.named_steps['preprocessor']
    
    # Try to get feature names (works in sklearn 1.0+)
    try:
        feature_names = preprocessor_fitted.get_feature_names_out()
    except:
        # Fallback for older sklearn versions
        feature_names = [f'feature_{i}' for i in range(
            preprocessor_fitted.transform(X_test).shape[1]
        )]
    
    importance_df = calculate_permutation_importance(
        best_grid_search.best_estimator_, X_test, y_test, feature_names
    )
    
    print("\n📈 Top 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save results
    save_results(best_grid_search, best_metrics, importance_df, OUTPUT_DIR)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nBest Model: {best_model_name}")
    print(f"Best CV F1 Score: {best_grid_search.best_score_:.4f}")
    print(f"\nTest Set Metrics:")
    for metric, value in best_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\n✅ Pipeline complete!")
    
    return best_grid_search.best_estimator_


if __name__ == "__main__":
    best_model = main()
