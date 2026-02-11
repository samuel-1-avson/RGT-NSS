"""
Week 6: Supervised Learning 2 - ML Pipelines & Hyperparameter Tuning
======================================================================

This script implements a complete ML pipeline with preprocessing,
hyperparameter tuning, and feature importance analysis.

Pipeline Components:
    1. Preprocessing Pipeline (ColumnTransformer)
       - Numeric features: StandardScaler (z-score normalization)
       - Categorical features: OneHotEncoder (drop first to avoid collinearity)
    
    2. Model Pipeline (sklearn Pipeline)
       - Combines preprocessing and classifier
       - Ensures consistent transformation on train/test
    
    3. Hyperparameter Tuning (GridSearchCV)
       - Systematic search over parameter grid
       - 5-fold cross-validation
       - Optimizes for F1 score (best for imbalanced data)
    
    4. Feature Importance Analysis
       - Permutation importance (model-agnostic)
       - Tree-based feature importance (for Random Forest)

Model: Random Forest (selected from Week 5)
Dataset: Telco Customer Churn

Author: RGT-NSS Training Program
Week: 6 - Supervised Learning 2 (Pipelines, Tuning, Interpretability)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Configure plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                         '01-month-1-data-analytics', 'week-01-tools-setup',
                         'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_data():
    """
    Load and prepare the Telco Customer Churn dataset.
    
    Returns:
        tuple: (df, numeric_features, categorical_features)
            - df: Full dataframe with target
            - numeric_features: List of numeric column names
            - categorical_features: List of categorical column names
    """
    print("=" * 70)
    print("Week 6: ML Pipelines & Hyperparameter Tuning")
    print("=" * 70)
    
    print("\n[Step 1] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Define feature groups
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    # Create target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    print(f"    Records: {len(df):,}")
    print(f"    Numeric features: {len(numeric_features)}")
    print(f"    Categorical features: {len(categorical_features)}")
    
    return df, numeric_features, categorical_features


def create_preprocessor(numeric_features, categorical_features):
    """
    Create a preprocessing pipeline using ColumnTransformer.
    
    This applies different transformations to different column types:
        - Numeric: StandardScaler (mean=0, std=1)
        - Categorical: OneHotEncoder (drop first level)
    
    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
    
    Returns:
        ColumnTransformer: Preprocessor object
    """
    print("\n[Step 2] Creating preprocessing pipeline...")
    
    # Numeric preprocessing: Standardize features
    # Formula: z = (x - mean) / std
    numeric_transformer = StandardScaler()
    
    # Categorical preprocessing: One-hot encode
    # drop='first' avoids multicollinearity (dummy variable trap)
    categorical_transformer = OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'  # Ignore unknown categories in test set
    )
    
    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    print("    [OK] Preprocessor created")
    print("        Numeric: StandardScaler")
    print("        Categorical: OneHotEncoder(drop='first')")
    
    return preprocessor


def build_pipeline(preprocessor):
    """
    Build complete ML pipeline with preprocessing and classifier.
    
    The pipeline ensures:
        - Same transformations applied to train and test
        - No data leakage (transformers fit only on training data)
        - Easy deployment (single object to save/load)
    
    Args:
        preprocessor: ColumnTransformer object
    
    Returns:
        Pipeline: Complete pipeline object
    """
    print("\n[Step 3] Building ML pipeline...")
    
    # Random Forest Classifier
    # Selected from Week 5 as baseline model
    classifier = RandomForestClassifier(
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Create pipeline: preprocessing -> classification
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    print("    [OK] Pipeline created")
    print("        Steps: preprocessor -> RandomForest")
    
    return pipeline


def hyperparameter_tuning(pipeline, X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Grid Search:
        - Exhaustively tries all parameter combinations
        - Uses 5-fold cross-validation for robust evaluation
        - Optimizes for F1 score (best for imbalanced classification)
    
    Parameters Tuned:
        - n_estimators: Number of trees (100, 200)
        - max_depth: Maximum tree depth (10, 20, None)
        - min_samples_split: Min samples to split node (2, 5)
        - min_samples_leaf: Min samples at leaf (1, 2)
    
    Args:
        pipeline: ML pipeline object
        X_train: Training features
        y_train: Training labels
    
    Returns:
        GridSearchCV: Fitted grid search object
    """
    print("\n[Step 4] Hyperparameter tuning (GridSearchCV)...")
    
    # Define parameter grid
    # Format: step_name__parameter_name
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    print(f"    Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
    print(f"    Cross-validation: 5-fold")
    print(f"    Scoring: F1 (macro)")
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,              # 5-fold cross-validation
        scoring='f1',      # Optimize for F1 score
        n_jobs=-1,         # Use all CPU cores
        verbose=0
    )
    
    # Fit grid search
    print("    Training... (this may take a minute)")
    grid_search.fit(X_train, y_train)
    
    print(f"\n    [OK] Best F1 Score: {grid_search.best_score_:.4f}")
    print(f"    Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"        {param.replace('classifier__', '')}: {value}")
    
    return grid_search


def evaluate_tuned_model(grid_search, X_test, y_test):
    """
    Evaluate the tuned model on test set.
    
    Args:
        grid_search: Fitted GridSearchCV object
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n[Step 5] Evaluating tuned model...")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Classification report
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Cross-validation scores for more robust estimate
    cv_scores = cross_val_score(best_model, X_test, y_test, cv=5, scoring='f1')
    print(f"    Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return best_model


def analyze_feature_importance(best_model, X_train, numeric_features, categorical_features):
    """
    Analyze feature importance using multiple methods.
    
    Methods:
        1. Permutation Importance: Model-agnostic, shuffles each feature
        2. Tree Importance: Built-in Random Forest feature importance
    
    Args:
        best_model: Fitted pipeline
        X_train: Training features (for getting feature names)
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
    """
    print("\n[Step 6] Analyzing feature importance...")
    
    # Get feature names after preprocessing
    # Numeric features keep their names
    # Categorical features get expanded (e.g., 'Contract' -> 'Contract_One year')
    preprocessor = best_model.named_steps['preprocessor']
    classifier = best_model.named_steps['classifier']
    
    # Get categorical feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features)
    
    # Combine all feature names
    all_features = list(numeric_features) + list(cat_features_encoded)
    
    # Method 1: Tree-based feature importance
    importances = classifier.feature_importances_
    
    # Create DataFrame for easier handling
    feature_importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Display top 10 features
    print("\n    Top 10 Most Important Features:")
    print("    " + "-" * 50)
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importances (Random Forest)', fontweight='bold')
    plt.gca().invert_yaxis()  # Most important at top
    plt.tight_layout()
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_PATH, 'feature_importance.png'), dpi=300)
    plt.close()
    print(f"\n    [OK] Saved: feature_importance.png")
    
    return feature_importance_df


def save_model(best_model, grid_search):
    """
    Save the trained model and tuning results.
    
    Args:
        best_model: Best fitted pipeline
        grid_search: GridSearchCV object with results
    """
    print("\n[Step 7] Saving model and results...")
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Save the complete pipeline
    model_file = os.path.join(MODEL_PATH, 'tuned_random_forest.pkl')
    joblib.dump(best_model, model_file)
    print(f"    [OK] Model saved: tuned_random_forest.pkl")
    
    # Save GridSearchCV results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_file = os.path.join(OUTPUT_PATH, 'grid_search_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"    [OK] Grid search results saved")
    
    # Save best parameters as JSON
    import json
    params_file = os.path.join(OUTPUT_PATH, 'best_parameters.json')
    with open(params_file, 'w') as f:
        json.dump(grid_search.best_params_, f, indent=2)
    print(f"    [OK] Best parameters saved")


def main():
    """
    Main execution pipeline for Week 6.
    
    Steps:
        1. Load data
        2. Create preprocessor
        3. Build pipeline
        4. Hyperparameter tuning
        5. Evaluate model
        6. Feature importance
        7. Save model
    """
    # Step 1: Load data
    df, numeric_features, categorical_features = load_data()
    
    # Split data
    X = df.drop(['Churn', 'customerID'], axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 2: Create preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Step 3: Build pipeline
    pipeline = build_pipeline(preprocessor)
    
    # Step 4: Hyperparameter tuning
    grid_search = hyperparameter_tuning(pipeline, X_train, y_train)
    
    # Step 5: Evaluate
    best_model = evaluate_tuned_model(grid_search, X_test, y_test)
    
    # Step 6: Feature importance
    analyze_feature_importance(best_model, X_train, numeric_features, categorical_features)
    
    # Step 7: Save
    save_model(best_model, grid_search)
    
    print("\n" + "=" * 70)
    print("Week 6 Complete!")
    print("=" * 70)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("\nNext: Deploy this model as API (Week 7)")


if __name__ == "__main__":
    main()
