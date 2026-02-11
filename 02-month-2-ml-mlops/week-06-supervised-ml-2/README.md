# Week 6: Supervised Learning - ML Pipelines

## Overview
This week focuses on production-ready machine learning with scikit-learn Pipelines. You'll learn to build reproducible ML workflows, perform hyperparameter tuning, and extract meaningful feature importance.

## Learning Objectives
By the end of this week, you will:
- Build complete ML pipelines with preprocessing
- Use ColumnTransformer for different feature types
- Perform hyperparameter tuning with GridSearchCV
- Analyze permutation importance for model interpretability
- Serialize models with joblib for deployment

## Files
```
week-06-supervised-ml-2/
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── week06_pipeline.ipynb       # Interactive pipeline demo
├── scripts/
│   └── week06_ml_pipeline.py       # Production-ready script
├── outputs/
│   ├── best_pipeline.pkl           # Serialized model
│   ├── grid_search_results.csv     # Tuning results
│   └── permutation_importance.csv  # Feature importance
└── README.md
```

## Setup

### Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Pipeline                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │              ColumnTransformer                       │   │
│  │  ┌───────────────┐      ┌──────────────────┐       │   │
│  │  │ Numeric Pipe  │      │  Categorical Pipe│       │   │
│  │  │ - SimpleImputer│      │  - SimpleImputer │       │   │
│  │  │ - StandardScaler│     │  - OneHotEncoder │       │   │
│  │  └───────────────┘      └──────────────────┘       │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RandomForestClassifier                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. ColumnTransformer
Handles different preprocessing for numeric and categorical features:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
```

### 2. Pipeline
Ensures consistent preprocessing during training and inference:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
```

### 3. GridSearchCV
Systematic hyperparameter tuning with cross-validation:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    pipeline, param_grid, 
    cv=5, scoring='f1', 
    n_jobs=-1, verbose=2
)
```

## Hyperparameter Tuning Results

Example output from GridSearchCV:
```
Best Parameters:
- classifier__max_depth: 20
- classifier__min_samples_split: 2
- classifier__n_estimators: 200

Best CV Score: 0.67 (F1)
```

## Feature Importance

### Permutation Importance
Measures feature importance by shuffling values:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    best_model, X_test, y_test, 
    n_repeats=10, random_state=42
)
```

Top features typically include:
1. Contract type
2. Tenure
3. MonthlyCharges
4. TotalCharges
5. InternetService

## Running the Script

### Option 1: Jupyter Notebook (Interactive)
```bash
jupyter notebook notebooks/week06_pipeline.ipynb
```

### Option 2: Python Script (Production)
```bash
python scripts/week06_ml_pipeline.py
```

## Outputs

After running, you'll find in `outputs/`:

1. **best_pipeline.pkl** - Serialized pipeline ready for deployment
2. **grid_search_results.csv** - All hyperparameter combinations tested
3. **permutation_importance.csv** - Feature importance rankings
4. **feature_importance_plot.png** - Visualization

## Model Serialization

Save the trained pipeline:
```python
import joblib

joblib.dump(grid_search.best_estimator_, 'outputs/best_pipeline.pkl')
```

Load for inference:
```python
model = joblib.load('outputs/best_pipeline.pkl')
predictions = model.predict(new_data)
```

## Best Practices

1. **Always use Pipelines** - Prevents data leakage between train/test
2. **Set `handle_unknown='ignore'`** - For production robustness
3. **Use cross-validation** - Better performance estimates
4. **Save feature names** - For validation during inference
5. **Version your models** - Track what was trained when

## Exercises

1. **Add more models** - Try GradientBoosting or XGBoost in the pipeline
2. **RandomizedSearchCV** - Compare with GridSearchCV for efficiency
3. **Custom transformers** - Create a transformer for feature engineering
4. **Pipeline visualization** - Use `sklearn.set_config(display='diagram')`

## Common Issues

### Issue: "Unknown categories during transform"
**Solution**: Use `handle_unknown='ignore'` in OneHotEncoder

### Issue: Memory error with GridSearchCV
**Solution**: Use RandomizedSearchCV or reduce parameter grid

### Issue: Feature names lost after ColumnTransformer
**Solution**: Use `get_feature_names_out()` in sklearn 1.0+

## Next Week Preview
Week 7 covers deploying models with FastAPI - taking your serialized model and serving predictions via a REST API.

---

## Assignment
Build and tune a complete pipeline for the House Prices dataset:
1. Handle missing values appropriately
2. Engineer at least 2 new features
3. Tune hyperparameters with GridSearchCV
4. Document your best model's performance
5. Save the pipeline for deployment
