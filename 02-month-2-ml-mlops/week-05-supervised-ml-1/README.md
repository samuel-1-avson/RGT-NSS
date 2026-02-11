# Week 5: Supervised Learning - Baseline Models

## Overview
This week introduces fundamental supervised learning concepts through hands-on implementation of baseline classification models. We use the Telco Customer Churn dataset to predict customer churn and compare different algorithms.

## Learning Objectives
By the end of this week, you will:
- Understand the supervised learning workflow
- Prepare data for machine learning (handle missing values, encode categoricals)
- Implement train/test splits with stratification
- Train and evaluate baseline models
- Compare models using multiple metrics
- Visualize model performance with ROC curves

## Dataset
**Telco Customer Churn Dataset**
- 7,043 customers
- 20 features + 1 target (Churn)
- Binary classification problem
- Features include demographics, services, and account information

## Files
```
week-05-supervised-ml-1/
├── data/
│   └── Telco-Customer-Churn.csv    # Dataset
├── notebooks/
│   └── week05_baseline_models.ipynb  # Main notebook
├── outputs/
│   ├── baseline_results.csv        # Model comparison
│   └── roc_curves.png              # ROC visualization
├── docs/
│   └── model_selection_rationale.md  # Model selection guide
└── README.md
```

## Setup

### Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Download Dataset
```python
import pandas as pd
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df.to_csv('data/Telco-Customer-Churn.csv', index=False)
```

## Models Covered

### 1. Logistic Regression
- **Type**: Linear classifier
- **Strengths**: Fast training, interpretable coefficients, probabilistic output
- **Best for**: Linearly separable problems, baseline comparison

### 2. Random Forest Classifier
- **Type**: Ensemble (bagging) method
- **Strengths**: Handles non-linear relationships, feature importance, robust to overfitting
- **Best for**: Complex datasets with mixed feature types

## Evaluation Metrics

| Metric | Description | When to Use |
|--------|-------------|-------------|
| Accuracy | Overall correct predictions | Balanced classes |
| Precision | True positives / Predicted positives | Minimize false positives |
| Recall | True positives / Actual positives | Minimize false negatives |
| F1 Score | Harmonic mean of precision and recall | Balance precision/recall |
| ROC-AUC | Area under ROC curve | Compare models across thresholds |

## Running the Notebook

```bash
jupyter notebook notebooks/week05_baseline_models.ipynb
```

## Expected Outputs

1. **Model Comparison Table**
   ```
   | Model              | Accuracy | Precision | Recall | F1    | ROC-AUC |
   |--------------------|----------|-----------|--------|-------|---------|
   | Logistic Regression| 0.80     | 0.65      | 0.55   | 0.60  | 0.82    |
   | Random Forest      | 0.79     | 0.63      | 0.50   | 0.56  | 0.81    |
   ```

2. **ROC Curve Plot**: Visual comparison of model discrimination ability

## Key Concepts

### Stratified Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
Ensures train and test sets have same class distribution.

### Data Preprocessing
- Handle missing values (especially `TotalCharges`)
- Encode categorical variables
- Scale features (important for Logistic Regression)

## Exercises

1. **Experiment with different random seeds** - How stable are the results?
2. **Try different test sizes** - 10%, 30% - How does this affect performance estimates?
3. **Add a third model** - Try XGBoost or SVM and compare
4. **Feature importance** - Which features are most predictive in Random Forest?

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Understanding Confusion Matrix](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)
- [ROC Curves and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

## Next Week Preview
Week 6 will cover ML Pipelines, hyperparameter tuning with GridSearchCV, and feature importance analysis.

---

## Assignment
Submit your completed notebook with:
1. All cells executed showing results
2. Answers to the exercise questions
3. Your interpretation of which model performed better and why
