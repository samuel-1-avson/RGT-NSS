# Model Selection Rationale

## Overview
This document explains the rationale behind selecting baseline models for the Telco Customer Churn prediction task.

## Problem Characteristics

### Dataset Properties
- **Type**: Binary classification (Churn: Yes/No)
- **Size**: 7,043 samples
- **Features**: 20 (mix of numeric and categorical)
- **Class Distribution**: Imbalanced (~27% churn rate)
- **Feature Types**:
  - Numeric: tenure, MonthlyCharges, TotalCharges
  - Categorical: Contract, PaymentMethod, InternetService, etc.

### Key Challenges
1. **Mixed feature types**: Requires different preprocessing
2. **Class imbalance**: Standard accuracy may be misleading
3. **Interpretability needs**: Business stakeholders need to understand predictions
4. **Non-linear relationships**: Some features interact non-linearly

## Model Selection Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Performance | 35% | ROC-AUC, F1-score on validation set |
| Interpretability | 25% | Feature importance, coefficient clarity |
| Training Speed | 20% | Time to train and tune |
| Scalability | 10% | Inference speed, memory requirements |
| Robustness | 10% | Performance stability across folds |

## Selected Models

### 1. Logistic Regression

**Why Selected:**
- **Baseline benchmark**: Simple, interpretable linear model
- **Fast training**: Quick to train and tune
- **Probabilistic output**: Natural probability estimates
- **Regularization**: Built-in L1/L2 to prevent overfitting

**Strengths:**
- Coefficients indicate feature direction and magnitude
- Well-suited for linearly separable problems
- Efficient inference
- Extensive theoretical foundation

**Limitations:**
- Assumes linear relationship between features and log-odds
- May underfit complex interactions
- Requires feature scaling
- Sensitive to outliers

**When to Choose:**
- When interpretability is critical
- As a baseline for comparison
- When you suspect linear relationships
- For probability calibration

### 2. Random Forest Classifier

**Why Selected:**
- **Handles non-linearity**: Captures complex feature interactions
- **Feature importance**: Built-in feature ranking
- **Robust to overfitting**: Ensemble averaging reduces variance
- **No preprocessing needed**: Handles mixed types natively

**Strengths:**
- Excellent performance on tabular data
- Handles missing values well
- Provides feature importance scores
- Parallelizable training
- No need for feature scaling

**Limitations:**
- Can overfit with noisy data
- Less interpretable than linear models
- Slower inference than logistic regression
- Memory intensive for large ensembles
- Biased toward high-cardinality features

**When to Choose:**
- When you have mixed feature types
- When feature interactions are suspected
- When you need feature importance
- As a strong baseline before trying boosting

## Model Comparison Strategy

### Evaluation Protocol
1. **Stratified K-Fold Cross-Validation**: 5 folds with stratification
2. **Metrics**:
   - ROC-AUC: Primary metric (threshold-independent)
   - F1-Score: Balance precision and recall
   - Precision/Recall: For business understanding
3. **Statistical Testing**: McNemar's test for significance

### Decision Framework
```
┌─────────────────────────────────────────────────────────┐
│                    Model Selection                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ROC-AUC Diff > 0.02 and                               │
│   Statistically Significant?                            │
│         │                                               │
│    ┌────┴────┐                                          │
│    ▼         ▼                                          │
│   Yes       No                                          │
│    │         │                                          │
│    ▼         ▼                                          │
│ Select     Consider                                      │
│ Better     Other Factors                                 │
│ Model      (speed, interpretability)                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Expected Results

Based on similar churn prediction problems:

| Model | Expected ROC-AUC | Expected F1 | Training Time |
|-------|------------------|-------------|---------------|
| Logistic Regression | 0.80-0.84 | 0.58-0.62 | < 1 second |
| Random Forest | 0.81-0.85 | 0.60-0.64 | 2-5 seconds |

## Recommendation

### Primary Recommendation
Start with **Random Forest** as the main model because:
1. Typically achieves better performance on this dataset
2. Handles categorical features well
3. Provides useful feature importance for business insights

### Baseline Recommendation
Use **Logistic Regression** as the baseline because:
1. Simpler and more interpretable
2. Faster training for experimentation
3. Calibrated probabilities useful for ranking

### Next Steps
After establishing baselines:
1. Try Gradient Boosting (XGBoost/LightGBM)
2. Experiment with feature engineering
3. Address class imbalance (SMOTE, class weights)
4. Build ensemble of best models

## Feature Importance Interpretation

### Logistic Regression
- **Positive coefficients**: Increase churn probability
- **Negative coefficients**: Decrease churn probability
- **Magnitude**: Strength of effect (after scaling)

### Random Forest
- **Gini importance**: Average decrease in impurity
- **Permutation importance**: Drop in performance when feature shuffled

## Business Considerations

### Cost of Errors
- **False Negative (missed churn)**: Lost customer (~$500-1000 value)
- **False Positive (unnecessary retention)**: Retention cost (~$50-100)

### Threshold Selection
- Default 0.5 may not be optimal
- Use precision-recall curve to select threshold
- Consider business cost ratio

## References

1. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
2. Hosmer, D. W., & Lemeshow, S. (2004). Applied Logistic Regression.
3. He, H., & Ma, Y. (2013). Imbalanced Learning: Foundations, Algorithms, and Applications.
