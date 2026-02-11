# Model Card: Customer Churn Prediction Model

## Model Details

### Overview
- **Model Name**: Customer Churn Predictor v1.0
- **Model Type**: Gradient Boosting Classifier (XGBoost)
- **Version**: 1.0.0
- **Date Created**: 2024-01-15
- **Last Updated**: 2024-01-15
- **Organization**: RGT-NSS AI Training Program
- **Contact**: ml-team@rgt-nss.edu

### Model Description
This model predicts the probability that a telecommunications customer will churn (cancel their service) within the next month. The model is trained on historical customer data including demographics, service usage, and account information.

### Intended Use

#### Primary Use Cases
- **Proactive Retention**: Identify customers at risk of churning for targeted retention campaigns
- **Resource Allocation**: Prioritize retention efforts on high-value at-risk customers
- **Business Intelligence**: Understand factors driving customer churn

#### Target Users
- Customer Success Teams
- Marketing Analysts
- Business Intelligence Teams

#### Out-of-Scope Uses
- This model should NOT be used for:
  - Real-time pricing decisions
  - Automated account termination
  - Credit scoring or loan decisions
  - Any use outside telecommunications industry without retraining

### Model Architecture

```
Input Features (20) → Preprocessing Pipeline → XGBoost Classifier → Churn Probability
```

#### Preprocessing Steps
1. **Numeric Features**: StandardScaler normalization
2. **Categorical Features**: One-hot encoding with handle_unknown='ignore'
3. **Missing Values**: Imputed with median (numeric) or mode (categorical)

#### Hyperparameters
```yaml
n_estimators: 200
max_depth: 6
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
min_child_weight: 1
objective: binary:logistic
eval_metric: auc
```

## Training Data

### Data Source
- **Dataset**: Telco Customer Churn
- **Source**: IBM Sample Data
- **URL**: https://www.kaggle.com/blastchar/telco-customer-churn
- **License**: CC0: Public Domain

### Data Statistics
- **Total Samples**: 7,043
- **Time Period**: Q1-Q4 2023
- **Geographic Coverage**: United States

#### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| No Churn | 5,174 | 73.5% |
| Churn | 1,869 | 26.5% |

#### Feature Summary
| Feature Type | Count | Examples |
|--------------|-------|----------|
| Demographic | 3 | gender, SeniorCitizen, Partner |
| Services | 7 | InternetService, OnlineSecurity, TechSupport |
| Account | 5 | Contract, PaymentMethod, MonthlyCharges |
| Usage | 3 | tenure, TotalCharges |

### Data Preprocessing
- **Train/Validation/Test Split**: 70/15/15
- **Stratified Split**: Yes (maintains class distribution)
- **Missing Values**: 11 samples with missing TotalCharges (imputed with 0)

### Data Limitations
- Dataset represents single telecommunications company
- Limited temporal information (snapshot in time)
- No customer satisfaction scores
- No competitive pricing information

## Evaluation Data

### Test Set Performance

#### Overall Metrics
| Metric | Score | Confidence Interval |
|--------|-------|---------------------|
| Accuracy | 0.804 | [0.782, 0.826] |
| Precision | 0.673 | [0.635, 0.711] |
| Recall | 0.526 | [0.481, 0.571] |
| F1 Score | 0.591 | [0.552, 0.630] |
| ROC-AUC | 0.845 | [0.825, 0.865] |

#### Performance by Segment

| Segment | Precision | Recall | F1 | Sample Size |
|---------|-----------|--------|-----|-------------|
| Month-to-Month Contract | 0.71 | 0.62 | 0.66 | 423 |
| One Year Contract | 0.45 | 0.28 | 0.35 | 142 |
| Two Year Contract | 0.33 | 0.15 | 0.21 | 95 |
| Fiber Optic Customers | 0.78 | 0.58 | 0.67 | 312 |
| DSL Customers | 0.58 | 0.42 | 0.49 | 241 |
| Senior Citizens | 0.72 | 0.55 | 0.62 | 183 |
| Non-Senior Citizens | 0.64 | 0.51 | 0.57 | 477 |

### Cross-Validation Results
- **Method**: 5-Fold Stratified Cross-Validation
- **Mean F1**: 0.589 (±0.023)
- **Mean ROC-AUC**: 0.842 (±0.018)

## Performance Metrics

### Threshold Analysis
The default prediction threshold is 0.5. Performance at different thresholds:

| Threshold | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| 0.3 | 0.52 | 0.78 | 0.62 |
| 0.4 | 0.60 | 0.65 | 0.62 |
| **0.5** | **0.67** | **0.53** | **0.59** |
| 0.6 | 0.74 | 0.42 | 0.54 |
| 0.7 | 0.81 | 0.31 | 0.45 |

### Confusion Matrix (Threshold = 0.5)
```
                Predicted
              No Churn  Churn
Actual No Churn   734    42
       Churn      101   112
```

### Feature Importance
Top 10 features by permutation importance:

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | Contract | 0.184 | Contract type (Month-to-month highest risk) |
| 2 | tenure | 0.152 | Months as customer |
| 3 | MonthlyCharges | 0.128 | Monthly bill amount |
| 4 | TotalCharges | 0.095 | Total amount charged |
| 5 | InternetService | 0.087 | Type of internet service |
| 6 | PaymentMethod | 0.062 | Payment method used |
| 7 | OnlineSecurity | 0.048 | Online security service |
| 8 | TechSupport | 0.041 | Tech support subscription |
| 9 | OnlineBackup | 0.035 | Online backup service |
| 10 | DeviceProtection | 0.028 | Device protection plan |

## Ethical Considerations

### Fairness Analysis

#### Demographic Parity
- **Gender**: Model shows slight bias toward predicting male customers as higher churn risk (2% difference in positive prediction rate)
- **Age (Senior Citizens)**: Model performs better for senior citizens (higher recall)

#### Recommendations for Fair Use
1. Monitor prediction rates across demographic groups
2. Ensure retention offers are equally accessible
3. Avoid using predictions to justify discriminatory pricing

### Privacy Considerations
- Model does not use sensitive personal identifiers (SSN, account numbers)
- Customer ID is used only for tracking, not as a feature
- All data is anonymized in training dataset

### Potential Biases
1. **Survivorship Bias**: Only existing customers in training data
2. **Temporal Bias**: Model trained on specific time period; economic conditions may change
3. **Geographic Bias**: US-only data may not generalize to other markets

## Limitations and Recommendations

### Known Limitations
1. **Binary Classification**: Only predicts churn/no churn, not time-to-churn
2. **Snapshot Data**: Doesn't capture temporal patterns or seasonality
3. **Missing Context**: No competitive intelligence or market factors
4. **Class Imbalance**: Despite balancing techniques, minority class performance is lower

### When NOT to Use
- During major service disruptions (data distribution shift)
- For customers with < 1 month tenure (insufficient data)
- Without human review for high-stakes decisions

### Recommendations for Use

#### Pre-Deployment
- [ ] Validate on recent data (within 3 months)
- [ ] Test on target geographic region
- [ ] Establish monitoring dashboard
- [ ] Define rollback criteria

#### In Production
- [ ] Monitor prediction drift weekly
- [ ] Retrain monthly with new data
- [ ] A/B test retention strategies
- [ ] Collect feedback on prediction accuracy

#### Threshold Selection Guidance
| Use Case | Recommended Threshold | Rationale |
|----------|----------------------|-----------|
| High-precision targeting | 0.6 | Minimize false positives (wasted retention spend) |
| Balanced approach | 0.5 | Default balanced precision/recall |
| Maximum recall | 0.3 | Capture all possible churners (high retention budget) |

## Monitoring and Maintenance

### Monitoring Metrics
- Prediction distribution drift
- Feature drift (PSI > 0.2)
- Model accuracy decay
- Prediction latency (target: <100ms p99)

### Retraining Schedule
- **Frequency**: Monthly
- **Trigger**: Accuracy drop >5% or data drift detected
- **Validation**: 1-week holdout before full deployment

### Incident Response
| Severity | Criteria | Response |
|----------|----------|----------|
| Critical | Accuracy <70% or latency >500ms | Rollback to previous version |
| High | PSI >0.3 on key features | Immediate investigation |
| Medium | Prediction distribution shift | Schedule review within 48h |
| Low | Single feature drift | Log and monitor |

## Citation

If using this model in research or production, please cite:

```bibtex
@misc{churn_model_v1,
  title={Customer Churn Prediction Model v1.0},
  author={RGT-NSS AI Training Program},
  year={2024},
  organization={RGT-NSS},
  howpublished={Model Card}
}
```

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-15 | Initial release |

## Contact

For questions about this model:
- **Email**: ml-team@rgt-nss.edu
- **Issues**: https://github.com/rgt-nss/ml-models/issues
- **Documentation**: https://docs.rgt-nss.edu/models/churn-v1

---

*This Model Card follows the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) framework proposed by Mitchell et al. (2019).*
