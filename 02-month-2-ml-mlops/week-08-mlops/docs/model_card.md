# Model Card: Customer Churn Prediction

## Model Overview

| Attribute | Value |
|-----------|-------|
| **Model Name** | Telco Customer Churn Predictor |
| **Version** | 1.0.0 |
| **Date** | February 2025 |
| **Author** | RGT-NSS Training Program |
| **Model Type** | Random Forest Classifier |
| **Task** | Binary Classification |

---

## Model Description

### Purpose
This model predicts whether a telecommunications customer is likely to churn (cancel their service) based on demographic, account, and service usage information.

### Intended Use
- **Primary**: Identify at-risk customers for proactive retention campaigns
- **Users**: Marketing teams, customer success managers, business analysts
- **Deployment**: Real-time API integration with CRM systems

### Out of Scope
- Predicting exact churn date
- Determining root causes of churn (correlation ≠ causation)
- Making automated decisions without human review

---

## Training Data

### Dataset
- **Source**: IBM Telco Customer Churn Dataset
- **Size**: 7,043 customers
- **Time Period**: Cross-sectional (snapshot)
- **Geography**: United States

### Features (21 total)

| Category | Features |
|----------|----------|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod |
| Charges | MonthlyCharges, TotalCharges |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |

### Target Variable
- **Churn**: Whether customer left service (Yes/No)
- **Distribution**: 26.5% churn, 73.5% retained

### Preprocessing
1. **TotalCharges**: Converted to numeric, 11 missing values filled with 0
2. **Encoding**: One-hot encoding for categorical variables
3. **Scaling**: StandardScaler applied to numeric features

---

## Model Architecture

### Algorithm
**Random Forest Classifier**

### Hyperparameters (Tuned)
| Parameter | Value | Description |
|-----------|-------|-------------|
| n_estimators | 200 | Number of trees in forest |
| max_depth | 10 | Maximum depth of each tree |
| min_samples_split | 2 | Minimum samples required to split node |
| min_samples_leaf | 1 | Minimum samples required at leaf node |
| class_weight | balanced | Handles class imbalance |

### Training Process
- **Algorithm**: Grid Search with 5-fold Cross-Validation
- **Optimization Metric**: F1 Score
- **Train/Test Split**: 80/20 with stratification

---

## Performance Metrics

### Test Set Results (n=1,409)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 77.3% | Overall correct predictions |
| **Precision** | 54.2% | Of predicted churners, 54% actually churned |
| **Recall** | 73.8% | Identified 74% of actual churners |
| **F1 Score** | 63.3% | Balance of precision and recall |
| **ROC-AUC** | 84.2% | Good discrimination ability |

### Cross-Validation
- **F1 Score**: 0.6372 (±0.027)
- **Stability**: Low variance indicates robust model

### Feature Importance (Top 5)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | tenure | 17.5% |
| 2 | TotalCharges | 14.1% |
| 3 | Contract_Two year | 10.2% |
| 4 | MonthlyCharges | 10.1% |
| 5 | InternetService_Fiber optic | 6.3% |

---

## Limitations

### Known Limitations
1. **Temporal**: Cross-sectional data doesn't capture time-series patterns
2. **Causality**: Correlations don't imply causation
3. **Geographic**: Trained on US data, may not generalize globally
4. **Temporal Validity**: Model may degrade as customer behavior changes

### Performance Limitations
- **False Positives**: 46% of churn predictions are false alarms
- **Imbalanced Data**: May be biased toward majority class
- **Feature Coverage**: Doesn't include customer service interactions

---

## Ethical Considerations

### Fairness
- **Protected Attributes**: Model uses gender and senior citizen status
- **Monitoring**: Should audit predictions for demographic bias
- **Recommendation**: Regular fairness audits across demographic groups

### Privacy
- **Data Sensitivity**: Uses customer financial and demographic data
- **Compliance**: Ensure GDPR/CCPA compliance in deployment
- **Retention**: Implement data retention policies

### Transparency
- **Explainability**: Random Forest provides feature importance
- **Documentation**: This model card provides full transparency
- **User Notification**: Customers should be informed about prediction use

---

## Deployment Information

### API Specification
- **Framework**: FastAPI
- **Endpoint**: POST /predict
- **Input**: JSON with 19 customer features
- **Output**: Churn prediction (0/1) and probability (0-1)

### Monitoring
- **Health Check**: GET /health
- **Logging**: All predictions logged with timestamp
- **Versioning**: API version tracked separately from model version

### Maintenance
- **Retraining**: Monthly or when performance degrades >5%
- **Threshold**: F1 score < 0.60 triggers review
- **Data Drift**: Monitor input feature distributions

---

## How to Use

### Python Example
```python
import requests

customer = {
    "tenure": 12,
    "Contract": "Month-to-month",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
    # ... other features
}

response = requests.post("http://api/predict", json=customer)
result = response.json()

print(f"Churn Probability: {result['churn_probability']:.2%}")
```

### Interpreting Results
- **Probability > 0.5**: Model predicts churn
- **Probability 0.3-0.5**: Medium risk, monitor closely
- **Probability < 0.3**: Low risk

---

## Citation

```
RGT-NSS Training Program (2025). 
Customer Churn Prediction Model v1.0.0.
Week 6-7: ML Pipelines and Deployment.
```

---

## Contact

For questions or issues:
- Email: ml-team@example.com
- Issues: https://github.com/samuel-1-avson/RGT-NSS/issues

---

*Last Updated: February 2025*
