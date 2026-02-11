# Model Card: Customer Churn Prediction API

## Model Details

### Overview
- **Model Name**: Customer Churn Predictor
- **Model Type**: Random Forest Classifier
- **Version**: 1.0.0
- **Date Created**: 2024-01-15
- **Organization**: RGT-NSS AI Training Program
- **License**: Educational Use

### Model Description
This model predicts the probability that a telecommunications customer will churn (cancel their service) within the next month. The model is designed to be deployed as a REST API microservice.

### Intended Use

#### Primary Use Cases
- **Customer Retention**: Identify at-risk customers for proactive retention campaigns
- **Resource Allocation**: Prioritize retention efforts on high-value customers
- **Business Analytics**: Understand factors driving customer churn

#### Users
- Customer Success Teams
- Marketing Analysts
- Data Science Teams

#### Out-of-Scope Uses
- This model should NOT be used for:
  - Real-time pricing decisions
  - Automated account termination without human review
  - Credit scoring or loan decisions
  - Any use outside telecommunications without retraining

## Model Architecture

```
Input Features (19) → ColumnTransformer → Random Forest → Churn Probability
```

### Preprocessing Pipeline
1. **Numeric Features** (`tenure`, `MonthlyCharges`, `TotalCharges`)
   - StandardScaler normalization
   
2. **Categorical Features** (16 features)
   - One-hot encoding with `handle_unknown='ignore'`

### Hyperparameters
```python
{
    'classifier__n_estimators': 200,
    'classifier__max_depth': 20,
    'classifier__min_samples_split': 2,
    'classifier__min_samples_leaf': 1
}
```

## Training Data

### Dataset
- **Source**: IBM Telco Customer Churn
- **URL**: https://www.kaggle.com/blastchar/telco-customer-churn
- **License**: CC0: Public Domain

### Statistics
| Property | Value |
|----------|-------|
| Total Samples | 7,043 |
| Training Samples | 4,930 (70%) |
| Validation Samples | 1,057 (15%) |
| Test Samples | 1,056 (15%) |
| Features | 19 |
| Churn Rate | 26.5% |

### Features

#### Demographics (4 features)
- `gender`: Male/Female
- `SeniorCitizen`: 0/1
- `Partner`: Yes/No
- `Dependents`: Yes/No

#### Account Information (8 features)
- `tenure`: Months as customer
- `Contract`: Month-to-month/One year/Two year
- `PaperlessBilling`: Yes/No
- `PaymentMethod`: Payment method used
- `MonthlyCharges`: Monthly bill amount
- `TotalCharges`: Total amount charged
- `PhoneService`: Yes/No
- `MultipleLines`: Yes/No/No phone service

#### Services (7 features)
- `InternetService`: DSL/Fiber optic/No
- `OnlineSecurity`: Yes/No/No internet service
- `OnlineBackup`: Yes/No/No internet service
- `DeviceProtection`: Yes/No/No internet service
- `TechSupport`: Yes/No/No internet service
- `StreamingTV`: Yes/No/No internet service
- `StreamingMovies`: Yes/No/No internet service

### Data Preprocessing
- **Missing Values**: 11 samples with missing `TotalCharges` (imputed with 0)
- **Encoding**: Target encoded as 1=Churn, 0=No Churn
- **Scaling**: StandardScaler for numeric features
- **One-Hot Encoding**: For categorical features

## Performance Metrics

### Test Set Performance
| Metric | Score |
|--------|-------|
| Accuracy | 0.79-0.82 |
| Precision | 0.65-0.70 |
| Recall | 0.50-0.60 |
| F1 Score | 0.57-0.64 |
| ROC-AUC | 0.82-0.85 |

### Cross-Validation Results
- **Method**: 5-Fold Stratified Cross-Validation
- **Mean F1**: 0.61 (±0.03)
- **Mean ROC-AUC**: 0.83 (±0.02)

### Performance by Segment

| Segment | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Month-to-Month | 0.71 | 0.62 | 0.66 |
| One Year Contract | 0.45 | 0.28 | 0.35 |
| Two Year Contract | 0.33 | 0.15 | 0.21 |
| Fiber Optic | 0.78 | 0.58 | 0.67 |
| DSL | 0.58 | 0.42 | 0.49 |

## Feature Importance

Top 10 features by permutation importance:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Contract | 0.184 |
| 2 | tenure | 0.152 |
| 3 | MonthlyCharges | 0.128 |
| 4 | TotalCharges | 0.095 |
| 5 | InternetService | 0.087 |
| 6 | PaymentMethod | 0.062 |
| 7 | OnlineSecurity | 0.048 |
| 8 | TechSupport | 0.041 |
| 9 | OnlineBackup | 0.035 |
| 10 | DeviceProtection | 0.028 |

## Ethical Considerations

### Fairness
- Model performance varies across contract types
- No sensitive demographic attributes (race, ethnicity) used
- Gender is included but not a primary driver of predictions

### Privacy
- No personally identifiable information used as features
- CustomerID is used for tracking but not as a model feature
- All data used is anonymized

### Limitations
- Dataset represents single company; may not generalize
- Snapshot in time; doesn't capture temporal trends
- Class imbalance affects minority class performance
- Geographic limitation (US-based data)

## API Specification

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc documentation |

### Example Request
```json
{
  "customer_id": "cust_001",
  "data": {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0
  }
}
```

### Example Response
```json
{
  "customer_id": "cust_001",
  "churn_probability": 0.23,
  "churn_prediction": false,
  "confidence": "high",
  "model_version": "1.0.0",
  "prediction_time": "2024-01-15T10:30:00Z"
}
```

## Deployment

### Requirements
- Python 3.9+
- FastAPI 0.104+
- scikit-learn 1.3+
- See `requirements.txt` for complete list

### Running the API
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Access documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Monitoring

### Metrics to Track
- Prediction volume
- Response latency (p50, p95, p99)
- Error rate
- Prediction distribution drift
- Feature drift (PSI)

### Retraining Schedule
- **Frequency**: Monthly
- **Trigger**: Accuracy drop >5% or significant drift
- **Validation**: Holdout validation before deployment

## Maintenance

### Version History
| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-15 | Initial release |

### Known Issues
- Lower performance on long-term contracts
- May require retraining for significant market changes

## Citation

```bibtex
@misc{churn_api_v1,
  title={Customer Churn Prediction API v1.0},
  author={RGT-NSS AI Training Program},
  year={2024},
  howpublished={\url{https://github.com/rgt-nss/churn-api}}
}
```

## Contact

For questions or issues:
- **Email**: ml-team@rgt-nss.edu
- **Issues**: https://github.com/rgt-nss/churn-api/issues

---

*This Model Card follows the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) framework.*
