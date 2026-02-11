# Milestone Project B: ML Microservice

## Overview
This milestone project consolidates everything learned in Weeks 5-8 into a production-ready ML microservice. You'll build an end-to-end customer churn prediction API with proper MLOps practices.

## Project Goals
Build a complete ML microservice that:
- Trains and validates a churn prediction model
- Exposes predictions via REST API
- Includes comprehensive testing
- Documents the model with a Model Card
- Implements monitoring and logging

## Final Deliverables
```
milestone-project-b/
├── api/
│   ├── main.py              # FastAPI application
│   ├── __init__.py
│   └── config.py            # Configuration settings
├── models/
│   ├── churn_model.pkl      # Serialized model
│   ├── preprocessor.pkl     # Preprocessing pipeline
│   ├── feature_names.json   # Feature metadata
│   └── model_metadata.json  # Training info
├── notebooks/
│   └── train_model.ipynb    # Training notebook
├── tests/
│   ├── test_api.py          # API tests
│   ├── test_model.py        # Model tests
│   └── __init__.py
├── docs/
│   └── model_card.md        # Model documentation
├── requirements.txt         # All dependencies
└── README.md                # This file
```

## Quick Start

### 1. Installation
```bash
# Clone/navigate to project
cd milestone-project-b

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model (if needed)
```bash
jupyter notebook notebooks/train_model.ipynb
```

### 3. Run API
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test API
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "monthly_charges": 70.5,
    "total_charges": 850.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check"
  }'
```

### 5. Run Tests
```bash
pytest tests/ -v --cov=api --cov-report=html
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model status |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model metadata |
| `/docs` | GET | Swagger UI documentation |

### Request/Response Examples

#### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "tenure": 24,
  "monthly_charges": 65.0,
  "total_charges": 1560.0,
  "contract": "One year",
  "internet_service": "DSL",
  "payment_method": "Mailed check",
  "online_security": "Yes",
  "tech_support": "No"
}
```
Response:
```json
{
  "customer_id": "cust_001",
  "churn_probability": 0.23,
  "churn_prediction": false,
  "confidence": "high",
  "model_version": "1.0.0",
  "prediction_time": "2024-01-15T10:30:05Z"
}
```

#### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "customers": [
    {"tenure": 12, "monthly_charges": 70.0, ...},
    {"tenure": 24, "monthly_charges": 65.0, ...}
  ]
}
```

## Model Information

### Algorithm
- **Type**: Gradient Boosting Classifier
- **Framework**: Scikit-learn
- **Version**: 1.0.0

### Performance Metrics
| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| Accuracy | 0.85 | 0.82 | 0.81 |
| Precision | 0.78 | 0.74 | 0.73 |
| Recall | 0.72 | 0.68 | 0.67 |
| F1 Score | 0.75 | 0.71 | 0.70 |
| ROC-AUC | 0.89 | 0.85 | 0.84 |

### Features
| Feature | Type | Description |
|---------|------|-------------|
| tenure | numeric | Months as customer |
| monthly_charges | numeric | Monthly bill amount |
| total_charges | numeric | Total amount charged |
| contract | categorical | Contract type |
| internet_service | categorical | Internet service type |
| payment_method | categorical | Payment method |

## Testing

### Test Coverage
```
tests/
├── test_api.py          # API endpoint tests
│   ├── test_health_endpoint
│   ├── test_predict_endpoint
│   ├── test_batch_predict
│   └── test_error_handling
├── test_model.py        # Model tests
│   ├── test_prediction
│   ├── test_feature_validation
│   └── test_performance
└── conftest.py          # Pytest fixtures
```

### Running Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=api --cov-report=html

# Specific test
pytest tests/test_api.py::test_predict_endpoint -v
```

## Model Card

See [docs/model_card.md](docs/model_card.md) for complete model documentation including:
- Model description and intended use
- Training data summary
- Performance metrics
- Limitations and biases
- Ethical considerations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client                                    │
│                   (Web App / Mobile)                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/HTTPS
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Load Balancer                               │
│                     (Nginx / AWS ALB)                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │ API     │   │ API     │   │ API     │
        │ Instance│   │ Instance│   │ Instance│
        │ 1       │   │ 2       │   │ 3       │
        └────┬────┘   └────┬────┘   └────┬────┘
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    ┌──────────────┐
                    │ Monitoring   │
                    │ (Prometheus) │
                    └──────────────┘
```

## Deployment

### Docker
```bash
# Build
docker build -t churn-api:latest .

# Run
docker run -p 8000:8000 churn-api:latest
```

### Docker Compose
```bash
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

## Monitoring

### Metrics Tracked
- Request latency (p50, p95, p99)
- Prediction throughput
- Error rates
- Model prediction distribution

### Logging
Structured JSON logs including:
- Request ID
- Input features
- Prediction output
- Latency
- Timestamp

## Development

### Adding New Features
1. Update Pydantic models in `api/main.py`
2. Add tests in `tests/`
3. Update documentation
4. Run full test suite

### Retraining Model
1. Update training data
2. Run `notebooks/train_model.ipynb`
3. Validate new model performance
4. Update `model_metadata.json`
5. Deploy new model

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Functionality | 30% | API works correctly, all endpoints functional |
| Code Quality | 25% | Clean, documented, follows best practices |
| Testing | 20% | Comprehensive test coverage, all tests pass |
| Documentation | 15% | Clear README, Model Card, inline docs |
| MLOps | 10% | Logging, monitoring, proper serialization |

## Submission Checklist

- [ ] All tests pass (`pytest tests/`)
- [ ] Model Card completed
- [ ] README is comprehensive
- [ ] API runs without errors
- [ ] Requirements.txt is complete
- [ ] Code is well-documented
- [ ] Git repository initialized (if applicable)

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## Support

For questions or issues:
1. Check existing documentation
2. Review test cases for usage examples
3. Consult course materials from Weeks 5-8

---

## License

This project is for educational purposes as part of the RGT-NSS AI Training Program.

## Acknowledgments

- Telco Customer Churn dataset from IBM
- FastAPI framework by Sebastián Ramírez
- Scikit-learn community
