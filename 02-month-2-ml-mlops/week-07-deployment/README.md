# Week 7: Model Deployment with FastAPI

## Overview
This week teaches you how to deploy machine learning models as REST APIs using FastAPI. You'll build a production-ready prediction service with proper error handling, input validation, and testing.

## Learning Objectives
By the end of this week, you will:
- Understand REST API principles for ML deployment
- Build APIs with FastAPI framework
- Validate inputs with Pydantic models
- Handle errors gracefully
- Write unit and integration tests
- Document APIs with automatic OpenAPI/Swagger

## Files
```
week-07-deployment/
├── api/
│   ├── main.py              # FastAPI application
│   └── __init__.py
├── models/
│   └── house_price_model.pkl  # Serialized model (place here)
├── tests/
│   ├── test_api.py          # Pytest test suite
│   └── __init__.py
├── requirements.txt         # Dependencies
└── README.md
```

## Setup

### Install Dependencies
```bash
pip install fastapi uvicorn pydantic scikit-learn joblib pytest requests
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

## Running the API

### Development Mode
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "model_loaded": true
}
```

### Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "LotArea": 8450,
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "TotalBsmtSF": 856,
    "GrLivArea": 1710,
    "GarageCars": 2,
    "GarageArea": 548
  }'
```

Response:
```json
{
  "predicted_price": 208500.5,
  "prediction_interval": [195000, 222000],
  "model_version": "1.0.0"
}
```

## API Documentation

Once running, access interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Architecture

```
┌──────────────┐      HTTP       ┌─────────────────────────────────────┐
│   Client     │ ◄──────────────► │          FastAPI Server             │
│  (curl,      │                  │  ┌─────────────────────────────┐    │
│   Python,    │                  │  │  Pydantic Validation        │    │
│   JS, etc.)  │                  │  │  - Type checking            │    │
└──────────────┘                  │  │  - Range validation         │    │
                                  │  └─────────────────────────────┘    │
                                  │                ↓                    │
                                  │  ┌─────────────────────────────┐    │
                                  │  │  ML Model (joblib)          │    │
                                  │  │  - Preprocessing pipeline   │    │
                                  │  │  - Trained estimator        │    │
                                  │  └─────────────────────────────┘    │
                                  │                ↓                    │
                                  │  ┌─────────────────────────────┐    │
                                  │  │  Response Formatting        │    │
                                  │  │  - Prediction + metadata    │    │
                                  │  └─────────────────────────────┘    │
                                  └─────────────────────────────────────┘
```

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/test_api.py::test_health_check -v
```

### Test Coverage
```bash
pytest tests/ --cov=api --cov-report=html
```

## Project Structure Explained

### `api/main.py`
Main FastAPI application with:
- Pydantic request/response models
- Prediction endpoint with error handling
- Health check endpoint
- Logging configuration

### `tests/test_api.py`
Comprehensive test suite including:
- Health endpoint tests
- Prediction endpoint tests
- Error handling tests
- Input validation tests

## Error Handling

The API handles these error scenarios:

| Error | HTTP Status | Description |
|-------|-------------|-------------|
| Validation Error | 422 | Invalid input data |
| Model Error | 500 | Prediction failed |
| Not Found | 404 | Invalid endpoint |

Example error response:
```json
{
  "detail": [
    {
      "loc": ["body", "LotArea"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    }
  ]
}
```

## Deployment Options

### 1. Docker (Recommended)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Cloud Platforms
- **AWS**: ECS, EKS, or Lambda
- **GCP**: Cloud Run or GKE
- **Azure**: Container Instances or AKS
- **Heroku**: Container deployment

### 3. On-Premise
- Use systemd or supervisor for process management
- Place behind Nginx reverse proxy

## Best Practices

1. **Always validate inputs** - Use Pydantic for strict typing
2. **Handle errors gracefully** - Return meaningful error messages
3. **Log predictions** - Track API usage and model performance
4. **Version your API** - Use URL versioning (/v1/predict)
5. **Monitor health** - Implement health checks for load balancers
6. **Load test** - Test with expected traffic volumes

## Exercises

1. **Add batch prediction endpoint** - Accept multiple records
2. **Implement rate limiting** - Use slowapi or middleware
3. **Add authentication** - API key or JWT token validation
4. **Create async endpoints** - For I/O bound operations
5. **Add model versioning** - Support multiple model versions

## Common Issues

### Issue: "Module not found" when importing
**Solution**: Run from project root or use `python -m pytest`

### Issue: Model file not found
**Solution**: Ensure model is in `models/` directory or update path in main.py

### Issue: Port already in use
**Solution**: Change port: `--port 8001` or kill existing process

## Next Week Preview
Week 8 covers MLOps fundamentals including model cards, monitoring, and responsible AI practices.

---

## Assignment
Deploy your Week 6 model as a FastAPI service:
1. Create Pydantic models for all inputs
2. Implement /predict and /health endpoints
3. Write comprehensive tests (min 5 test cases)
4. Document the API with usage examples
5. Deploy locally and test with curl/requests
