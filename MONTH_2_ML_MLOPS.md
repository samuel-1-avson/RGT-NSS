# 🤖 Month 2: Applied ML & MLOps

> **"From data to deployment - building production-ready ML systems"**

![ML Banner](https://img.shields.io/badge/Month-2_ML_&_MLOps-purple?style=for-the-badge)
![Duration](https://img.shields.io/badge/Duration-4_Weeks-green?style=for-the-badge)
![Tools](https://img.shields.io/badge/Tools-scikit--learn_|_FastAPI_|_Docker-orange?style=for-the-badge)

---

## 🎯 Learning Outcomes

By the end of Month 2, you will be able to:

- ✅ Build and evaluate **machine learning models** with scikit-learn
- ✅ Create **reproducible ML pipelines** with proper preprocessing
- ✅ Perform **hyperparameter tuning** using cross-validation
- ✅ Deploy models as **REST APIs** with FastAPI
- ✅ Implement **MLOps best practices** including monitoring and model cards
- ✅ Ship a complete **ML microservice** to production

---

## 📅 Week-by-Week Breakdown

### 🧠 Week 5: Supervised Learning 1 - Baseline Models

**Theme:** *Building Your First Predictive Models*

#### What We Covered
- **ML Problem Framing**: Classification vs Regression
- **Train/Test Split**: Proper data partitioning with stratification
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Baseline Models**: Logistic Regression & Random Forest

#### 📈 Dataset: Telco Customer Churn (Same as Month 1)
| Metric | Value |
|--------|-------|
| **Records** | 7,043 customers |
| **Target** | Churn (Yes/No) |
| **Class Balance** | 73.5% No Churn, 26.5% Churn |
| **Features** | 30 (after encoding) |

#### 🏆 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 73.81% | 50.43% | 78.88% | 0.6152 | 0.8421 |
| **Random Forest** | 77.29% | 55.42% | 73.80% | **0.6330** | 0.8418 |

```
🎯 WINNER: Random Forest
   - Best F1 Score: 0.6330
   - Best Accuracy: 77.29%
   - Best Precision: 55.42%
   - Good balance of precision and recall
```

#### 🔍 Model Selection Rationale
```
Why Random Forest?
✅ Highest F1 score (best balance)
✅ Robust to outliers
✅ Handles mixed data types well
✅ Provides feature importance
✅ Less prone to overfitting than single trees

Why NOT Logistic Regression?
- Lower F1 score
- Assumes linear relationships
- Less expressive for complex patterns
```

#### 🛠️ Deliverables
- ✅ Trained baseline models
- ✅ ROC curve comparison plot
- ✅ Model selection rationale document
- ✅ Performance metrics in CSV/JSON

---

### ⚙️ Week 6: Supervised Learning 2 - ML Pipelines

**Theme:** *Production-Ready Machine Learning*

#### What We Covered
- **ML Pipelines**: Combining preprocessing and modeling
- **ColumnTransformer**: Different preprocessing for different feature types
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Feature Importance**: Understanding what drives predictions

#### 🔄 Pipeline Architecture
```python
# Step 1: Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),      # Scale numeric
    ('cat', OneHotEncoder(drop='first'), categorical) # Encode categorical
])

# Step 2: Full Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    ))
])

# Step 3: Hyperparameter Tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None]
}

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5,           # 5-fold cross-validation
    scoring='f1',   # Optimize F1 score
    n_jobs=-1       # Use all CPU cores
)
```

#### 📊 Hyperparameter Tuning Results

| Configuration | CV F1 Score | Status |
|--------------|-------------|--------|
| n_estimators=100, max_depth=10 | 0.6351 | ❌ |
| n_estimators=100, max_depth=20 | 0.6368 | ❌ |
| **n_estimators=200, max_depth=10** | **0.6372** | ✅ **BEST** |
| n_estimators=200, max_depth=20 | 0.6365 | ❌ |

#### 🔥 Feature Importance (Top 10)

| Rank | Feature | Importance | Business Insight |
|------|---------|------------|------------------|
| 1 | **tenure** | 17.50% | Customer loyalty matters most |
| 2 | **TotalCharges** | 14.10% | Lifetime value indicator |
| 3 | **Contract_Two year** | 10.17% | Long contracts retain customers |
| 4 | **MonthlyCharges** | 10.14% | Price sensitivity |
| 5 | **InternetService_Fiber optic** | 6.30% | Fiber customers churn more |
| 6 | **PaymentMethod_Electronic check** | 5.39% | Payment method matters |
| 7 | **Contract_One year** | 4.35% | Annual contracts help |
| 8 | **OnlineSecurity_Yes** | 3.42% | Security features retain |
| 9 | **TechSupport_Yes** | 2.58% | Support matters |
| 10 | **PaperlessBilling_Yes** | 2.05% | Digital billing slight impact |

#### 🎯 Key Insights
```
💡 TOP 3 DRIVERS OF CHURN:
   1. Low tenure (new customers at risk)
   2. Low total charges (new/disengaged)
   3. Month-to-month contracts (no commitment)

🎯 ACTIONABLE RECOMMENDATIONS:
   - Focus retention on customers < 12 months tenure
   - Encourage contract upgrades (monthly → annual)
   - Improve fiber optic service quality
   - Promote online security & tech support
```

#### 🛠️ Deliverables
- ✅ Complete ML pipeline (preprocessing + model)
- ✅ Tuned model saved: `tuned_random_forest.pkl`
- ✅ Feature importance visualization
- ✅ Cross-validation results

---

### 🚀 Week 7: Data to Deployment - FastAPI

**Theme:** *Serving ML Models in Production*

#### What We Covered
- **API Design**: RESTful principles for ML services
- **FastAPI Framework**: Modern, fast Python web framework
- **Pydantic**: Data validation with Python type hints
- **Model Serving**: Loading and using trained models
- **API Documentation**: Auto-generated Swagger UI

#### 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/docs` | Swagger UI documentation |

#### 📡 API Request/Response Example

**Request:**
```json
{
  "tenure": 1,
  "Contract": "Month-to-month",
  "MonthlyCharges": 99.65,
  "TotalCharges": 99.65,
  "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check"
}
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8472,
  "model_version": "1.0.0",
  "timestamp": "2025-02-11T10:30:00"
}
```

#### 🔧 API Features

```python
# Input validation with Pydantic
class CustomerData(BaseModel):
    tenure: int = Field(default=1, description="Months as customer")
    Contract: str = Field(default="Month-to-month")
    MonthlyCharges: float = Field(default=29.85)
    # ... 16 more features

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": "1.0.0"
    }

# Prediction endpoint with error handling
@app.post("/predict")
def predict(customer: CustomerData):
    try:
        result = model.predict([customer.dict()])
        return {
            "churn_prediction": int(result[0]),
            "churn_probability": float(proba[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 🛠️ Deliverables
- ✅ FastAPI application (`api/main.py`)
- ✅ Auto-generated API docs (Swagger UI)
- ✅ Health check endpoint
- ✅ Error handling and logging
- ✅ API testing script

---

### 🔍 Week 8: MLOps & Model Monitoring

**Theme:** *Production ML Best Practices*

#### What We Covered
- **Model Cards**: Documenting models for transparency
- **Monitoring**: Tracking predictions and detecting drift
- **Logging**: Structured logging for observability
- **Ethics**: Fairness, privacy, and responsible AI

#### 📋 Model Card Highlights

```yaml
Model: Telco Customer Churn Predictor v1.0.0
Type: Random Forest Classifier

Intended Use:
  - Identify at-risk customers
  - Enable proactive retention
  
Performance:
  - F1 Score: 0.6372
  - ROC-AUC: 0.8421
  - Accuracy: 77.3%

Limitations:
  - Trained on US data only
  - May not generalize globally
  - 46% false positive rate

Ethical Considerations:
  - Uses gender/senior status (monitor for bias)
  - Requires GDPR compliance
  - Should not be sole decision factor
```

#### 📊 Monitoring System

```python
class ModelMonitor:
    """Track predictions and detect drift"""
    
    def log_prediction(self, customer_id, prediction, prob):
        # Log to file
        logger.info(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'customer_id': customer_id,
            'prediction': prediction,
            'probability': prob
        }))
    
    def check_drift(self, reference_rate=0.265):
        """Alert if churn rate drifts >10%"""
        current_rate = self.churn_count / self.total_count
        drift = abs(current_rate - reference_rate)
        
        if drift > 0.10:
            alert(f"DRIFT: Churn rate changed by {drift:.1%}")
```

#### 🚨 Monitoring Metrics

| Metric | Threshold | Action |
|--------|-----------|--------|
| **Prediction Count** | Daily tracking | Monitor volume |
| **Churn Rate** | ±10% from 26.5% | Alert on drift |
| **Latency** | <100ms average | Optimize if high |
| **Error Rate** | <1% | Investigate if exceeded |

#### 🛠️ Deliverables
- ✅ Complete Model Card documentation
- ✅ Monitoring system (`scripts/monitoring.py`)
- ✅ Structured JSON logging
- ✅ Drift detection alerts

---

## 🏆 Milestone Project B: ML Microservice

**Weight:** 25% of total grade

### Project Overview
Ship a complete ML microservice with trained model, API, and documentation.

### Architecture
```
┌─────────────────┐
│   Client/App    │
└────────┬────────┘
         │ HTTP POST /predict
         ▼
┌─────────────────┐
│   FastAPI App   │
│   (Port 8000)   │
└────────┬────────┘
         │ Load Model
         ▼
┌─────────────────┐
│  Trained Model  │
│  (joblib .pkl)  │
└─────────────────┘
```

### Components

#### 1️⃣ Trained Model
```python
# Complete training pipeline
- Data preprocessing
- Hyperparameter tuning
- Cross-validation
- Model serialization

# Saved to:
models/churn_model.pkl
```

#### 2️⃣ FastAPI Application
```python
# Endpoints:
GET  /health     → Check service status
POST /predict    → Single prediction
POST /predict/batch → Multiple predictions

# Features:
- Input validation (Pydantic)
- Error handling
- Request logging
- Auto-generated docs
```

#### 3️⃣ Testing Suite
```python
# Unit tests for:
- Data preprocessing
- Model predictions
- API endpoints

# Integration tests:
- Full prediction flow
- Error scenarios
```

#### 4️⃣ Documentation
- **README**: Setup instructions, API usage
- **Model Card**: Performance, limitations, ethics
- **API Docs**: Auto-generated at /docs

### Performance Summary

| Metric | Value |
|--------|-------|
| **F1 Score** | 0.6217 |
| **Test Accuracy** | ~77% |
| **API Latency** | <50ms |
| **Model Size** | ~2 MB |

### Running the Microservice

```bash
# Install dependencies
pip install fastapi uvicorn scikit-learn pandas

# Start API server
uvicorn api.main:app --reload --port 8000

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "Contract": "Month-to-month",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0
  }'
```

---

## 🛠️ Tech Stack Used

| Tool | Purpose |
|------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) | Programming language |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | Machine learning |
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) | API framework |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) | Data manipulation |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) | Numerical computing |
| ![Joblib](https://img.shields.io/badge/Joblib-blue?logo=python&logoColor=white) | Model serialization |
| ![Uvicorn](https://img.shields.io/badge/Uvicorn-46A2C8?logo=python&logoColor=white) | ASGI server |
| ![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white) | Version control |

---

## 📊 Key Metrics Summary

| Week | Focus | Model Performance | Key Deliverable |
|------|-------|-------------------|-----------------|
| Week 5 | Baseline Models | F1: 0.6330 | Model comparison |
| Week 6 | Pipelines & Tuning | F1: 0.6372 | Tuned model + pipeline |
| Week 7 | Deployment | API live | FastAPI service |
| Week 8 | MLOps | Monitoring | Model card + monitoring |
| **Milestone B** | Full Microservice | F1: 0.6217 | Production API |

---

## 🎓 Skills Acquired

### Technical Skills
- ✅ ML model training and evaluation
- ✅ Feature engineering and preprocessing
- ✅ Hyperparameter tuning (GridSearchCV)
- ✅ ML pipeline architecture
- ✅ REST API development (FastAPI)
- ✅ Model serialization (joblib)
- ✅ API testing and documentation
- ✅ MLOps monitoring and logging

### Soft Skills
- ✅ Model selection and justification
- ✅ Production system design
- ✅ Technical documentation
- ✅ API design principles

---

## 🚀 Model Performance Journey

```
Week 5 (Baseline)          Week 6 (Tuned)             Milestone B
┌─────────────┐           ┌─────────────┐           ┌─────────────┐
│ F1: 0.6330  │   ──▶    │ F1: 0.6372  │   ──▶    │ F1: 0.6217  │
│ AUC: 0.8418 │           │ AUC: 0.8421 │           │ AUC: ~0.84  │
│ Acc: 77.3%  │           │ Acc: 77.3%  │           │ Acc: ~77%   │
└─────────────┘           └─────────────┘           └─────────────┘
    Basic RF                  Optimized                  Deployed
```

---

## 📁 Repository Structure

```
02-month-2-ml-mlops/
├── week-05-supervised-ml-1/
│   └── scripts/baseline_models.py
├── week-06-supervised-ml-2/
│   ├── scripts/ml_pipeline.py
│   └── models/tuned_random_forest.pkl
├── week-07-deployment/
│   ├── api/main.py
│   └── scripts/test_api.py
├── week-08-mlops/
│   ├── docs/model_card.md
│   └── scripts/monitoring.py
└── milestone-project-b/
    ├── api/main.py
    ├── models/churn_model.pkl
    └── scripts/train_and_deploy.py
```

---

## 📈 Model in Production

### API Usage Example
```python
import requests

# Customer data
customer = {
    "tenure": 5,
    "Contract": "Month-to-month",
    "MonthlyCharges": 85.0,
    "TotalCharges": 425.0,
    "InternetService": "Fiber optic"
}

# Get prediction
response = requests.post(
    "http://api.example.com/predict",
    json=customer
)

result = response.json()
print(f"Churn Risk: {result['churn_probability']:.1%}")
# Output: Churn Risk: 72.4%
```

---

## 🎯 Business Impact

| Metric | Before ML | After ML | Improvement |
|--------|-----------|----------|-------------|
| **Retention Rate** | 73.5% | Target: 78% | +4.5% projected |
| **Campaign Efficiency** | Generic | Targeted | 2x better targeting |
| **Response Time** | Manual analysis | <50ms API | Real-time |
| **Cost per Retention** | $100 | Target: $60 | 40% reduction |

---

## 🔗 Resources

- **scikit-learn**: [scikit-learn.org](https://scikit-learn.org/stable/)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
- **MLOps Guide**: [ml-ops.org](https://ml-ops.org/)
- **Model Cards**: [Google AI](https://modelcards.withgoogle.com/)

---

**🔗 Repository:** https://github.com/samuel-1-avson/RGT-NSS

**📅 Completed:** February 2025

---

*This project was completed as part of the RGT 2025 NSP AI/Data/LLM Training Program.*

**🎉 Ready for Month 3: Generative AI & LLMs!**
