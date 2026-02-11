# Month 2: Machine Learning & MLOps (Weeks 5-8)

## Overview

This module covers the foundations of supervised machine learning, ML pipelines, model deployment, and MLOps practices. By the end of this month, you will be able to build, tune, deploy, and monitor machine learning models in production.

## Learning Path

```
Week 5: Baseline Models
    ↓
Week 6: ML Pipelines
    ↓
Week 7: FastAPI Deployment
    ↓
Week 8: MLOps & Monitoring
    ↓
Milestone Project B: ML Microservice
```

## Week-by-Week Breakdown

### Week 5: Supervised Learning - Baseline Models
**Directory**: `week-05-supervised-ml-1/`

Learn the fundamentals of supervised learning by training and comparing baseline models.

**Topics Covered**:
- Data preparation and preprocessing
- Train/test splits with stratification
- Logistic Regression and Random Forest
- Evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ROC curve visualization
- Model comparison

**Key Files**:
- `notebooks/week05_baseline_models.ipynb` - Interactive notebook
- `docs/model_selection_rationale.md` - Model selection guide

**Dataset**: Telco Customer Churn (7,043 customers)

### Week 6: Supervised Learning - ML Pipelines
**Directory**: `week-06-supervised-ml-2/`

Build production-ready ML pipelines with proper preprocessing and hyperparameter tuning.

**Topics Covered**:
- sklearn Pipelines
- ColumnTransformer for mixed data types
- GridSearchCV for hyperparameter tuning
- Permutation importance analysis
- Model serialization with joblib

**Key Files**:
- `scripts/week06_ml_pipeline.py` - Production-ready script
- `notebooks/week06_pipeline.ipynb` - Interactive demonstration

**Deliverables**:
- Serialized pipeline (`outputs/best_pipeline.pkl`)
- Hyperparameter tuning results
- Feature importance analysis

### Week 7: Model Deployment with FastAPI
**Directory**: `week-07-deployment/`

Deploy ML models as REST APIs using FastAPI.

**Topics Covered**:
- FastAPI framework fundamentals
- Pydantic data validation
- REST API design for ML
- Error handling and logging
- API testing with pytest
- Swagger/OpenAPI documentation

**Key Files**:
- `api/main.py` - FastAPI application
- `tests/test_api.py` - Comprehensive test suite
- `requirements.txt` - Dependencies

**API Endpoints**:
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /docs` - Swagger UI

### Week 8: MLOps Fundamentals
**Directory**: `week-08-mlops/`

Learn MLOps practices including model documentation, monitoring, and responsible AI.

**Topics Covered**:
- Model Cards for transparency
- Structured logging
- Data drift detection (PSI)
- Performance monitoring
- Ethical considerations in ML
- Responsible AI practices

**Key Files**:
- `docs/model_card.md` - Model Card template
- `docs/monitoring_setup.md` - Monitoring guide
- `logs/monitoring.py` - Monitoring utilities

### Milestone Project B: ML Microservice
**Directory**: `milestone-project-b/`

Build a complete, production-ready ML microservice that consolidates all Month 2 concepts.

**Project Components**:
- Trained churn prediction model
- FastAPI REST API with comprehensive endpoints
- Input validation and error handling
- Complete test suite (unit and integration)
- Model Card documentation
- API documentation

**Features**:
- Single and batch prediction endpoints
- Health checking and model info
- Structured logging
- Comprehensive error handling
- Pydantic validation

## Directory Structure

```
02-month-2-ml-mlops/
├── week-05-supervised-ml-1/       # Baseline models
│   ├── data/
│   ├── notebooks/
│   ├── outputs/
│   ├── docs/
│   └── README.md
├── week-06-supervised-ml-2/       # ML pipelines
│   ├── data/
│   ├── notebooks/
│   ├── scripts/
│   ├── outputs/
│   └── README.md
├── week-07-deployment/            # FastAPI
│   ├── api/
│   ├── models/
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
├── week-08-mlops/                 # MLOps
│   ├── docs/
│   ├── logs/
│   └── README.md
├── milestone-project-b/           # Capstone project
│   ├── api/
│   ├── models/
│   ├── notebooks/
│   ├── tests/
│   ├── docs/
│   ├── requirements.txt
│   └── README.md
└── README.md                      # This file
```

## Prerequisites

### Technical Requirements
- Python 3.9 or higher
- pip package manager
- Git (for version control)
- 8GB RAM minimum (16GB recommended)

### Knowledge Prerequisites
- Python programming fundamentals
- Pandas and NumPy basics
- Basic statistics
- Command line familiarity

## Installation

### 1. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install Week-Specific Dependencies

Each week has its own `requirements.txt`. Navigate to the week directory and install:

```bash
cd week-05-supervised-ml-1
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

cd ../week-07-deployment
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import sklearn, pandas, numpy, matplotlib; print('✅ All packages installed')"
```

## Quick Start

### Week 5: Run Baseline Models
```bash
cd week-05-supervised-ml-1
jupyter notebook notebooks/week05_baseline_models.ipynb
```

### Week 6: Run ML Pipeline
```bash
cd week-06-supervised-ml-2
python scripts/week06_ml_pipeline.py
# Or use the notebook:
jupyter notebook notebooks/week06_pipeline.ipynb
```

### Week 7: Start API Server
```bash
cd week-07-deployment
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, test:
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"LotArea": 8450, "OverallQual": 7, ...}'
```

### Week 8: Run Monitoring Example
```bash
cd week-08-mlops
python logs/monitoring.py
```

### Milestone Project B: Full Microservice
```bash
cd milestone-project-b
pip install -r requirements.txt

# Train model
jupyter notebook notebooks/train_model.ipynb

# Start API
uvicorn api.main:app --reload

# Run tests
pytest tests/ -v --cov=api
```

## Skills You'll Gain

### Technical Skills
1. **Machine Learning**
   - Supervised learning algorithms
   - Model evaluation and selection
   - Feature engineering and preprocessing

2. **ML Engineering**
   - Pipeline construction
   - Hyperparameter tuning
   - Model serialization

3. **Software Engineering**
   - API development with FastAPI
   - Test-driven development
   - Documentation

4. **MLOps**
   - Model versioning
   - Monitoring and logging
   - Drift detection
   - Responsible AI practices

### Soft Skills
- Technical documentation writing
- Code review practices
- Project planning and organization

## Assessment Criteria

### Weekly Assignments (60%)
| Week | Deliverable | Weight |
|------|-------------|--------|
| 5 | Completed notebook with analysis | 15% |
| 6 | Pipeline script + tuned model | 15% |
| 7 | Working API + tests | 15% |
| 8 | Model Card + monitoring setup | 15% |

### Milestone Project B (40%)
| Component | Weight |
|-----------|--------|
| Functional API | 10% |
| Test Coverage | 10% |
| Documentation | 10% |
| Code Quality | 10% |

## Resources

### Required Reading
1. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
2. [FastAPI Documentation](https://fastapi.tiangolo.com/)
3. [Model Cards Paper (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993)

### Recommended Reading
- "Hands-On Machine Learning" by Aurélien Géron (Chapters 1-4)
- "Designing Machine Learning Systems" by Chip Huyen
- [Google ML Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)

### Online Courses
- FastAPI Tutorial (official docs)
- pytest Documentation
- MLOps Specialization (Coursera)

## Troubleshooting

### Common Issues

#### ImportError: No module named 'sklearn'
```bash
pip install scikit-learn
```

#### Jupyter notebook kernel not found
```bash
pip install ipykernel
python -m ipykernel install --user --name=venv
```

#### FastAPI port already in use
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
# Or use different port
uvicorn api.main:app --port 8001
```

#### Permission denied when saving models
```bash
chmod -R 755 outputs/
chmod -R 755 models/
```

### Getting Help
1. Check the week's README.md for specific instructions
2. Review error messages carefully
3. Search existing issues on GitHub
4. Ask in the course forum
5. Attend office hours

## Next Steps

After completing Month 2:
- Review all projects and create a portfolio
- Deploy your microservice to a cloud platform
- Add advanced features (authentication, rate limiting)
- Explore advanced MLOps tools (MLflow, Kubeflow)

## License

This material is for educational purposes as part of the RGT-NSS AI Training Program.

## Acknowledgments

- Telco Customer Churn dataset from IBM
- Scikit-learn community
- FastAPI framework by Sebastián Ramírez
- Pydantic for data validation

---

**Version**: 1.0.0  
**Last Updated**: January 2024  
**Maintainers**: RGT-NSS AI Training Team
