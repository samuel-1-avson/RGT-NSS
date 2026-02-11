# Week 8: MLOps Fundamentals

## Overview
This week introduces MLOps (Machine Learning Operations) - the practice of deploying and maintaining ML models in production reliably and efficiently. You'll learn about model documentation, monitoring, and responsible AI practices.

## Learning Objectives
By the end of this week, you will:
- Create comprehensive Model Cards for transparency
- Set up basic monitoring and logging
- Understand ML system reliability
- Recognize ethical considerations in ML
- Implement responsible AI practices

## Files
```
week-08-mlops/
├── docs/
│   ├── model_card.md          # Model Card template
│   └── monitoring_setup.md    # Monitoring guide
├── logs/
│   ├── monitoring.py          # Monitoring utilities
│   └── .gitkeep
└── README.md
```

## What is MLOps?

MLOps is the intersection of:
```
        ┌─────────────────┐
        │   ML/DATA SCIENCE │
        └────────┬────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌───────┐   ┌─────────┐   ┌─────────┐
│ DevOps │ ◄─┤  MLOps  ├─► │ Software│
│        │   │         │   │Engineering
└───────┘   └─────────┘   └─────────┘
```

## Model Cards

### What is a Model Card?
A Model Card is a documentation framework proposed by Google for transparent model reporting. It includes:
- Model description and intended use
- Training data and methodology
- Performance metrics and limitations
- Ethical considerations

### Why Model Cards Matter
- **Transparency**: Stakeholders understand model capabilities
- **Accountability**: Document decisions and trade-offs
- **Reproducibility**: Enable others to replicate results
- **Compliance**: Meet regulatory requirements

## Monitoring in Production

### Key Metrics to Track

| Category | Metrics | Purpose |
|----------|---------|---------|
| **System** | Latency, throughput, errors | Infrastructure health |
| **Data** | Distribution drift, missing values | Data quality |
| **Model** | Accuracy decay, prediction distribution | Model health |
| **Business** | Prediction confidence, user feedback | Business impact |

### Monitoring Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Monitoring System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Logging   │    │   Metrics   │    │   Alerting  │     │
│  │             │    │             │    │             │     │
│  │ - Requests  │    │ - Latency   │    │ - Threshold │     │
│  │ - Responses │    │ - Accuracy  │    │ - Anomaly   │     │
│  │ - Errors    │    │ - Drift     │    │ - PagerDuty │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Dashboard (Grafana/CloudWatch)          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Responsible AI

### Key Principles

1. **Fairness**
   - Models should not discriminate based on protected attributes
   - Test for disparate impact across demographic groups
   - Document known biases

2. **Transparency**
   - Explainable predictions where possible
   - Document model limitations
   - Clear communication of confidence levels

3. **Privacy**
   - Protect sensitive training data
   - Secure inference endpoints
   - Data minimization in production

4. **Reliability**
   - Graceful degradation on errors
   - Fallback mechanisms
   - Regular retraining schedules

### Bias Detection Checklist

- [ ] Test model performance across demographic groups
- [ ] Analyze feature importance for proxy variables
- [ ] Review training data for historical biases
- [ ] Document fairness metrics in Model Card
- [ ] Establish bias monitoring in production

## Setup

### Install Dependencies
```bash
pip install prometheus-client psutil logging-json
```

## Running Monitoring Examples

```bash
python logs/monitoring.py
```

This will demonstrate:
- Structured logging
- Performance metrics collection
- Health check implementation

## Model Card Template

See `docs/model_card.md` for a complete template including:
- Model Details
- Intended Use
- Factors
- Metrics
- Training Data
- Evaluation Data
- Ethical Considerations
- Caveats and Recommendations

## Monitoring Architecture

```
                    ┌─────────────────┐
                    │  Application    │
                    │   (FastAPI)     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Logs    │  │ Metrics  │  │ Traces   │
        │ (File)   │  │(Prometheus│  │ (Jaeger) │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    ┌──────────────┐
                    │  Dashboard   │
                    │  (Grafana)   │
                    └──────────────┘
```

## Best Practices

### 1. Logging
- Use structured logging (JSON format)
- Include correlation IDs for request tracking
- Log both inputs and predictions (with privacy safeguards)
- Set appropriate log levels

### 2. Model Registry
- Version all models
- Track training parameters and metrics
- Maintain model lineage
- Enable model rollback

### 3. CI/CD for ML
```
Code Commit → Tests → Build → Train → Evaluate → Deploy
                ↓       ↓       ↓        ↓         ↓
            Unit   Docker  Data   Model  A/B    Canary
            Tests  Image   Val.   Perf.  Test   Deploy
```

### 4. A/B Testing
- Gradually roll out new models
- Compare business metrics
- Monitor for degradation
- Enable instant rollback

## Tools Overview

| Category | Open Source | Cloud |
|----------|-------------|-------|
| **Experiment Tracking** | MLflow, DVC | AWS SageMaker, Vertex AI |
| **Model Registry** | MLflow, Feast | AWS Model Registry |
| **Monitoring** | Prometheus, Grafana | Datadog, New Relic |
| **Feature Store** | Feast, Tecton | SageMaker Feature Store |
| **Workflow** | Kubeflow, Airflow | AWS Step Functions |

## Exercises

1. **Create a Model Card** for your Week 7 API model
2. **Set up Prometheus metrics** in your FastAPI application
3. **Implement drift detection** for input features
4. **Design an A/B test** framework for model comparison
5. **Document ethical considerations** for your use case

## Case Studies

### Case 1: Credit Scoring
- **Challenge**: Ensure fairness across demographic groups
- **Solution**: Separate model evaluation by protected attributes
- **Lesson**: Document fairness metrics explicitly

### Case 2: Healthcare Prediction
- **Challenge**: HIPAA compliance and patient privacy
- **Solution**: Differential privacy during training
- **Lesson**: Privacy must be designed in from start

### Case 3: Recommendation System
- **Challenge**: Filter bubbles and echo chambers
- **Solution**: Diversity metrics in evaluation
- **Lesson**: Consider broader societal impact

## Next Steps

After completing Week 8, you'll move to Milestone Project B where you'll build a complete ML microservice incorporating all MLOps principles.

---

## Assignment

1. **Complete Model Card** (`docs/model_card.md`) for your deployed model:
   - Fill in all sections honestly
   - Include performance metrics across subgroups if applicable
   - Document known limitations

2. **Implement Monitoring** (`logs/monitoring.py`):
   - Add logging to your Week 7 API
   - Create at least 3 custom metrics
   - Set up alerting thresholds

3. **Ethical Analysis**:
   - Identify potential biases in your model
   - Propose mitigation strategies
   - Document in Model Card

4. **Presentation**:
   - Present your Model Card to the class
   - Discuss ethical considerations
   - Answer questions about limitations
