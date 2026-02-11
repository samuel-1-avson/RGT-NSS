# MLOps Monitoring Setup Guide

## Overview

This guide covers setting up monitoring for ML models in production. Monitoring ensures your models remain performant, reliable, and aligned with business objectives.

## Why Monitor ML Models?

```
┌─────────────────────────────────────────────────────────────┐
│                  ML Model Lifecycle                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Training → Validation → Deployment → Monitoring → Retrain │
│                               ↑                             │
│                         ┌─────┴─────┐                       │
│                         │  Monitor  │                       │
│                         │ - Drift   │                       │
│                         │ - Metrics │                       │
│                         │ - Latency │                       │
│                         └───────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Monitoring Dimensions

| Dimension | What to Monitor | Why It Matters |
|-----------|-----------------|----------------|
| **Data Quality** | Missing values, feature distributions | Garbage in, garbage out |
| **Model Performance** | Accuracy, precision, recall | Business impact |
| **System Health** | Latency, throughput, errors | User experience |
| **Concept Drift** | Target distribution changes | Model relevance |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Monitoring Stack                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │ Application │    │ Application │    │ Application │        │
│   │  (FastAPI)  │    │  (FastAPI)  │    │  (FastAPI)  │        │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│          │                  │                  │                │
│          └──────────────────┼──────────────────┘                │
│                             │                                   │
│                             ▼                                   │
│   ┌───────────────────────────────────────────────────────┐    │
│   │              Metrics Collection                        │    │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│   │  │  Logs    │  │ Metrics  │  │  Traces  │            │    │
│   │  │(Structured│  │(Prometheus│  │  (OpenTelemetry)    │    │
│   │  └────┬─────┘  └────┬─────┘  └────┬─────┘            │    │
│   └───────┼─────────────┼─────────────┼──────────────────┘    │
│           │             │             │                         │
│           ▼             ▼             ▼                         │
│   ┌───────────────────────────────────────────────────────┐    │
│   │              Visualization Layer                       │    │
│   │  ┌───────────────────────────────────────────────┐   │    │
│   │  │           Dashboard (Grafana)                  │   │    │
│   │  │  - Real-time metrics                          │   │    │
│   │  │  - Alert rules                                │   │    │
│   │  │  - Historical trends                          │   │    │
│   │  └───────────────────────────────────────────────┘   │    │
│   └───────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│   ┌───────────────────────────────────────────────────────┐    │
│   │              Alerting                                  │    │
│   │  Email → Slack → PagerDuty                            │    │
│   └───────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### 1. Structured Logging

Create a logger that outputs JSON for easy parsing:

```python
# logging_config.py
import logging
import json
from datetime import datetime
from pythonjsonlogger import jsonlogger

def setup_logging():
    """Configure structured JSON logging."""
    logHandler = logging.StreamHandler()
    
    formatter = jsonlogger.JsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s',
        rename_fields={'levelname': 'level', 'asctime': 'timestamp'}
    )
    
    logHandler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(logHandler)
    logger.setLevel(logging.INFO)
    
    return logger
```

### 2. Custom Metrics

Track ML-specific metrics:

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total predictions',
    ['model_version', 'status']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_duration_seconds',
    'Prediction latency',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PREDICTION_SCORE = Histogram(
    'model_prediction_score',
    'Distribution of prediction scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

FEATURE_DRIFT = Gauge(
    'model_feature_drift_psi',
    'Population Stability Index for features',
    ['feature_name']
)

class MetricsMiddleware:
    """FastAPI middleware for collecting metrics."""
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        # Record latency
        duration = time.time() - start_time
        PREDICTION_LATENCY.observe(duration)
        
        # Record prediction status
        status = 'success' if response.status_code == 200 else 'error'
        PREDICTION_COUNTER.labels(
            model_version='1.0.0',
            status=status
        ).inc()
        
        return response
```

### 3. Data Drift Detection

Monitor for changes in input distributions:

```python
# drift_detection.py
import numpy as np
from scipy import stats

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI).
    
    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change
    PSI > 0.2: Significant change (investigate)
    """
    def scale_range(input, min_val, max_val):
        return (input - min_val) / (max_val - min_val)
    
    # Create bins
    breakpoints = np.linspace(0, 1, buckets + 1)
    breakpoints[0] = -0.001  # Adjust for edge cases
    breakpoints[-1] = 1.001
    
    # Scale data to 0-1 range
    expected_scaled = scale_range(expected, expected.min(), expected.max())
    actual_scaled = scale_range(actual, actual.min(), actual.max())
    
    # Calculate proportions
    expected_percents = np.histogram(expected_scaled, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual_scaled, breakpoints)[0] / len(actual)
    
    # Calculate PSI
    psi = np.sum((actual_percents - expected_percents) 
                 * np.log(actual_percents / expected_percents + 1e-10))
    
    return psi

def detect_drift(reference_data, current_data, threshold=0.2):
    """
    Detect drift across all features.
    
    Returns:
        dict: Drift status for each feature
    """
    drift_results = {}
    
    for feature in reference_data.columns:
        if feature in current_data.columns:
            psi = calculate_psi(
                reference_data[feature].values,
                current_data[feature].values
            )
            
            drift_results[feature] = {
                'psi': psi,
                'drift_detected': psi > threshold,
                'severity': 'high' if psi > 0.3 else 'medium' if psi > 0.1 else 'low'
            }
    
    return drift_results
```

### 4. Alerting Rules

Set up alerts for critical conditions:

```yaml
# alerts.yml
groups:
  - name: ml_model_alerts
    rules:
      # High latency alert
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.99, model_prediction_duration_seconds_bucket) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency detected"
          description: "p99 latency is above 500ms for 5 minutes"
      
      # Model accuracy drop
      - alert: ModelAccuracyDrop
        expr: model_accuracy < 0.7
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy has dropped significantly"
          description: "Current accuracy is below 70%"
      
      # Feature drift
      - alert: FeatureDriftDetected
        expr: model_feature_drift_psi > 0.2
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Feature drift detected"
          description: "PSI for {{ $labels.feature_name }} is {{ $value }}"
      
      # High error rate
      - alert: HighPredictionErrorRate
        expr: rate(model_predictions_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High prediction error rate"
          description: "Error rate is above 10%"
```

## Setting Up with Docker Compose

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  prometheus_data:
  grafana_data:
```

## Dashboard Setup

### Key Metrics to Display

1. **Overview Panel**
   - Total predictions (last 24h)
   - Average latency
   - Error rate
   - Model version

2. **Performance Panel**
   - Accuracy over time
   - Precision/Recall trends
   - ROC-AUC tracking

3. **Drift Panel**
   - PSI scores by feature
   - Input distribution comparisons
   - Prediction score distribution

4. **System Panel**
   - Request rate
   - Response time percentiles
   - Resource utilization

### Example Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "ML Model Monitoring",
    "panels": [
      {
        "title": "Prediction Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(model_predictions_total[5m])"
          }
        ]
      },
      {
        "title": "Latency (p99)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, model_prediction_duration_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

## Best Practices

### 1. Log Everything
```python
logger.info("Prediction made", extra={
    "customer_id": customer_id,
    "prediction": prediction,
    "confidence": confidence,
    "model_version": "1.0.0",
    "latency_ms": latency
})
```

### 2. Use Correlation IDs
```python
import uuid

correlation_id = str(uuid.uuid4())
# Pass through entire request lifecycle
```

### 3. Monitor Business Metrics
- Prediction volume
- Revenue impact
- User engagement
- Cost per prediction

### 4. Establish Baselines
- Document normal operating ranges
- Set alert thresholds based on historical data
- Review and adjust quarterly

### 5. Alert Fatigue Prevention
- Use severity levels
- Implement alert grouping
- Set up escalation policies

## Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| High latency | Model too large | Optimize model or scale horizontally |
| Accuracy drop | Data drift | Retrain model with recent data |
| Memory errors | Memory leak | Restart containers, profile memory |
| False alerts | Wrong thresholds | Adjust based on historical data |

## Next Steps

1. Set up basic logging in your application
2. Deploy Prometheus and Grafana
3. Create custom dashboards
4. Configure alerting rules
5. Document runbooks for common issues

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/)
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [ML Monitoring: A Comprehensive Guide](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)
