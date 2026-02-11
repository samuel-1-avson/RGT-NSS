"""
Shared utility functions for machine learning tasks.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
import json
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_classification(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """Evaluate classification model with multiple metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            pass
    
    return metrics


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    """Evaluate regression model with multiple metrics."""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2_score': r2_score(y_true, y_pred),
    }


def save_model(model, filepath: str, metadata: Dict = None):
    """Save model with metadata."""
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")
    
    if metadata:
        meta_path = filepath.replace('.pkl', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


def load_model(filepath: str):
    """Load saved model."""
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model
