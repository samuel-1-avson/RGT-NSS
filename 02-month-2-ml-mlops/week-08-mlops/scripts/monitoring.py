"""
Week 8: Model Monitoring Setup

Tracks model performance and data drift.
"""

import json
import logging
import os
from datetime import datetime
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'predictions.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model predictions and detect drift."""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.count = 0
        self.churn_count = 0
    
    def log(self, customer_id: str, prediction: int, prob: float):
        """Log a prediction."""
        self.count += 1
        self.churn_count += int(prediction)
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'customer_id': customer_id,
            'prediction': int(prediction),
            'probability': float(prob)
        }
        logger.info(f"Prediction: {json.dumps(entry)}")
    
    def check_drift(self, reference: float = 0.265) -> dict:
        """Check for prediction drift."""
        if self.count < 100:
            return {"status": "insufficient_data"}
        
        current = self.churn_count / self.count
        drift = abs(current - reference)
        
        return {
            'reference_rate': reference,
            'current_rate': round(current, 4),
            'drift': round(drift, 4),
            'alert': drift > self.threshold
        }
    
    def report(self):
        """Print monitoring report."""
        print("\nModel Monitoring Report")
        print("-" * 40)
        print(f"Total predictions: {self.count}")
        print(f"Churn predictions: {self.churn_count}")
        if self.count > 0:
            print(f"Churn rate: {self.churn_count/self.count:.2%}")


def main():
    """Demo monitoring."""
    print("Week 8: Model Monitoring")
    print("=" * 40)
    
    monitor = ModelMonitor()
    
    # Simulate predictions
    np.random.seed(42)
    for i in range(100):
        pred = np.random.choice([0, 1], p=[0.75, 0.25])
        prob = np.random.uniform(0, 1)
        monitor.log(f"CUST_{i:04d}", pred, prob)
    
    monitor.report()
    drift = monitor.check_drift()
    print(f"\nDrift check: {drift}")


if __name__ == "__main__":
    main()
