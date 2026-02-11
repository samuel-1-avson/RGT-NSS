"""
Download Telco Customer Churn Dataset
"""

import pandas as pd
import os

DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def download_dataset():
    """Download the Telco Customer Churn dataset."""
    print("Downloading Telco Customer Churn dataset...")
    
    try:
        df = pd.read_csv(DATASET_URL)
        
        output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df.to_csv(output_path, index=False)
        
        print(f"Dataset downloaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    download_dataset()
