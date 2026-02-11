"""
Download Superstore Dataset
"""

import pandas as pd
import os

# Sample Superstore data URL (subset for training)
# Full dataset available at: https://www.kaggle.com/datasets/vivek468/superstore-sales-dataset
DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/Superstore.csv"

def download_dataset():
    """Download the Superstore dataset."""
    print("Downloading Superstore dataset...")
    
    try:
        df = pd.read_csv(DATASET_URL)
        
        output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'superstore.csv')
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
