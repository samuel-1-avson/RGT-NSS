"""
Week 1: Data Download Script
===========================

This script downloads the Telco Customer Churn dataset from IBM's open data repository.
The dataset is used for exploratory data analysis (EDA) in Week 1 of the RGT-NSS training program.

Dataset Information:
    - Name: Telco Customer Churn
    - Source: IBM Sample Data Sets (via GitHub)
    - Original Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    - Records: 7,043 customers
    - Features: 21 columns including demographics, services, and churn status

Target Variable:
    - Churn: Whether the customer left the service (Yes/No)

Author: RGT-NSS Training Program
Week: 1 - Data Literacy, CRISP-DM, Tools Setup
"""

import pandas as pd
import os

# Direct download URL from IBM's GitHub repository
# This is a public dataset used for educational purposes
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def download_dataset():
    """
    Download the Telco Customer Churn dataset from IBM's repository.
    
    This function:
        1. Downloads the CSV file from the public URL
        2. Saves it to the local data/ directory
        3. Prints summary statistics about the dataset
    
    Returns:
        pandas.DataFrame: The downloaded dataset, or None if download fails
    
    Raises:
        Exception: If network error or file system error occurs
    
    Example:
        >>> df = download_dataset()
        >>> print(f"Downloaded {len(df)} records")
    """
    print("=" * 60)
    print("Week 1: Downloading Telco Customer Churn Dataset")
    print("=" * 60)
    
    try:
        # Download the dataset from IBM's GitHub repository
        print(f"\nFetching data from: {DATASET_URL}")
        df = pd.read_csv(DATASET_URL)
        
        # Construct output path relative to script location
        # ../data/ goes up one level from scripts/ then into data/
        output_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'data', 
            'WA_Fn-UseC_-Telco-Customer-Churn.csv'
        )
        
        # Ensure data directory exists before saving
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV file (no index column)
        df.to_csv(output_path, index=False)
        
        # Print success message with dataset summary
        print("\n[OK] Dataset downloaded successfully!")
        print(f"    Location: {output_path}")
        print(f"    Records: {len(df):,} rows")
        print(f"    Features: {len(df.columns)} columns")
        print(f"\nColumn Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"    {i:2d}. {col}")
        
        return df
        
    except Exception as e:
        # Handle download errors gracefully
        print(f"\n[ERROR] Failed to download dataset: {e}")
        print("\nTroubleshooting:")
        print("    1. Check internet connection")
        print("    2. Verify the URL is accessible")
        print("    3. Download manually from:")
        print("       https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        return None


if __name__ == "__main__":
    # Execute download when script is run directly
    download_dataset()
