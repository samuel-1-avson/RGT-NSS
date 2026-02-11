"""
Data Cleaning Pipeline for Superstore Dataset

This module provides reusable functions for cleaning and transforming
the Superstore sales dataset.
"""

import pandas as pd
import numpy as np


def load_data(filepath):
    """
    Load raw data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Raw data
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows from {filepath}")
    return df


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with duplicates removed
    """
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"Removed {removed} duplicate rows ({removed/initial_rows*100:.2f}%)")
    return df


def handle_missing_values(df):
    """
    Handle missing values appropriately.
    
    - Postal Code: Fill with 0 and convert to int
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    # Handle Postal Code missing values
    if 'Postal Code' in df.columns:
        missing_postal = df['Postal Code'].isnull().sum()
        df['Postal Code'] = df['Postal Code'].fillna(0).astype(int)
        print(f"Filled {missing_postal} missing Postal Codes with 0")
    
    return df


def fix_data_types(df):
    """
    Convert columns to appropriate data types.
    
    - Order Date: datetime
    - Ship Date: datetime
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with corrected data types
    """
    # Convert date columns
    date_columns = ['Order Date', 'Ship Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    print(f"Converted {len(date_columns)} date columns to datetime")
    return df


def create_features(df):
    """
    Create new features from existing columns.
    
    New features:
    - Shipping Days: Days between order and ship dates
    - Profit Margin: Profit as percentage of Sales
    - Order Year: Year extracted from Order Date
    - Order Month: Month extracted from Order Date
    - Discount Category: Binned discount levels
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    # Shipping time
    if 'Order Date' in df.columns and 'Ship Date' in df.columns:
        df['Shipping Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    # Profit margin
    if 'Profit' in df.columns and 'Sales' in df.columns:
        df['Profit Margin'] = df['Profit'] / df['Sales']
        df['Profit Margin'] = df['Profit Margin'].replace([np.inf, -np.inf], 0)
    
    # Time features
    if 'Order Date' in df.columns:
        df['Order Year'] = df['Order Date'].dt.year
        df['Order Month'] = df['Order Date'].dt.month
        df['Order Quarter'] = df['Order Date'].dt.quarter
    
    # Discount category
    if 'Discount' in df.columns:
        df['Discount Category'] = pd.cut(
            df['Discount'],
            bins=[-0.01, 0, 0.2, 0.5, 1.0],
            labels=['No Discount', 'Low', 'Medium', 'High']
        )
    
    # Sales per unit
    if 'Sales' in df.columns and 'Quantity' in df.columns:
        df['Unit Price'] = df['Sales'] / df['Quantity']
    
    print(f"Created {5} new features")
    return df


def validate_data(df):
    """
    Perform data validation checks.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Validation results
    """
    results = {
        'total_rows': len(df),
        'duplicates': df.duplicated().sum(),
        'missing_values': df.isnull().sum().sum(),
        'negative_sales': (df['Sales'] < 0).sum() if 'Sales' in df.columns else 0,
        'negative_profit': (df['Profit'] < 0).sum() if 'Profit' in df.columns else 0,
    }
    
    # Check for invalid shipping days
    if 'Shipping Days' in df.columns:
        results['negative_shipping_days'] = (df['Shipping Days'] < 0).sum()
    
    print("\nValidation Results:")
    print("-" * 40)
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    return results


def clean_data(filepath, output_path=None):
    """
    Complete data cleaning pipeline.
    
    Args:
        filepath (str): Path to raw data CSV
        output_path (str, optional): Path to save cleaned data
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("=" * 50)
    print("DATA CLEANING PIPELINE")
    print("=" * 50)
    
    # Load data
    df = load_data(filepath)
    
    # Cleaning steps
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = fix_data_types(df)
    df = create_features(df)
    
    # Validate
    validation = validate_data(df)
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n[OK] Cleaned data saved to: {output_path}")
    
    print(f"\n[OK] Pipeline complete: {len(df):,} rows processed")
    return df


if __name__ == "__main__":
    # Example usage
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, '..', 'data', 'superstore.csv')
    output_path = os.path.join(script_dir, '..', 'data', 'superstore_cleaned.csv')
    df_clean = clean_data(input_path, output_path)
