"""
Week 1: Exploratory Data Analysis (EDA) Script
==============================================

This script performs comprehensive exploratory data analysis on the Telco Customer Churn dataset.
It follows the CRISP-DM methodology for data understanding phase.

Analysis Pipeline:
    1. Load Data - Import the dataset from CSV
    2. Basic Info - Display shape, columns, and data types
    3. Data Quality - Check for missing values, duplicates, anomalies
    4. Descriptive Statistics - Numeric and categorical summaries
    5. Visualizations - Create and save 4 key charts
    6. Summary Report - Generate findings report

Output Files:
    - outputs/churn_distribution.png - Overall churn rate visualization
    - outputs/monthly_charges_by_churn.png - Box plot comparison
    - outputs/tenure_distribution.png - Histogram with KDE
    - outputs/correlation_heatmap.png - Feature correlations
    - outputs/eda_summary_report.txt - Text summary of findings

Author: RGT-NSS Training Program
Week: 1 - Data Literacy, CRISP-DM, Tools Setup
Date: February 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure visualization settings
# Using whitegrid style for professional-looking charts
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Define file paths relative to script location
# Script is in scripts/, data is in ../data/, outputs in ../outputs/
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
OUTPUTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def load_data():
    """
    Load the Telco Customer Churn dataset from CSV.
    
    Returns:
        pandas.DataFrame: The loaded dataset with 7,043 rows and 21 columns
    
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
    """
    print("=" * 60)
    print("Week 1: Exploratory Data Analysis (EDA)")
    print("Dataset: Telco Customer Churn")
    print("=" * 60)
    
    # Load CSV file into pandas DataFrame
    df = pd.read_csv(DATA_PATH)
    print(f"\n[OK] Dataset loaded successfully")
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df


def basic_info(df):
    """
    Display basic information about the dataset.
    
    Shows:
        - Dataset dimensions (rows × columns)
        - Column names with index numbers
        - Data types for each column
    
    Args:
        df (pandas.DataFrame): The dataset to analyze
    
    Returns:
        pandas.DataFrame: The same dataset (unchanged)
    """
    print("\n" + "=" * 60)
    print("Step 1: Basic Information")
    print("=" * 60)
    
    # Display dataset shape
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # List all columns with numbering
    print("\nColumns:")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")
    
    # Display data types
    print("\nData Types:")
    print(df.dtypes.to_string())
    
    return df


def data_quality_check(df):
    """
    Perform data quality assessment.
    
    Checks performed:
        - Missing values count per column
        - Empty strings in TotalCharges (common issue with this dataset)
        - Duplicate row count
    
    Args:
        df (pandas.DataFrame): The dataset to check
    
    Returns:
        pandas.DataFrame: The same dataset (unchanged)
    """
    print("\n" + "=" * 60)
    print("Step 2: Data Quality Check")
    print("=" * 60)
    
    # Check for missing (null) values
    missing = df.isnull().sum()
    print(f"\nMissing Values: {missing.sum()}")
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("    None found")
    
    # Check for empty strings in TotalCharges
    # This is a known issue: some rows have ' ' instead of numeric values
    empty_total = (df['TotalCharges'] == ' ').sum()
    print(f"\nEmpty strings in TotalCharges: {empty_total}")
    if empty_total > 0:
        print("    Note: These correspond to customers with tenure=0 (new customers)")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    return df


def descriptive_stats(df):
    """
    Calculate and display descriptive statistics.
    
    Includes:
        - Numeric columns: count, mean, std, min, quartiles, max
        - Categorical: Churn distribution with counts and percentages
    
    Args:
        df (pandas.DataFrame): The dataset to analyze
    
    Returns:
        pandas.DataFrame: The same dataset (unchanged)
    """
    print("\n" + "=" * 60)
    print("Step 3: Descriptive Statistics")
    print("=" * 60)
    
    # Statistics for numeric columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'SeniorCitizen']
    print("\nNumeric Columns Summary:")
    print(df[numeric_cols].describe().round(2).to_string())
    
    # Churn distribution (target variable)
    print("\nChurn Distribution:")
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    for val in churn_counts.index:
        print(f"    {val:>3}: {churn_counts[val]:>5,} ({churn_pct[val]:>5.1f}%)")
    
    return df


def create_visualizations(df):
    """
    Create and save visualization charts.
    
    Generates 4 charts:
        1. Churn Distribution - Bar chart showing Yes/No counts
        2. Monthly Charges by Churn - Box plot comparing distributions
        3. Tenure Distribution - Histogram with KDE by churn status
        4. Correlation Heatmap - Correlations between numeric features
    
    Args:
        df (pandas.DataFrame): The dataset to visualize
    
    Returns:
        pandas.DataFrame: The same dataset (unchanged)
    """
    print("\n" + "=" * 60)
    print("Step 4: Creating Visualizations")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    
    # Chart 1: Churn Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='Churn', hue='Churn', palette='Set2', legend=False)
    plt.title('Customer Churn Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Churn Status')
    plt.ylabel('Number of Customers')
    
    # Add value labels on bars
    for i, v in enumerate(df['Churn'].value_counts().sort_index()):
        ax.text(i, v + 50, f'{v:,}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, 'churn_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    [OK] churn_distribution.png")
    
    # Chart 2: Monthly Charges by Churn Status
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', hue='Churn', palette='Set2', legend=False)
    plt.title('Monthly Charges by Churn Status', fontsize=14, fontweight='bold')
    plt.xlabel('Churn Status')
    plt.ylabel('Monthly Charges ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, 'monthly_charges_by_churn.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    [OK] monthly_charges_by_churn.png")
    
    # Chart 3: Tenure Distribution by Churn
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True, palette='Set2')
    plt.title('Tenure Distribution by Churn Status', fontsize=14, fontweight='bold')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, 'tenure_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    [OK] tenure_distribution.png")
    
    # Chart 4: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    # Prepare numeric data including converted TotalCharges
    numeric_df = df[['tenure', 'MonthlyCharges', 'SeniorCitizen']].copy()
    numeric_df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    numeric_df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})  # Convert to binary
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Create heatmap with annotations
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    [OK] correlation_heatmap.png")
    
    return df


def generate_summary_report(df):
    """
    Generate a text summary report of EDA findings.
    
    Key Insights:
        - Dataset overview
        - Churn rate analysis
        - Key statistics
        - Recommendations for next steps
    
    Args:
        df (pandas.DataFrame): The analyzed dataset
    
    Returns:
        str: The report text content
    """
    print("\n" + "=" * 60)
    print("Step 5: Generating Summary Report")
    print("=" * 60)
    
    # Build report content
    report_lines = [
        "=" * 60,
        "Week 1: EDA Summary Report",
        "Telco Customer Churn Analysis",
        "=" * 60,
        "",
        f"Dataset: Telco Customer Churn",
        f"Total Records: {len(df):,}",
        f"Total Features: {len(df.columns)}",
        "",
        "-" * 60,
        "CHURN OVERVIEW",
        "-" * 60,
    ]
    
    # Calculate churn rate
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    report_lines.append(f"Overall Churn Rate: {churn_rate:.1f}%")
    report_lines.append(f"Churned Customers: {(df['Churn'] == 'Yes').sum():,}")
    report_lines.append(f"Retained Customers: {(df['Churn'] == 'No').sum():,}")
    
    report_lines.extend([
        "",
        "-" * 60,
        "KEY STATISTICS",
        "-" * 60,
        f"Average Tenure: {df['tenure'].mean():.1f} months",
        f"Average Monthly Charges: ${df['MonthlyCharges'].mean():.2f}",
        f"Average Total Charges: ${pd.to_numeric(df['TotalCharges'], errors='coerce').mean():.2f}",
        "",
        "-" * 60,
        "KEY INSIGHTS",
        "-" * 60,
        "1. Class Imbalance: The dataset has moderate class imbalance",
        f"   with {churn_rate:.1f}% churn rate (26.5% is typical for telecom).",
        "",
        "2. Monthly Charges Impact: Churned customers have higher",
        "   average monthly charges, suggesting price sensitivity.",
        "",
        "3. Tenure Factor: Customers with lower tenure (new customers)",
        "   show higher churn rates. Early engagement is critical.",
        "",
        "4. Data Quality: TotalCharges has 11 empty strings",
        "   corresponding to new customers (tenure=0 months).",
        "",
        "-" * 60,
        "NEXT STEPS",
        "-" * 60,
        "1. Clean TotalCharges: Convert to numeric, handle empty values",
        "2. Feature Engineering: Create tenure groups, encode categoricals",
        "3. Baseline Model: Start with Logistic Regression",
        "4. Advanced Analysis: Explore categorical feature relationships",
        "",
        "=" * 60,
    ])
    
    report_text = "\n".join(report_lines)
    
    # Save to file
    report_path = os.path.join(OUTPUTS_PATH, 'eda_summary_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n    [OK] Report saved: eda_summary_report.txt")
    
    return report_text


def main():
    """
    Main execution pipeline for Week 1 EDA.
    
    Runs all analysis steps in sequence:
        1. Load data
        2. Display basic info
        3. Check data quality
        4. Show descriptive statistics
        5. Create visualizations
        6. Generate summary report
    """
    # Step 1: Load the dataset
    df = load_data()
    
    # Step 2: Display basic information
    df = basic_info(df)
    
    # Step 3: Data quality assessment
    df = data_quality_check(df)
    
    # Step 4: Descriptive statistics
    df = descriptive_stats(df)
    
    # Step 5: Create visualizations
    df = create_visualizations(df)
    
    # Step 6: Generate report
    generate_summary_report(df)
    
    print("\n" + "=" * 60)
    print("Week 1 EDA Complete!")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUTS_PATH}")
    print("\nGenerated files:")
    print("    - churn_distribution.png")
    print("    - monthly_charges_by_churn.png")
    print("    - tenure_distribution.png")
    print("    - correlation_heatmap.png")
    print("    - eda_summary_report.txt")


if __name__ == "__main__":
    # Execute main pipeline when script is run directly
    main()
