"""
Week 1 EDA Analysis Script

This script performs complete exploratory data analysis on the Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
OUTPUTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs')

def load_data():
    """Load the dataset."""
    print("="*60)
    print("WEEK 1: EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

def basic_info(df):
    """Generate basic dataset information."""
    print("\n" + "="*60)
    print("1. BASIC INFORMATION")
    print("="*60)
    
    print(f"\nShape: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes)
    
    return df

def data_quality_check(df):
    """Check data quality."""
    print("\n" + "="*60)
    print("2. DATA QUALITY CHECK")
    print("="*60)
    
    # Missing values
    missing = df.isnull().sum()
    print(f"\nMissing Values:\n{missing[missing > 0]}")
    
    # Empty strings in TotalCharges
    empty_total = (df['TotalCharges'] == ' ').sum()
    print(f"\nEmpty strings in TotalCharges: {empty_total}")
    
    # Duplicates
    dups = df.duplicated().sum()
    print(f"Duplicate rows: {dups}")
    
    return df

def descriptive_stats(df):
    """Generate descriptive statistics."""
    print("\n" + "="*60)
    print("3. DESCRIPTIVE STATISTICS")
    print("="*60)
    
    # Numeric columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'SeniorCitizen']
    print("\nNumeric Columns:")
    print(df[numeric_cols].describe().round(2))
    
    # Churn distribution
    print("\nChurn Distribution:")
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    for val in churn_counts.index:
        print(f"  {val}: {churn_counts[val]:,} ({churn_pct[val]:.1f}%)")
    
    return df

def create_visualizations(df):
    """Create and save visualizations."""
    print("\n" + "="*60)
    print("4. CREATING VISUALIZATIONS")
    print("="*60)
    
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    
    # 1. Churn Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='Churn', palette='Set2')
    plt.title('Customer Churn Distribution', fontsize=14)
    plt.xlabel('Churn')
    plt.ylabel('Count')
    
    # Add value labels
    for i, v in enumerate(df['Churn'].value_counts()):
        ax.text(i, v + 50, str(v), ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, 'churn_distribution.png'), dpi=300)
    plt.close()
    print("  [OK] Saved: churn_distribution.png")
    
    # 2. Monthly Charges by Churn
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='Set2')
    plt.title('Monthly Charges by Churn Status', fontsize=14)
    plt.ylabel('Monthly Charges ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, 'monthly_charges_by_churn.png'), dpi=300)
    plt.close()
    print("  [OK] Saved: monthly_charges_by_churn.png")
    
    # 3. Tenure Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True, palette='Set2')
    plt.title('Tenure Distribution by Churn', fontsize=14)
    plt.xlabel('Tenure (months)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, 'tenure_distribution.png'), dpi=300)
    plt.close()
    print("  [OK] Saved: tenure_distribution.png")
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df[['tenure', 'MonthlyCharges', 'SeniorCitizen']].copy()
    numeric_df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    numeric_df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
    plt.title('Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    print("  [OK] Saved: correlation_heatmap.png")
    
    return df

def generate_summary_report(df):
    """Generate summary report."""
    print("\n" + "="*60)
    print("5. GENERATING SUMMARY REPORT")
    print("="*60)
    
    report = []
    report.append("="*60)
    report.append("WEEK 1 EDA SUMMARY REPORT")
    report.append("="*60)
    report.append(f"\nDataset: Telco Customer Churn")
    report.append(f"Records: {len(df):,}")
    report.append(f"Features: {len(df.columns)}")
    
    report.append("\n--- CHURN OVERVIEW ---")
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    report.append(f"Overall Churn Rate: {churn_rate:.1f}%")
    
    report.append("\n--- KEY STATISTICS ---")
    report.append(f"Average Tenure: {df['tenure'].mean():.1f} months")
    report.append(f"Average Monthly Charges: ${df['MonthlyCharges'].mean():.2f}")
    
    report.append("\n--- INSIGHTS ---")
    report.append("1. Dataset has moderate class imbalance (26.5% churn)")
    report.append("2. Churned customers have higher average monthly charges")
    report.append("3. Tenure shows negative correlation with churn")
    report.append("4. TotalCharges has 11 empty values (tenure=0)")
    
    report.append("\n--- NEXT STEPS ---")
    report.append("1. Clean TotalCharges (convert to numeric)")
    report.append("2. Encode categorical variables")
    report.append("3. Feature engineering (tenure groups)")
    report.append("4. Build baseline model")
    
    report.append("\n" + "="*60)
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = os.path.join(OUTPUTS_PATH, 'eda_summary_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n  [OK] Saved: eda_summary_report.txt")
    
    return report_text

def main():
    """Main EDA pipeline."""
    # Load data
    df = load_data()
    
    # Run analysis
    df = basic_info(df)
    df = data_quality_check(df)
    df = descriptive_stats(df)
    df = create_visualizations(df)
    
    # Generate report
    generate_summary_report(df)
    
    print("\n" + "="*60)
    print("EDA COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUTS_PATH}")

if __name__ == "__main__":
    main()
