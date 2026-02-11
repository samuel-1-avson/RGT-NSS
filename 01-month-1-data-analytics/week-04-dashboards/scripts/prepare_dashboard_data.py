"""
Week 4: Dashboard Data Preparation
"""

import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'week-03-python-analysis', 'data', 'superstore_cleaned.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

def prepare_dashboard_data():
    """Prepare data for dashboard."""
    print("="*60)
    print("WEEK 4: DASHBOARD DATA PREPARATION")
    print("="*60)
    
    df = pd.read_csv(DATA_PATH, parse_dates=['Order Date', 'Ship Date'])
    print(f"\nLoaded {len(df):,} rows")
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # KPIs
    kpis = {
        'Total Sales': df['Sales'].sum(),
        'Total Profit': df['Profit'].sum(),
        'Order Count': df['Order ID'].nunique(),
        'Average Order Value': df['Sales'].sum() / df['Order ID'].nunique(),
        'Profit Margin': (df['Profit'].sum() / df['Sales'].sum()) * 100
    }
    
    print("\n--- KEY METRICS ---")
    for k, v in kpis.items():
        if 'Margin' in k:
            print(f"{k}: {v:.1f}%")
        else:
            print(f"{k}: ${v:,.2f}")
    
    # Monthly summary
    monthly = df.groupby(df['Order Date'].dt.to_period('M')).agg({
        'Sales': 'sum', 'Profit': 'sum', 'Order ID': 'nunique'
    }).reset_index()
    monthly.columns = ['Month', 'Sales', 'Profit', 'Orders']
    monthly['Month'] = monthly['Month'].astype(str)
    monthly.to_csv(os.path.join(OUTPUT_PATH, 'monthly_summary.csv'), index=False)
    print(f"\n[OK] monthly_summary.csv")
    
    # By Category
    category = df.groupby('Category').agg({
        'Sales': 'sum', 'Profit': 'sum', 'Order ID': 'nunique'
    }).reset_index()
    category.to_csv(os.path.join(OUTPUT_PATH, 'by_category.csv'), index=False)
    print("[OK] by_category.csv")
    
    # By Region
    region = df.groupby('Region').agg({
        'Sales': 'sum', 'Profit': 'sum', 'Order ID': 'nunique'
    }).reset_index()
    region.to_csv(os.path.join(OUTPUT_PATH, 'by_region.csv'), index=False)
    print("[OK] by_region.csv")
    
    # Dashboard data
    df[[
        'Order Date', 'Category', 'Sub-Category', 'Region', 'Segment',
        'Sales', 'Profit', 'Quantity', 'Discount'
    ]].to_csv(os.path.join(OUTPUT_PATH, 'dashboard_data.csv'), index=False)
    print("[OK] dashboard_data.csv")
    
    print("\n[OK] Dashboard preparation complete")
    return kpis

if __name__ == "__main__":
    prepare_dashboard_data()
