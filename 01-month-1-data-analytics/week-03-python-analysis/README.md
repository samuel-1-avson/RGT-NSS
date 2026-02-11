# Week 3: Python for Data Analysis

> **Branch**: `week-03-python-analysis` | **Review Required**: Yes  
> **Dataset**: [Kaggle - Superstore Sales](https://www.kaggle.com/datasets/vivek468/superstore-sales-dataset)

---

## Git Workflow
```bash
git checkout main
git pull origin main
git checkout -b week-03-python-analysis
# Work and commit regularly
git push origin week-03-python-analysis
```

---

## Learning Objectives
- Master pandas for data manipulation
- Create reusable data cleaning pipelines
- Use numpy for numerical operations
- Visualize data with matplotlib/seaborn
- Write unit tests for data functions

---

## Dataset

**Name**: Superstore Sales  
**Source**: [Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-sales-dataset)  
**Size**: 9,994 rows × 21 columns  
**Description**: Retail sales data with orders, customers, products, and regions

---

## Weekly Structure

### Prep (≤60 min)
- [ ] Review pandas user guide
- [ ] Read Matplotlib tutorial basics

### Guided Lab (≤120 min)

#### Lab 3.1: Data Loading and Initial Exploration
```python
# notebooks/01_exploration.ipynb
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('../data/superstore.csv')

# Basic info
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicated Rows: {df.duplicated().sum()}")
```

#### Lab 3.2: Data Cleaning Pipeline
```python
# scripts/data_cleaning.py
"""
Data cleaning pipeline for Superstore dataset.
"""
import pandas as pd
import numpy as np

def load_data(filepath):
    """Load raw data from CSV."""
    return pd.read_csv(filepath)

def remove_duplicates(df):
    """Remove duplicate rows."""
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows")
    return df

def handle_missing_values(df):
    """Handle missing values appropriately."""
    # Postal Code has some missing values
    df['Postal Code'] = df['Postal Code'].fillna(0).astype(int)
    return df

def fix_data_types(df):
    """Convert columns to appropriate data types."""
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    return df

def create_features(df):
    """Create new features from existing columns."""
    # Shipping time
    df['Shipping Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    # Profit margin
    df['Profit Margin'] = df['Profit'] / df['Sales']
    df['Profit Margin'] = df['Profit Margin'].replace([np.inf, -np.inf], 0)
    
    # Year and Month
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    
    # Discount category
    df['Discount Category'] = pd.cut(
        df['Discount'],
        bins=[-0.01, 0, 0.2, 0.5, 1.0],
        labels=['No Discount', 'Low', 'Medium', 'High']
    )
    
    return df

def clean_data(filepath):
    """Complete cleaning pipeline."""
    df = load_data(filepath)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = fix_data_types(df)
    df = create_features(df)
    return df

if __name__ == "__main__":
    df_clean = clean_data('../data/superstore.csv')
    df_clean.to_csv('../data/superstore_cleaned.csv', index=False)
    print(f"Cleaned data saved: {len(df_clean)} rows")
```

#### Lab 3.3: Data Analysis
```python
# notebooks/02_analysis.ipynb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/superstore_cleaned.csv', parse_dates=['Order Date', 'Ship Date'])

# Analysis 1: Sales by Category
category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
print("Sales by Category:")
print(category_sales)

# Analysis 2: Profit by Region
region_profit = df.groupby('Region')['Profit'].sum().sort_values(ascending=False)
print("\nProfit by Region:")
print(region_profit)

# Analysis 3: Monthly sales trend
monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
print("\nMonthly Sales Trend:")
print(monthly_sales)

# Visualization 1: Sales by Category
plt.figure(figsize=(10, 6))
category_sales.plot(kind='bar', color='steelblue')
plt.title('Total Sales by Category')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../outputs/sales_by_category.png', dpi=300)
plt.show()

# Visualization 2: Monthly Sales Trend
plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend')
plt.ylabel('Sales ($)')
plt.xlabel('Month')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/monthly_sales_trend.png', dpi=300)
plt.show()

# Visualization 3: Profit vs Sales Scatter
plt.figure(figsize=(10, 6))
plt.scatter(df['Sales'], df['Profit'], alpha=0.5, c='steelblue')
plt.xlabel('Sales ($)')
plt.ylabel('Profit ($)')
plt.title('Profit vs Sales')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('../outputs/profit_vs_sales.png', dpi=300)
plt.show()
```

### Independent Work (≤120 min)

#### Task 1: Complete Pipeline
- [ ] Add more data quality checks
- [ ] Create additional features
- [ ] Document all functions

#### Task 2: Write Unit Tests
```python
# tests/test_cleaning.py
import unittest
import pandas as pd
import sys
sys.path.append('../scripts')
from data_cleaning import remove_duplicates, handle_missing_values, create_features

class TestDataCleaning(unittest.TestCase):
    
    def test_remove_duplicates(self):
        df = pd.DataFrame({'A': [1, 1, 2], 'B': [3, 3, 4]})
        result = remove_duplicates(df)
        self.assertEqual(len(result), 2)
    
    def test_handle_missing_values(self):
        df = pd.DataFrame({'Postal Code': [10001, None, 10002]})
        result = handle_missing_values(df)
        self.assertEqual(result['Postal Code'].isnull().sum(), 0)
    
    def test_create_features(self):
        df = pd.DataFrame({
            'Order Date': pd.to_datetime(['2023-01-01', '2023-01-15']),
            'Ship Date': pd.to_datetime(['2023-01-05', '2023-01-20']),
            'Sales': [100, 200],
            'Profit': [20, 40],
            'Discount': [0, 0.1]
        })
        result = create_features(df)
        self.assertIn('Shipping Days', result.columns)
        self.assertIn('Profit Margin', result.columns)

if __name__ == '__main__':
    unittest.main()
```

---

## Deliverable

**Cleaned Dataset + Notebook** with:
- Data cleaning pipeline (`scripts/data_cleaning.py`)
- Analysis notebook (`notebooks/week03_analysis.ipynb`)
- Visualizations (`outputs/`)
- Unit tests (`tests/test_cleaning.py`)
- Narrative documentation

---

## Folder Structure
```
week-03-python-analysis/
├── data/
│   ├── superstore.csv
│   └── superstore_cleaned.csv
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_analysis.ipynb
├── scripts/
│   └── data_cleaning.py
├── tests/
│   └── test_cleaning.py
├── outputs/
│   ├── sales_by_category.png
│   ├── monthly_sales_trend.png
│   └── profit_vs_sales.png
└── README.md
```

---

## Commit Message
```
week-03: Complete data cleaning pipeline for Superstore

- Add data cleaning functions with docstrings
- Create analysis notebook with visualizations
- Add unit tests for cleaning functions
- Generate cleaned dataset with new features
```
