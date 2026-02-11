"""
Unit Tests for Data Cleaning Pipeline

Run with: python -m pytest test_cleaning.py -v
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from data_cleaning import (
    remove_duplicates,
    handle_missing_values,
    fix_data_types,
    create_features,
    validate_data
)


class TestDataCleaning(unittest.TestCase):
    """Test cases for data cleaning functions."""
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        df = pd.DataFrame({
            'A': [1, 1, 2, 3],
            'B': [3, 3, 4, 5]
        })
        result = remove_duplicates(df)
        self.assertEqual(len(result), 3)
        
    def test_remove_duplicates_no_dups(self):
        """Test with no duplicates."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        result = remove_duplicates(df)
        self.assertEqual(len(result), 3)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df = pd.DataFrame({
            'Postal Code': [10001.0, np.nan, 10002.0]
        })
        result = handle_missing_values(df)
        self.assertEqual(result['Postal Code'].isnull().sum(), 0)
        self.assertEqual(result['Postal Code'].dtype, 'int64')
    
    def test_handle_missing_values_no_missing(self):
        """Test with no missing values."""
        df = pd.DataFrame({
            'Postal Code': [10001, 10002, 10003]
        })
        result = handle_missing_values(df)
        self.assertEqual(len(result), 3)
    
    def test_fix_data_types(self):
        """Test date column conversion."""
        df = pd.DataFrame({
            'Order Date': ['2023-01-01', '2023-01-15'],
            'Ship Date': ['2023-01-05', '2023-01-20']
        })
        result = fix_data_types(df)
        self.assertEqual(result['Order Date'].dtype, 'datetime64[ns]')
        self.assertEqual(result['Ship Date'].dtype, 'datetime64[ns]')
    
    def test_create_features_shipping_days(self):
        """Test shipping days feature creation."""
        df = pd.DataFrame({
            'Order Date': pd.to_datetime(['2023-01-01', '2023-01-15']),
            'Ship Date': pd.to_datetime(['2023-01-05', '2023-01-20']),
            'Sales': [100, 200],
            'Profit': [20, 40],
            'Discount': [0, 0.1],
            'Quantity': [1, 2]
        })
        result = create_features(df)
        self.assertIn('Shipping Days', result.columns)
        self.assertEqual(result['Shipping Days'].iloc[0], 4)
        self.assertEqual(result['Shipping Days'].iloc[1], 5)
    
    def test_create_features_profit_margin(self):
        """Test profit margin feature creation."""
        df = pd.DataFrame({
            'Order Date': pd.to_datetime(['2023-01-01']),
            'Ship Date': pd.to_datetime(['2023-01-05']),
            'Sales': [100],
            'Profit': [20],
            'Discount': [0],
            'Quantity': [1]
        })
        result = create_features(df)
        self.assertIn('Profit Margin', result.columns)
        self.assertAlmostEqual(result['Profit Margin'].iloc[0], 0.2)
    
    def test_create_features_time_features(self):
        """Test time-based feature creation."""
        df = pd.DataFrame({
            'Order Date': pd.to_datetime(['2023-03-15', '2023-07-20']),
            'Ship Date': pd.to_datetime(['2023-03-20', '2023-07-25']),
            'Sales': [100, 200],
            'Profit': [20, 40],
            'Discount': [0, 0.1],
            'Quantity': [1, 2]
        })
        result = create_features(df)
        self.assertIn('Order Year', result.columns)
        self.assertIn('Order Month', result.columns)
        self.assertIn('Order Quarter', result.columns)
        self.assertEqual(result['Order Year'].iloc[0], 2023)
        self.assertEqual(result['Order Month'].iloc[0], 3)
        self.assertEqual(result['Order Quarter'].iloc[0], 1)
    
    def test_validate_data(self):
        """Test data validation."""
        df = pd.DataFrame({
            'Sales': [100, 200, -50],
            'Profit': [20, -10, 30],
            'Shipping Days': [3, -1, 5]
        })
        results = validate_data(df)
        self.assertIn('total_rows', results)
        self.assertIn('negative_sales', results)
        self.assertIn('negative_profit', results)
        self.assertIn('negative_shipping_days', results)
        self.assertEqual(results['negative_sales'], 1)
        self.assertEqual(results['negative_profit'], 1)
        self.assertEqual(results['negative_shipping_days'], 1)


class TestDataCleaningIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""
    
    def test_full_pipeline(self):
        """Test complete cleaning pipeline."""
        # Create sample data
        df = pd.DataFrame({
            'Order Date': ['2023-01-01', '2023-01-01', '2023-01-15'],
            'Ship Date': ['2023-01-05', '2023-01-05', '2023-01-20'],
            'Sales': [100, 100, 200],  # First two are duplicates
            'Profit': [20, 20, 40],
            'Discount': [0, 0, 0.1],
            'Quantity': [1, 1, 2],
            'Postal Code': [10001.0, 10001.0, np.nan]
        })
        
        # Apply cleaning steps
        df = remove_duplicates(df)
        df = handle_missing_values(df)
        df = fix_data_types(df)
        df = create_features(df)
        
        # Assertions
        self.assertEqual(len(df), 2)  # Duplicates removed
        self.assertEqual(df['Postal Code'].isnull().sum(), 0)  # Missing filled
        self.assertEqual(df['Order Date'].dtype, 'datetime64[ns]')  # Dates converted
        self.assertIn('Shipping Days', df.columns)  # Features created


if __name__ == '__main__':
    unittest.main()
