"""
Generate Synthetic Superstore Dataset for Week 3
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

def generate_superstore_data(n_rows=1000):
    """Generate synthetic Superstore-like data."""
    print("Generating Superstore dataset...")
    
    # Categories and sub-categories
    categories = {
        'Furniture': ['Chairs', 'Tables', 'Bookcases', 'Furnishings'],
        'Office Supplies': ['Binders', 'Paper', 'Storage', 'Art', 'Accessories', 'Envelopes', 'Labels', 'Supplies'],
        'Technology': ['Phones', 'Copiers', 'Machines', 'Accessories']
    }
    
    # Regions and states
    regions = {
        'West': ['California', 'Washington', 'Arizona', 'Nevada'],
        'East': ['New York', 'Pennsylvania', 'Florida'],
        'Central': ['Texas', 'Illinois', 'Ohio'],
        'South': ['North Carolina', 'Georgia']
    }
    
    # Segments
    segments = ['Consumer', 'Corporate', 'Home Office']
    
    # Ship modes
    ship_modes = ['First Class', 'Second Class', 'Standard Class', 'Same Day']
    
    data = []
    
    for i in range(n_rows):
        # Dates
        order_date = fake.date_between('-2y', 'today')
        ship_date = order_date + timedelta(days=random.randint(1, 7))
        
        # Category and sub-category
        category = random.choice(list(categories.keys()))
        sub_category = random.choice(categories[category])
        
        # Region and state
        region = random.choice(list(regions.keys()))
        state = random.choice(regions[region])
        
        # Sales and quantities
        quantity = random.randint(1, 10)
        sales = round(random.uniform(10, 1000) * quantity, 2)
        discount = round(random.choice([0, 0, 0, 0.1, 0.15, 0.2, 0.3]), 2)
        profit = round(sales * random.uniform(-0.2, 0.3) * (1 - discount), 2)
        
        row = {
            'Row ID': i + 1,
            'Order ID': f'CA-{random.randint(100000, 999999)}',
            'Order Date': order_date.strftime('%Y-%m-%d'),
            'Ship Date': ship_date.strftime('%Y-%m-%d'),
            'Ship Mode': random.choice(ship_modes),
            'Customer ID': f'CG-{random.randint(10000, 99999)}',
            'Customer Name': fake.name(),
            'Segment': random.choice(segments),
            'Country': 'United States',
            'City': fake.city(),
            'State': state,
            'Postal Code': fake.zipcode(),
            'Region': region,
            'Product ID': f'OFF-{random.randint(1000, 9999)}',
            'Category': category,
            'Sub-Category': sub_category,
            'Product Name': fake.catch_phrase(),
            'Sales': sales,
            'Quantity': quantity,
            'Discount': discount,
            'Profit': profit
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'superstore.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
    
    return df

if __name__ == "__main__":
    generate_superstore_data(1000)
