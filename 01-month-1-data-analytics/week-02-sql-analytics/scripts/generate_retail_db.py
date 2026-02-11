"""
Generate Synthetic Retail Database for Week 2 SQL Analytics

This script creates a SQLite database with synthetic retail data including:
- Customers (100 records)
- Products (50 records)
- Orders (500 records)
- Order Items (1-5 items per order)
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from faker import Faker
import random

# Set seeds for reproducibility
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)


def generate_customers(n=100):
    """Generate synthetic customer data."""
    customers = pd.DataFrame({
        'customer_id': range(1, n + 1),
        'first_name': [fake.first_name() for _ in range(n)],
        'last_name': [fake.last_name() for _ in range(n)],
        'email': [fake.email() for _ in range(n)],
        'city': [fake.city() for _ in range(n)],
        'country': [fake.country() for _ in range(n)],
        'signup_date': [fake.date_between('-2y', 'today') for _ in range(n)],
        'customer_segment': np.random.choice(
            ['Bronze', 'Silver', 'Gold', 'Platinum'], 
            n, 
            p=[0.4, 0.3, 0.2, 0.1]
        )
    })
    return customers


def generate_products(n=50):
    """Generate synthetic product data."""
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    products = pd.DataFrame({
        'product_id': range(1, n + 1),
        'product_name': [fake.catch_phrase() for _ in range(n)],
        'category': np.random.choice(categories, n),
        'price': np.round(np.random.uniform(10, 500, n), 2),
        'cost': np.round(np.random.uniform(5, 300, n), 2)
    })
    return products


def generate_orders(n=500, n_customers=100):
    """Generate synthetic order data."""
    orders = pd.DataFrame({
        'order_id': range(1, n + 1),
        'customer_id': np.random.choice(range(1, n_customers + 1), n),
        'order_date': [fake.date_between('-1y', 'today') for _ in range(n)],
        'status': np.random.choice(
            ['completed', 'pending', 'cancelled'], 
            n, 
            p=[0.8, 0.1, 0.1]
        )
    })
    return orders


def generate_order_items(orders, products):
    """Generate synthetic order items (1-5 items per order)."""
    order_items_list = []
    
    for order_id in orders['order_id']:
        n_items = random.randint(1, 5)
        for _ in range(n_items):
            product_id = random.randint(1, len(products))
            quantity = random.randint(1, 10)
            price = products[products['product_id'] == product_id]['price'].values[0]
            order_items_list.append({
                'order_item_id': len(order_items_list) + 1,
                'order_id': order_id,
                'product_id': product_id,
                'quantity': quantity,
                'unit_price': price
            })
    
    return pd.DataFrame(order_items_list)


def create_database(db_path=None):
    if db_path is None:
        # Get the script's directory and construct absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, '..', 'data', 'retail.db')
        db_path = os.path.abspath(db_path)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    """Create SQLite database with all tables."""
    print("Generating synthetic retail database...")
    print("=" * 50)
    
    # Generate data
    print("Generating customers...")
    customers = generate_customers(100)
    
    print("Generating products...")
    products = generate_products(50)
    
    print("Generating orders...")
    orders = generate_orders(500, 100)
    
    print("Generating order items...")
    order_items = generate_order_items(orders, products)
    
    # Save to SQLite
    print("\nSaving to SQLite database...")
    conn = sqlite3.connect(db_path)
    
    customers.to_sql('customers', conn, index=False, if_exists='replace')
    products.to_sql('products', conn, index=False, if_exists='replace')
    orders.to_sql('orders', conn, index=False, if_exists='replace')
    order_items.to_sql('order_items', conn, index=False, if_exists='replace')
    
    # Create indexes for better performance
    cursor = conn.cursor()
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id)')
    
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 50)
    print("Database created successfully!")
    print(f"  - Customers: {len(customers):,}")
    print(f"  - Products: {len(products):,}")
    print(f"  - Orders: {len(orders):,}")
    print(f"  - Order Items: {len(order_items):,}")
    print(f"\nDatabase saved to: {db_path}")
    
    return {
        'customers': customers,
        'products': products,
        'orders': orders,
        'order_items': order_items
    }


if __name__ == "__main__":
    data = create_database()
