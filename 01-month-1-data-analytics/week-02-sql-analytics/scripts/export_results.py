"""
Export SQL Query Results to CSV

This script executes key queries and exports results to CSV files.
"""

import sqlite3
import pandas as pd
import os

# Connect to database
db_path = '../data/retail.db'
if not os.path.exists(db_path):
    print(f"Error: Database not found at {db_path}")
    print("Please run generate_retail_db.py first.")
    exit(1)

conn = sqlite3.connect(db_path)

# Define queries to export
queries = {
    '01_customer_count_by_city': '''
        SELECT city, COUNT(*) as customer_count
        FROM customers
        GROUP BY city
        ORDER BY customer_count DESC
    ''',
    
    '02_monthly_revenue': '''
        SELECT 
            strftime('%Y-%m', o.order_date) as month,
            ROUND(SUM(oi.quantity * oi.unit_price), 2) as revenue,
            COUNT(DISTINCT o.order_id) as order_count
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.status = 'completed'
        GROUP BY month
        ORDER BY month
    ''',
    
    '03_top_customers': '''
        SELECT 
            c.customer_id,
            c.first_name || ' ' || c.last_name as customer_name,
            c.customer_segment,
            ROUND(SUM(oi.quantity * oi.unit_price), 2) as total_revenue,
            COUNT(DISTINCT o.order_id) as total_orders
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.status = 'completed'
        GROUP BY c.customer_id
        ORDER BY total_revenue DESC
        LIMIT 20
    ''',
    
    '04_revenue_by_category': '''
        SELECT 
            p.category,
            ROUND(SUM(oi.quantity * oi.unit_price), 2) as revenue,
            SUM(oi.quantity) as units_sold,
            ROUND(AVG(oi.unit_price), 2) as avg_unit_price
        FROM products p
        JOIN order_items oi ON p.product_id = oi.product_id
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.status = 'completed'
        GROUP BY p.category
        ORDER BY revenue DESC
    ''',
    
    '05_avg_order_by_segment': '''
        SELECT 
            c.customer_segment,
            ROUND(AVG(order_total), 2) as avg_order_value,
            COUNT(*) as order_count,
            ROUND(MIN(order_total), 2) as min_order,
            ROUND(MAX(order_total), 2) as max_order
        FROM customers c
        JOIN (
            SELECT 
                o.customer_id,
                o.order_id,
                SUM(oi.quantity * oi.unit_price) as order_total
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE o.status = 'completed'
            GROUP BY o.order_id
        ) ot ON c.customer_id = ot.customer_id
        GROUP BY c.customer_segment
        ORDER BY avg_order_value DESC
    ''',
    
    '06_orders_by_status': '''
        SELECT 
            status,
            COUNT(*) as order_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 2) as percentage
        FROM orders
        GROUP BY status
        ORDER BY order_count DESC
    ''',
    
    '07_customers_no_orders': '''
        SELECT 
            c.customer_id,
            c.first_name || ' ' || c.last_name as customer_name,
            c.email,
            c.signup_date,
            c.customer_segment
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        WHERE o.order_id IS NULL
    ''',
    
    '08_daily_order_count': '''
        SELECT 
            order_date,
            COUNT(*) as orders,
            COUNT(DISTINCT customer_id) as unique_customers
        FROM orders
        GROUP BY order_date
        ORDER BY order_date
    ''',
    
    '09_customer_retention': '''
        WITH first_order AS (
            SELECT 
                customer_id,
                MIN(order_date) as first_order_date
            FROM orders
            WHERE status = 'completed'
            GROUP BY customer_id
        ),
        repeat_customers AS (
            SELECT 
                o.customer_id,
                COUNT(DISTINCT o.order_id) as total_orders
            FROM orders o
            WHERE o.status = 'completed'
            GROUP BY o.customer_id
            HAVING COUNT(DISTINCT o.order_id) > 1
        )
        SELECT 
            ROUND(COUNT(DISTINCT rc.customer_id) * 100.0 / COUNT(DISTINCT fo.customer_id), 2) as retention_rate_pct
        FROM first_order fo
        LEFT JOIN repeat_customers rc ON fo.customer_id = rc.customer_id
    ''',
    
    '10_running_total_revenue': '''
        SELECT 
            month,
            revenue,
            SUM(revenue) OVER (ORDER BY month) as running_total
        FROM (
            SELECT 
                strftime('%Y-%m', order_date) as month,
                ROUND(SUM(quantity * unit_price), 2) as revenue
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE status = 'completed'
            GROUP BY month
        ) monthly_revenue
        ORDER BY month
    '''
}

# Export each query
print("Exporting query results to CSV...")
print("=" * 50)

os.makedirs('../results', exist_ok=True)

for name, query in queries.items():
    try:
        df = pd.read_sql(query, conn)
        filepath = f'../results/{name}.csv'
        df.to_csv(filepath, index=False)
        print(f"✓ {name}.csv ({len(df)} rows)")
    except Exception as e:
        print(f"✗ {name}.csv - Error: {e}")

conn.close()

print("\n" + "=" * 50)
print("Export complete!")
