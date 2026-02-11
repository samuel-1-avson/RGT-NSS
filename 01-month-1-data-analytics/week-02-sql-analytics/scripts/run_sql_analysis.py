"""
Week 2 SQL Analysis Script

Runs SQL queries and exports results to CSV.
"""

import sqlite3
import pandas as pd
import os

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'retail.db')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results')
SQL_PATH = os.path.join(os.path.dirname(__file__), '..', 'sql')

def run_queries():
    """Run SQL queries and export results."""
    print("="*60)
    print("WEEK 2: SQL ANALYSIS")
    print("="*60)
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    print(f"\nConnected to: {DB_PATH}")
    
    # Create results directory
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Query 1: Customer count by city
    print("\nRunning queries...")
    
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
        
        '09_running_total_revenue': '''
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
        ''',
        
        '10_customer_lifetime_value': '''
            SELECT 
                c.customer_id,
                c.first_name || ' ' || c.last_name as customer_name,
                c.customer_segment,
                COUNT(DISTINCT o.order_id) as total_orders,
                ROUND(SUM(oi.quantity * oi.unit_price), 2) as total_revenue,
                ROUND(AVG(order_total), 2) as avg_order_value
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN (
                SELECT order_id, SUM(quantity * unit_price) as order_total
                FROM order_items
                GROUP BY order_id
            ) ot ON o.order_id = ot.order_id
            WHERE o.status = 'completed'
            GROUP BY c.customer_id
            ORDER BY total_revenue DESC
            LIMIT 20
        '''
    }
    
    # Run each query and export
    for name, query in queries.items():
        try:
            df = pd.read_sql(query, conn)
            filepath = os.path.join(RESULTS_PATH, f'{name}.csv')
            df.to_csv(filepath, index=False)
            print(f"  [OK] {name}.csv ({len(df)} rows)")
        except Exception as e:
            print(f"  [ERROR] {name}.csv - {e}")
    
    conn.close()
    
    print("\n" + "="*60)
    print("SQL ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_PATH}")

def print_summary_stats():
    """Print summary statistics."""
    conn = sqlite3.connect(DB_PATH)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Total customers
    result = pd.read_sql("SELECT COUNT(*) as count FROM customers", conn)
    print(f"\nTotal Customers: {result['count'].iloc[0]}")
    
    # Total orders
    result = pd.read_sql("SELECT COUNT(*) as count FROM orders", conn)
    print(f"Total Orders: {result['count'].iloc[0]}")
    
    # Total revenue
    result = pd.read_sql('''
        SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) as revenue
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.status = 'completed'
    ''', conn)
    print(f"Total Revenue: ${result['revenue'].iloc[0]:,}")
    
    # Top customer segment
    result = pd.read_sql('''
        SELECT customer_segment, COUNT(*) as count
        FROM customers
        GROUP BY customer_segment
        ORDER BY count DESC
    ''', conn)
    print(f"\nCustomer Segments:")
    for _, row in result.iterrows():
        print(f"  {row['customer_segment']}: {row['count']}")
    
    conn.close()

if __name__ == "__main__":
    run_queries()
    print_summary_stats()
