# Week 2: SQL for Analytics

> **Branch**: `week-02-sql-analytics` | **Review Required**: Yes  
> **Dataset**: Synthetic Retail Database (generated via Python)

---

## Git Workflow for This Week

```bash
# 1. Start from updated main
git checkout main
git pull origin main

# 2. Create Week 2 branch
git checkout -b week-02-sql-analytics

# 3. Work on tasks, commit regularly
git add .
git commit -m "week-02: Description of changes"
git push origin week-02-sql-analytics

# 4. Create PR at end of week for supervisor review
```

---

## Learning Objectives
- Write complex SQL queries for data analysis
- Master JOINs, aggregations, and window functions
- Generate KPI reports from databases
- Export and document query results

---

## Dataset

**Name**: Synthetic Retail Database  
**Source**: Generated via Python (Faker library)  
**Tables**: customers, orders, products, order_items  
**Description**: Simulated retail transaction data

### Generate the Database

```python
# scripts/generate_retail_db.py

import sqlite3
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)

# Generate Customers (100 customers)
customers = pd.DataFrame({
    'customer_id': range(1, 101),
    'first_name': [fake.first_name() for _ in range(100)],
    'last_name': [fake.last_name() for _ in range(100)],
    'email': [fake.email() for _ in range(100)],
    'city': [fake.city() for _ in range(100)],
    'country': [fake.country() for _ in range(100)],
    'signup_date': [fake.date_between('-2y', 'today') for _ in range(100)],
    'customer_segment': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 100, p=[0.4, 0.3, 0.2, 0.1])
})

# Generate Products (50 products)
categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
products = pd.DataFrame({
    'product_id': range(1, 51),
    'product_name': [fake.catch_phrase() for _ in range(50)],
    'category': np.random.choice(categories, 50),
    'price': np.round(np.random.uniform(10, 500, 50), 2),
    'cost': np.round(np.random.uniform(5, 300, 50), 2)
})

# Generate Orders (500 orders)
orders = pd.DataFrame({
    'order_id': range(1, 501),
    'customer_id': np.random.choice(range(1, 101), 500),
    'order_date': [fake.date_between('-1y', 'today') for _ in range(500)],
    'status': np.random.choice(['completed', 'pending', 'cancelled'], 500, p=[0.8, 0.1, 0.1])
})

# Generate Order Items (1-5 items per order)
order_items_list = []
for order_id in range(1, 501):
    n_items = random.randint(1, 5)
    for _ in range(n_items):
        product_id = random.randint(1, 50)
        quantity = random.randint(1, 10)
        price = products[products['product_id'] == product_id]['price'].values[0]
        order_items_list.append({
            'order_item_id': len(order_items_list) + 1,
            'order_id': order_id,
            'product_id': product_id,
            'quantity': quantity,
            'unit_price': price
        })

order_items = pd.DataFrame(order_items_list)

# Save to SQLite
conn = sqlite3.connect('../data/retail.db')
customers.to_sql('customers', conn, index=False, if_exists='replace')
products.to_sql('products', conn, index=False, if_exists='replace')
orders.to_sql('orders', conn, index=False, if_exists='replace')
order_items.to_sql('order_items', conn, index=False, if_exists='replace')

print("Database created successfully!")
print(f"Customers: {len(customers)}")
print(f"Products: {len(products)}")
print(f"Orders: {len(orders)}")
print(f"Order Items: {len(order_items)}")
```

---

## Weekly Structure

### Prep (≤60 min)
- [ ] Complete SQLBolt interactive lessons 1-12
- [ ] Review Mode SQL Tutorial sections 1-4

### Guided Lab (≤120 min)

#### Lab 2.1: Basic Queries
```sql
-- queries/01_basic_queries.sql

-- Query 1: Get all customers
SELECT * FROM customers LIMIT 10;

-- Query 2: Count total customers
SELECT COUNT(*) as total_customers FROM customers;

-- Query 3: Customers by city (top 10)
SELECT city, COUNT(*) as customer_count
FROM customers
GROUP BY city
ORDER BY customer_count DESC
LIMIT 10;

-- Query 4: Customers by segment
SELECT customer_segment, COUNT(*) as count
FROM customers
GROUP BY customer_segment;

-- Query 5: Recent signups (last 30 days)
SELECT * FROM customers
WHERE signup_date >= DATE('now', '-30 days');
```

#### Lab 2.2: JOINs and Aggregations
```sql
-- queries/02_joins_aggregations.sql

-- Query 6: Orders with customer names
SELECT 
    o.order_id,
    o.order_date,
    c.first_name || ' ' || c.last_name as customer_name,
    c.city,
    o.status
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LIMIT 20;

-- Query 7: Total revenue by customer
SELECT 
    c.customer_id,
    c.first_name || ' ' || c.last_name as customer_name,
    SUM(oi.quantity * oi.unit_price) as total_revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY c.customer_id
ORDER BY total_revenue DESC
LIMIT 10;

-- Query 8: Revenue by product category
SELECT 
    p.category,
    SUM(oi.quantity * oi.unit_price) as revenue,
    SUM(oi.quantity) as units_sold
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY revenue DESC;

-- Query 9: Monthly revenue trend
SELECT 
    strftime('%Y-%m', o.order_date) as month,
    COUNT(DISTINCT o.order_id) as order_count,
    SUM(oi.quantity * oi.unit_price) as revenue
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY month
ORDER BY month;

-- Query 10: Average order value by customer segment
SELECT 
    c.customer_segment,
    AVG(order_total) as avg_order_value,
    COUNT(*) as order_count
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
) order_totals ON c.customer_id = order_totals.customer_id
GROUP BY c.customer_segment
ORDER BY avg_order_value DESC;
```

#### Lab 2.3: Window Functions
```sql
-- queries/03_window_functions.sql

-- Query 11: Running total revenue
SELECT 
    month,
    revenue,
    SUM(revenue) OVER (ORDER BY month) as running_total
FROM (
    SELECT 
        strftime('%Y-%m', order_date) as month,
        SUM(quantity * unit_price) as revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE status = 'completed'
    GROUP BY month
) monthly_revenue
ORDER BY month;

-- Query 12: Customer ranking by revenue
SELECT 
    customer_name,
    total_revenue,
    RANK() OVER (ORDER BY total_revenue DESC) as revenue_rank
FROM (
    SELECT 
        c.first_name || ' ' || c.last_name as customer_name,
        SUM(oi.quantity * oi.unit_price) as total_revenue
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id
) customer_revenue
LIMIT 20;

-- Query 13: Month-over-month growth
WITH monthly_revenue AS (
    SELECT 
        strftime('%Y-%m', order_date) as month,
        SUM(quantity * unit_price) as revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE status = 'completed'
    GROUP BY month
)
SELECT 
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) as prev_month_revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY month)) * 100.0 / 
        LAG(revenue) OVER (ORDER BY month), 2
    ) as growth_percent
FROM monthly_revenue
ORDER BY month;
```

#### Lab 2.4: Advanced Analytics
```sql
-- queries/04_advanced_analytics.sql

-- Query 14: Customer lifetime value (CLV)
SELECT 
    c.customer_id,
    c.first_name || ' ' || c.last_name as customer_name,
    c.customer_segment,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(oi.quantity * oi.unit_price) as total_revenue,
    AVG(order_total) as avg_order_value,
    JULIANDAY('now') - JULIANDAY(MIN(o.order_date)) as customer_tenure_days
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY c.customer_id
ORDER BY total_revenue DESC
LIMIT 20;

-- Query 15: Product affinity analysis (frequently bought together)
SELECT 
    p1.product_name as product_a,
    p2.product_name as product_b,
    COUNT(*) as times_bought_together
FROM order_items oi1
JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
JOIN products p1 ON oi1.product_id = p1.product_id
JOIN products p2 ON oi2.product_id = p2.product_id
GROUP BY p1.product_id, p2.product_id
HAVING times_bought_together >= 5
ORDER BY times_bought_together DESC
LIMIT 10;

-- Query 16: Cohort analysis (retention by signup month)
WITH customer_cohorts AS (
    SELECT 
        customer_id,
        strftime('%Y-%m', signup_date) as cohort_month
    FROM customers
),
order_cohorts AS (
    SELECT 
        o.customer_id,
        cc.cohort_month,
        strftime('%Y-%m', o.order_date) as order_month,
        (strftime('%Y', o.order_date) - strftime('%Y', cc.cohort_month || '-01')) * 12 +
        (strftime('%m', o.order_date) - strftime('%m', cc.cohort_month || '-01')) as periods_since_signup
    FROM orders o
    JOIN customer_cohorts cc ON o.customer_id = cc.customer_id
    WHERE o.status = 'completed'
)
SELECT 
    cohort_month,
    periods_since_signup,
    COUNT(DISTINCT customer_id) as active_customers
FROM order_cohorts
GROUP BY cohort_month, periods_since_signup
ORDER BY cohort_month, periods_since_signup;
```

### Independent Work (≤120 min)

#### Task 1: Export Query Results
```python
# scripts/export_results.py

import sqlite3
import pandas as pd

conn = sqlite3.connect('../data/retail.db')

# List of queries to export
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
            SUM(oi.quantity * oi.unit_price) as revenue
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
            SUM(oi.quantity * oi.unit_price) as total_revenue
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
            SUM(oi.quantity * oi.unit_price) as revenue
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
            AVG(order_total) as avg_order_value
        FROM customers c
        JOIN (
            SELECT customer_id, SUM(quantity * unit_price) as order_total
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE status = 'completed'
            GROUP BY order_id
        ) ot ON c.customer_id = ot.customer_id
        GROUP BY c.customer_segment
    '''
}

# Export each query
for name, query in queries.items():
    df = pd.read_sql(query, conn)
    df.to_csv(f'../results/{name}.csv', index=False)
    print(f"Exported: {name}.csv ({len(df)} rows)")

conn.close()
```

#### Task 2: Document Business Insights
Create `docs/sql_insights.md`:
```markdown
# SQL Analysis Insights

## Key Findings

### 1. Customer Distribution
- Top 3 cities account for 15% of customers
- Silver segment is the largest (30% of customers)

### 2. Revenue Trends
- Monthly revenue averages $45,000
- December shows highest revenue (holiday season)
- Growth rate: 5% month-over-month

### 3. Top Customers
- Top 10 customers generate 25% of revenue
- Average order value: $150
- Platinum customers have 3x higher AOV

### 4. Product Performance
- Electronics category leads with 35% of revenue
- Average margin: 40%
- Top product: Wireless Headphones

### 5. Recommendations
1. Focus retention efforts on top 20 customers
2. Expand Electronics inventory
3. Target Gold segment for upgrades
```

---

## Deliverable

### SQL Report containing:

1. **SQL Scripts** (`sql/` folder)
   - `01_basic_queries.sql` - Basic SELECT, WHERE, GROUP BY
   - `02_joins_aggregations.sql` - JOINs and aggregations
   - `03_window_functions.sql` - Window functions
   - `04_advanced_analytics.sql` - Cohort, CLV, affinity

2. **Results** (`results/` folder)
   - 5+ CSV files with query results

3. **Insights Document** (`docs/sql_insights.md`)
   - Business insights for each query
   - Recommendations

---

## Folder Structure

```
week-02-sql-analytics/
├── data/
│   └── retail.db              # SQLite database
├── scripts/
│   ├── generate_retail_db.py  # Database generation
│   └── export_results.py      # Export query results
├── sql/
│   ├── 01_basic_queries.sql
│   ├── 02_joins_aggregations.sql
│   ├── 03_window_functions.sql
│   └── 04_advanced_analytics.sql
├── results/
│   ├── 01_customer_count_by_city.csv
│   ├── 02_monthly_revenue.csv
│   ├── 03_top_customers.csv
│   ├── 04_revenue_by_category.csv
│   └── 05_avg_order_by_segment.csv
├── docs/
│   └── sql_insights.md
└── README.md
```

---

## Submission Checklist

- [ ] Synthetic database generated and saved
- [ ] All 16 queries written and tested
- [ ] Query results exported to CSV
- [ ] Business insights documented
- [ ] All SQL files properly commented
- [ ] All code committed with clear messages
- [ ] Branch pushed to GitHub
- [ ] Pull Request created for supervisor review

---

## Commit Message Example
```
week-02: Add 16 SQL queries with retail analysis

- Generate synthetic retail database with 100 customers, 500 orders
- Create basic queries, JOINs, window functions
- Export results to CSV files
- Document business insights and recommendations
```

---

## Resources
- [SQLBolt Interactive](https://sqlbolt.com/)
- [Mode SQL Tutorial](https://mode.com/sql-tutorial/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
