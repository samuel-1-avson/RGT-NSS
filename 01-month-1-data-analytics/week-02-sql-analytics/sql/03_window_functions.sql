-- Week 2: Window Functions
-- Retail Database Analysis

-- =============================================
-- Query 13: Running total revenue by month
-- =============================================
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
ORDER BY month;

-- =============================================
-- Query 14: Customer ranking by revenue
-- =============================================
SELECT 
    customer_name,
    customer_segment,
    total_revenue,
    RANK() OVER (ORDER BY total_revenue DESC) as revenue_rank,
    DENSE_RANK() OVER (ORDER BY total_revenue DESC) as dense_rank
FROM (
    SELECT 
        c.first_name || ' ' || c.last_name as customer_name,
        c.customer_segment,
        ROUND(SUM(oi.quantity * oi.unit_price), 2) as total_revenue
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id
) customer_revenue
ORDER BY total_revenue DESC
LIMIT 20;

-- =============================================
-- Query 15: Month-over-month growth rate
-- =============================================
WITH monthly_revenue AS (
    SELECT 
        strftime('%Y-%m', order_date) as month,
        ROUND(SUM(quantity * unit_price), 2) as revenue
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

-- =============================================
-- Query 16: Top 3 products by category (using ROW_NUMBER)
-- =============================================
WITH product_sales AS (
    SELECT 
        p.category,
        p.product_name,
        SUM(oi.quantity) as units_sold,
        ROW_NUMBER() OVER (
            PARTITION BY p.category 
            ORDER BY SUM(oi.quantity) DESC
        ) as rank_in_category
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.status = 'completed'
    GROUP BY p.product_id
)
SELECT 
    category,
    product_name,
    units_sold,
    rank_in_category
FROM product_sales
WHERE rank_in_category <= 3
ORDER BY category, rank_in_category;

-- =============================================
-- Query 17: Customer tenure and activity
-- =============================================
SELECT 
    c.customer_id,
    c.first_name || ' ' || c.last_name as customer_name,
    c.signup_date,
    JULIANDAY('now') - JULIANDAY(c.signup_date) as days_since_signup,
    COUNT(DISTINCT o.order_id) as total_orders,
    MAX(o.order_date) as last_order_date,
    JULIANDAY('now') - JULIANDAY(MAX(o.order_date)) as days_since_last_order
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id
ORDER BY total_orders DESC
LIMIT 20;
