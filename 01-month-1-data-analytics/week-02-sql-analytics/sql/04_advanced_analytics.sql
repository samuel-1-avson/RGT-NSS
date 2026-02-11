-- Week 2: Advanced Analytics
-- Retail Database Analysis

-- =============================================
-- Query 18: Customer Lifetime Value (CLV)
-- =============================================
SELECT 
    c.customer_id,
    c.first_name || ' ' || c.last_name as customer_name,
    c.customer_segment,
    COUNT(DISTINCT o.order_id) as total_orders,
    ROUND(SUM(oi.quantity * oi.unit_price), 2) as total_revenue,
    ROUND(AVG(order_total), 2) as avg_order_value,
    ROUND(JULIANDAY('now') - JULIANDAY(MIN(o.order_date)), 0) as customer_tenure_days,
    ROUND(
        SUM(oi.quantity * oi.unit_price) / 
        NULLIF(JULIANDAY('now') - JULIANDAY(MIN(o.order_date)), 0) * 30, 
        2
    ) as monthly_revenue_rate
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY c.customer_id
ORDER BY total_revenue DESC
LIMIT 20;

-- =============================================
-- Query 19: Product affinity analysis
-- (frequently bought together)
-- =============================================
SELECT 
    p1.category as category_a,
    p1.product_name as product_a,
    p2.category as category_b,
    p2.product_name as product_b,
    COUNT(*) as times_bought_together
FROM order_items oi1
JOIN order_items oi2 ON oi1.order_id = oi2.order_id 
    AND oi1.product_id < oi2.product_id
JOIN products p1 ON oi1.product_id = p1.product_id
JOIN products p2 ON oi2.product_id = p2.product_id
JOIN orders o ON oi1.order_id = o.order_id
WHERE o.status = 'completed'
GROUP BY p1.product_id, p2.product_id
HAVING times_bought_together >= 3
ORDER BY times_bought_together DESC
LIMIT 10;

-- =============================================
-- Query 20: Cohort analysis (retention by signup month)
-- =============================================
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
        (CAST(strftime('%Y', o.order_date) AS INTEGER) - CAST(strftime('%Y', cc.cohort_month || '-01') AS INTEGER)) * 12 +
        (CAST(strftime('%m', o.order_date) AS INTEGER) - CAST(strftime('%m', cc.cohort_month || '-01') AS INTEGER)) as periods_since_signup
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
ORDER BY cohort_month, periods_since_signup
LIMIT 30;

-- =============================================
-- Query 21: Revenue by customer segment and month
-- =============================================
SELECT 
    c.customer_segment,
    strftime('%Y-%m', o.order_date) as month,
    COUNT(DISTINCT o.order_id) as orders,
    ROUND(SUM(oi.quantity * oi.unit_price), 2) as revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY c.customer_segment, month
ORDER BY month, 
    CASE c.customer_segment
        WHEN 'Bronze' THEN 1
        WHEN 'Silver' THEN 2
        WHEN 'Gold' THEN 3
        WHEN 'Platinum' THEN 4
    END;

-- =============================================
-- Query 22: High-value customer identification
-- =============================================
WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        c.first_name || ' ' || c.last_name as customer_name,
        c.customer_segment,
        COUNT(DISTINCT o.order_id) as order_count,
        ROUND(SUM(oi.quantity * oi.unit_price), 2) as total_spent,
        ROUND(AVG(oi.quantity * oi.unit_price), 2) as avg_item_value,
        MAX(o.order_date) as last_order_date
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id
)
SELECT 
    *,
    CASE 
        WHEN total_spent >= 5000 AND order_count >= 5 THEN 'VIP'
        WHEN total_spent >= 2000 THEN 'High Value'
        WHEN total_spent >= 500 THEN 'Medium Value'
        ELSE 'Low Value'
    END as customer_tier
FROM customer_metrics
ORDER BY total_spent DESC
LIMIT 25;
