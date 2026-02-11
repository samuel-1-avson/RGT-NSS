-- Week 2: JOINs and Aggregations
-- Retail Database Analysis

-- =============================================
-- Query 6: Orders with customer names
-- =============================================
SELECT 
    o.order_id,
    o.order_date,
    c.first_name || ' ' || c.last_name as customer_name,
    c.city,
    o.status
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LIMIT 20;

-- =============================================
-- Query 7: Top 10 customers by total spend
-- =============================================
SELECT 
    c.customer_id,
    c.first_name || ' ' || c.last_name as customer_name,
    c.customer_segment,
    ROUND(SUM(oi.quantity * oi.unit_price), 2) as total_revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY c.customer_id
ORDER BY total_revenue DESC
LIMIT 10;

-- =============================================
-- Query 8: Revenue by product category
-- =============================================
SELECT 
    p.category,
    ROUND(SUM(oi.quantity * oi.unit_price), 2) as revenue,
    SUM(oi.quantity) as units_sold,
    COUNT(DISTINCT o.order_id) as order_count
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY revenue DESC;

-- =============================================
-- Query 9: Monthly revenue trend
-- =============================================
SELECT 
    strftime('%Y-%m', o.order_date) as month,
    COUNT(DISTINCT o.order_id) as order_count,
    ROUND(SUM(oi.quantity * oi.unit_price), 2) as revenue
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY month
ORDER BY month;

-- =============================================
-- Query 10: Average order value by customer segment
-- =============================================
SELECT 
    c.customer_segment,
    ROUND(AVG(order_total), 2) as avg_order_value,
    COUNT(*) as order_count,
    ROUND(SUM(order_total), 2) as total_revenue
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

-- =============================================
-- Query 11: Orders by status
-- =============================================
SELECT 
    status,
    COUNT(*) as order_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 2) as percentage
FROM orders
GROUP BY status;

-- =============================================
-- Query 12: Customers with no orders
-- =============================================
SELECT 
    c.customer_id,
    c.first_name || ' ' || c.last_name as customer_name,
    c.email,
    c.signup_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
