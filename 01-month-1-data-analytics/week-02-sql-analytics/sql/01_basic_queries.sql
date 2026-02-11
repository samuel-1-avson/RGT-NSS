-- Week 2: Basic SQL Queries
-- Retail Database Analysis

-- =============================================
-- Query 1: Get all customers (first 10)
-- =============================================
SELECT * FROM customers LIMIT 10;

-- =============================================
-- Query 2: Count total customers
-- =============================================
SELECT COUNT(*) as total_customers FROM customers;

-- =============================================
-- Query 3: Customers by city (top 10)
-- =============================================
SELECT 
    city, 
    COUNT(*) as customer_count
FROM customers
GROUP BY city
ORDER BY customer_count DESC
LIMIT 10;

-- =============================================
-- Query 4: Customers by segment
-- =============================================
SELECT 
    customer_segment, 
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) as percentage
FROM customers
GROUP BY customer_segment
ORDER BY count DESC;

-- =============================================
-- Query 5: Recent signups (last 30 days)
-- =============================================
SELECT 
    customer_id,
    first_name,
    last_name,
    email,
    signup_date
FROM customers
WHERE signup_date >= DATE('now', '-30 days')
ORDER BY signup_date DESC;
