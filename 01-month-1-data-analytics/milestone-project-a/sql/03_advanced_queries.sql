-- 03_advanced_queries.sql
-- Advanced Insights and Patient Profiling

-- Identify high-risk patient profiles
SELECT 
    COUNT(*) as high_risk_count,
    AVG(CASE WHEN target = 1 THEN 1.0 ELSE 0 END) * 100 as actual_disease_rate
FROM heart_disease
WHERE (age > 55 OR chol > 240 OR trestbps > 140)
    AND (exang = 1 OR oldpeak > 2);

-- Metrics summary by disease status
SELECT 
    CASE WHEN target = 1 THEN 'Disease' ELSE 'No Disease' END as diagnosis,
    ROUND(AVG(age), 1) as avg_age,
    ROUND(AVG(chol), 1) as avg_cholesterol,
    ROUND(AVG(trestbps), 1) as avg_bp,
    ROUND(AVG(thalach), 1) as avg_max_hr,
    ROUND(AVG(oldpeak), 2) as avg_st_depression
FROM heart_disease
GROUP BY target;
