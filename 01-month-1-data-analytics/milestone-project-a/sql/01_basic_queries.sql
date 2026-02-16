-- 01_basic_demographics.sql
-- Basic Patient Demographics Analysis

-- Patient count by gender
SELECT 
    CASE WHEN sex = 1 THEN 'Male' ELSE 'Female' END as gender,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM heart_disease), 2) as percentage
FROM heart_disease
GROUP BY sex;

-- Age distribution
SELECT 
    CASE 
        WHEN age < 40 THEN 'Under 40'
        WHEN age BETWEEN 40 AND 50 THEN '40-50'
        WHEN age BETWEEN 51 AND 60 THEN '51-60'
        WHEN age BETWEEN 61 AND 70 THEN '61-70'
        ELSE 'Over 70'
    END as age_group,
    COUNT(*) as count
FROM heart_disease
GROUP BY age_group
ORDER BY MIN(age);
