-- Milestone Project A: Heart Disease Analysis
-- SQL Queries for Healthcare Data Analysis

-- =============================================
-- 1. Basic Demographics
-- =============================================

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

-- =============================================
-- 2. Disease Prevalence
-- =============================================

-- Overall disease rate
SELECT 
    CASE WHEN target = 1 THEN 'Disease Present' ELSE 'No Disease' END as diagnosis,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM heart_disease), 2) as percentage
FROM heart_disease
GROUP BY target;

-- Disease by gender
SELECT 
    CASE WHEN sex = 1 THEN 'Male' ELSE 'Female' END as gender,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) as disease_count,
    COUNT(*) as total,
    ROUND(SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as disease_rate
FROM heart_disease
GROUP BY sex;

-- Disease by age group
SELECT 
    CASE 
        WHEN age < 50 THEN 'Under 50'
        WHEN age BETWEEN 50 AND 60 THEN '50-60'
        ELSE 'Over 60'
    END as age_group,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) as disease_count,
    COUNT(*) as total,
    ROUND(SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as disease_rate
FROM heart_disease
GROUP BY age_group
ORDER BY MIN(age);

-- =============================================
-- 3. Risk Factor Analysis
-- =============================================

-- Cholesterol levels
SELECT 
    CASE 
        WHEN chol < 200 THEN 'Desirable (<200)'
        WHEN chol BETWEEN 200 AND 239 THEN 'Borderline (200-239)'
        ELSE 'High (≥240)'
    END as cholesterol_level,
    COUNT(*) as count,
    AVG(CASE WHEN target = 1 THEN 1.0 ELSE 0 END) * 100 as disease_rate
FROM heart_disease
GROUP BY cholesterol_level
ORDER BY MIN(chol);

-- Blood pressure categories
SELECT 
    CASE 
        WHEN trestbps < 120 THEN 'Normal (<120)'
        WHEN trestbps BETWEEN 120 AND 139 THEN 'Elevated (120-139)'
        ELSE 'High (≥140)'
    END as bp_category,
    COUNT(*) as count,
    AVG(CASE WHEN target = 1 THEN 1.0 ELSE 0 END) * 100 as disease_rate
FROM heart_disease
GROUP BY bp_category
ORDER BY MIN(trestbps);

-- Chest pain type analysis
SELECT 
    CASE cp
        WHEN 0 THEN 'Typical Angina'
        WHEN 1 THEN 'Atypical Angina'
        WHEN 2 THEN 'Non-anginal Pain'
        WHEN 3 THEN 'Asymptomatic'
    END as chest_pain_type,
    COUNT(*) as count,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) as disease_count,
    ROUND(SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as disease_rate
FROM heart_disease
GROUP BY cp
ORDER BY disease_rate DESC;

-- =============================================
-- 4. Clinical Test Results
-- =============================================

-- Exercise angina impact
SELECT 
    CASE WHEN exang = 1 THEN 'Exercise Angina: Yes' ELSE 'Exercise Angina: No' END as ex_angina,
    COUNT(*) as count,
    AVG(CASE WHEN target = 1 THEN 1.0 ELSE 0 END) * 100 as disease_rate
FROM heart_disease
GROUP BY exang;

-- Max heart rate analysis
SELECT 
    CASE 
        WHEN thalach < 120 THEN 'Low (<120)'
        WHEN thalach BETWEEN 120 AND 150 THEN 'Medium (120-150)'
        ELSE 'High (>150)'
    END as heart_rate_category,
    COUNT(*) as count,
    AVG(CASE WHEN target = 1 THEN 1.0 ELSE 0 END) * 100 as disease_rate
FROM heart_disease
GROUP BY heart_rate_category
ORDER BY MIN(thalach);

-- ST depression (oldpeak) analysis
SELECT 
    CASE 
        WHEN oldpeak = 0 THEN 'No ST Depression'
        WHEN oldpeak BETWEEN 0.1 AND 2.0 THEN 'Mild (0.1-2.0)'
        ELSE 'Severe (>2.0)'
    END as st_depression,
    COUNT(*) as count,
    AVG(CASE WHEN target = 1 THEN 1.0 ELSE 0 END) * 100 as disease_rate
FROM heart_disease
GROUP BY st_depression
ORDER BY MIN(oldpeak);

-- =============================================
-- 5. High-Risk Patient Profile
-- =============================================

-- Identify high-risk patients (multiple risk factors)
SELECT 
    COUNT(*) as high_risk_count,
    AVG(CASE WHEN target = 1 THEN 1.0 ELSE 0 END) * 100 as actual_disease_rate
FROM heart_disease
WHERE (age > 55 OR chol > 240 OR trestbps > 140)
    AND (exang = 1 OR oldpeak > 2);

-- Average metrics by disease status
SELECT 
    CASE WHEN target = 1 THEN 'Disease' ELSE 'No Disease' END as diagnosis,
    ROUND(AVG(age), 1) as avg_age,
    ROUND(AVG(chol), 1) as avg_cholesterol,
    ROUND(AVG(trestbps), 1) as avg_bp,
    ROUND(AVG(thalach), 1) as avg_max_hr,
    ROUND(AVG(oldpeak), 2) as avg_st_depression
FROM heart_disease
GROUP BY target;
