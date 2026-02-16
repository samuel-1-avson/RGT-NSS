-- 02_risk_analysis.sql
-- Healthcare Risk Factor Analysis

-- Cholesterol levels impact
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
