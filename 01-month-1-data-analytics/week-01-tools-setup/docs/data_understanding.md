# Data Understanding

## Data Source

**Dataset**: Telco Customer Churn  
**Source**: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Provider**: IBM Sample Data Sets  

## Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Rows** | 7,043 |
| **Columns** | 21 |
| **Target Variable** | Churn (Yes/No) |
| **Time Period** | Not specified (cross-sectional) |

## Column Descriptions

### Customer Demographics (4 columns)

| Column | Type | Description |
|--------|------|-------------|
| `customerID` | String | Unique customer identifier |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary | 1 = Senior citizen, 0 = Not senior |
| `Partner` | Categorical | Has partner: Yes / No |
| `Dependents` | Categorical | Has dependents: Yes / No |

### Account Information (5 columns)

| Column | Type | Description |
|--------|------|-------------|
| `tenure` | Numeric | Number of months as customer |
| `Contract` | Categorical | Month-to-month, One year, Two year |
| `PaperlessBilling` | Categorical | Yes / No |
| `PaymentMethod` | Categorical | Electronic check, Mailed check, Bank transfer, Credit card |
| `MonthlyCharges` | Numeric | Monthly amount charged |
| `TotalCharges` | Numeric | Total amount charged (cumulative) |

### Services Subscribed (9 columns)

| Column | Type | Description |
|--------|------|-------------|
| `PhoneService` | Categorical | Has phone service: Yes / No |
| `MultipleLines` | Categorical | Multiple phone lines: Yes / No / No phone service |
| `InternetService` | Categorical | DSL, Fiber optic, No |
| `OnlineSecurity` | Categorical | Yes / No / No internet service |
| `OnlineBackup` | Categorical | Yes / No / No internet service |
| `DeviceProtection` | Categorical | Yes / No / No internet service |
| `TechSupport` | Categorical | Yes / No / No internet service |
| `StreamingTV` | Categorical | Yes / No / No internet service |
| `StreamingMovies` | Categorical | Yes / No / No internet service |

### Target Variable (1 column)

| Column | Type | Description |
|--------|------|-------------|
| `Churn` | Categorical | Customer churned: Yes / No |

## Data Quality Assessment

### Missing Values

| Column | Missing Count | Missing % | Notes |
|--------|--------------|-----------|-------|
| TotalCharges | 11 | 0.16% | Empty strings for tenure=0 customers |

### Data Quality Issues

1. **TotalCharges Type**: Stored as string, needs conversion to numeric
2. **Empty TotalCharges**: 11 customers with empty strings (all have tenure=0)
3. **Categorical Encoding**: Binary Yes/No fields could be encoded as 0/1
4. **Redundant Categories**: "No internet service" and "No phone service" categories

### Statistical Summary

#### Numeric Columns

| Column | Min | Max | Mean | Median |
|--------|-----|-----|------|--------|
| tenure | 0 | 72 | 32.37 | 29 |
| MonthlyCharges | 18.25 | 118.75 | 64.76 | 70.35 |
| TotalCharges* | - | - | - | - |
| SeniorCitizen | 0 | 1 | 0.16 | 0 |

*Note: TotalCharges requires cleaning before statistics

#### Target Distribution

| Churn | Count | Percentage |
|-------|-------|------------|
| No | 5,174 | 73.5% |
| Yes | 1,869 | 26.5% |

**Class Balance**: Moderately imbalanced (approx. 3:1 ratio)

## Initial Insights

### Patterns Observed

1. **Tenure Distribution**: Bimodal distribution with many new customers and many long-term customers
2. **Churn Rate**: 26.5% overall churn rate is significant for the business
3. **Service Adoption**: Most customers have phone service (~90%)
4. **Internet Service**: DSL and Fiber optic roughly split among internet users

### Potential Predictive Features

Based on business intuition and initial exploration:

- **tenure**: Expected negative correlation with churn
- **Contract**: Month-to-month expected to have higher churn
- **MonthlyCharges**: Higher charges may lead to higher churn
- **InternetService**: Fiber optic customers may have different churn patterns
- **TechSupport**: Lack of support may increase churn

## Data Preprocessing Needs

1. Convert TotalCharges to numeric
2. Handle empty TotalCharges (impute with 0 for tenure=0)
3. Encode categorical variables
4. Handle "No internet service" categories
5. Consider feature engineering (tenure groups, total services)

## Next Steps

1. Clean and preprocess the data
2. Perform detailed exploratory data analysis
3. Create visualizations for key relationships
4. Prepare data for modeling
