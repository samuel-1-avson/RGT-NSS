# Milestone Project A: Business Insights Pack

> **Branch**: `milestone-project-a` | **Due**: End of Week 4 | **Weight**: 25%  
> **Dataset**: [Kaggle - Heart Disease UCI](https://www.kaggle.com/datasets/uciml/heart-disease-database)

---

## Git Workflow
```bash
git checkout main
git pull origin main
git checkout -b milestone-project-a
# Work throughout Weeks 1-4, commit regularly
git push origin milestone-project-a
# Create PR when complete
```

---

## Overview

Create a comprehensive business insights package demonstrating your data analytics skills using a healthcare dataset.

---

## Dataset

**Name**: Heart Disease UCI  
**Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/heart-disease-database)  
**Size**: 303 rows × 14 columns  
**Target**: `target` (1 = heart disease, 0 = no heart disease)

---

## Components

### 1. Cleaned Dataset (25%)

**Requirements**:
- [ ] Raw data file (`data/raw/heart.csv`)
- [ ] Cleaned data file (`data/cleaned/heart_cleaned.csv`)
- [ ] Data dictionary (`docs/data_dictionary.md`)
- [ ] Cleaning pipeline script (`scripts/data_cleaning.py`)

**Data Dictionary Template**:
```markdown
# Data Dictionary: Heart Disease Dataset

| Column | Type | Description | Values |
|--------|------|-------------|--------|
| age | int | Age in years | 29-77 |
| sex | int | Gender | 0=Female, 1=Male |
| cp | int | Chest pain type | 0-3 |
| trestbps | int | Resting blood pressure | 94-200 mm Hg |
| chol | int | Serum cholesterol | 126-564 mg/dl |
| fbs | int | Fasting blood sugar > 120 | 0=No, 1=Yes |
| restecg | int | Resting ECG results | 0-2 |
| thalach | int | Max heart rate achieved | 71-202 |
| exang | int | Exercise induced angina | 0=No, 1=Yes |
| oldpeak | float | ST depression | 0.0-6.2 |
| slope | int | Slope of peak exercise ST | 0-2 |
| ca | int | Number of major vessels | 0-3 |
| thal | int | Thalassemia | 0-3 |
| target | int | Heart disease presence | 0=No, 1=Yes |
```

### 2. SQL Analysis (25%)

**Requirements**:
- [ ] 10+ analytical queries (`sql/` folder)
- [ ] Results exported as CSVs (`results/` folder)
- [ ] Query documentation with business context

**Required Queries**:
1. Patient count by age group
2. Heart disease rate by gender
3. Average cholesterol by target
4. Chest pain type distribution
5. Correlation between age and max heart rate
6. High-risk patients (multiple factors)
7. Age group analysis
8. Cholesterol categories
9. Blood pressure categories
10. Multi-factor risk analysis

### 3. Interactive Dashboard (25%)

**Requirements**:
- [ ] Google Looker Studio dashboard
- [ ] Minimum 5 visualizations
- [ ] Interactive filters
- [ ] KPI scorecards

**Dashboard Elements**:
- Total patients
- Heart disease rate
- Average age
- Gender distribution
- Age distribution chart
- Cholesterol by target
- Chest pain type breakdown
- Risk factor analysis

### 4. Documentation & Presentation (25%)

**Requirements**:
- [ ] Comprehensive README (`README.md`)
- [ ] Business problem statement
- [ ] Methodology explanation
- [ ] Key findings and recommendations
- [ ] 3-minute recorded walkthrough

---

## Folder Structure
```
milestone-project-a/
├── data/
│   ├── raw/
│   │   └── heart.csv
│   └── cleaned/
│       └── heart_cleaned.csv
├── sql/
│   ├── 01_basic_queries.sql
│   ├── 02_risk_analysis.sql
│   └── 03_advanced_queries.sql
├── results/
│   ├── age_group_analysis.csv
│   ├── gender_analysis.csv
│   └── risk_factors.csv
├── notebooks/
│   └── eda_and_cleaning.ipynb
├── scripts/
│   └── data_cleaning.py
├── docs/
│   ├── data_dictionary.md
│   ├── business_problem.md
│   └── key_findings.md
├── dashboard/
│   └── dashboard_link.md
├── outputs/
│   └── walkthrough_video.md
├── README.md
└── requirements.txt
```

---

## Submission Checklist

- [ ] All data files included
- [ ] 10+ SQL queries with comments
- [ ] Query results exported
- [ ] Dashboard published and link provided
- [ ] README with setup instructions
- [ ] 3-minute video walkthrough recorded
- [ ] All code committed
- [ ] Pull Request created

---

## Commit Message
```
milestone-a: Complete Business Insights Pack with heart disease analysis

- Add cleaned dataset with data dictionary
- Create 12 SQL queries for risk analysis
- Build Looker Studio dashboard with 8 visualizations
- Document findings and recommendations
- Record 3-minute walkthrough video
```
