# Week 1: Data Literacy, CRISP-DM, Tools Setup

> **Branch**: `week-01-tools-setup` | **Review Required**: Yes  
> **Dataset**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Git Workflow for This Week

```bash
# 1. Start from main
git checkout main
git pull origin main

# 2. Create Week 1 branch
git checkout -b week-01-tools-setup

# 3. Work on tasks, commit regularly
git add .
git commit -m "week-01: Description of changes"
git push origin week-01-tools-setup

# 4. At end of week, create PR on GitHub for supervisor review
```

---

## Learning Objectives
- Understand the CRISP-DM methodology for data mining projects
- Set up development environment (VS Code, Jupyter, Cursor)
- Learn Git/GitHub basics for version control
- Perform initial exploratory data analysis (EDA)

---

## Dataset

**Name**: Telco Customer Churn  
**Source**: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Size**: 7,043 rows × 21 columns  
**Description**: Customer demographic and account information with churn labels

### Download Instructions
1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Click "Download"
3. Extract `WA_Fn-UseC_-Telco-Customer-Churn.csv`
4. Place in `data/` folder

---

## Weekly Structure

### Prep (≤60 min)
- [ ] Watch CRISP-DM overview video
- [ ] Read GitHub getting started guide
- [ ] Complete Jupyter quickstart tutorial
- [ ] Set up Cursor IDE (cursor.so)

### Guided Lab (≤120 min)

#### Lab 1.1: Environment Setup
```bash
# Create project structure
mkdir -p week-01-tools-setup/{data,notebooks,outputs}
cd week-01-tools-setup

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install packages
pip install pandas numpy matplotlib seaborn jupyter
```

#### Lab 1.2: Load and Explore Data
```python
# notebooks/01_load_and_explore.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Task 1: Basic info
print("Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns.tolist())
print("\nData Types:\n", df.dtypes)

# Task 2: Descriptive statistics
print("\nDescriptive Statistics:\n", df.describe())

# Task 3: Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Task 4: Save basic info to file
with open('../outputs/data_summary.txt', 'w') as f:
    f.write(f"Dataset Shape: {df.shape}\n\n")
    f.write(f"Missing Values:\n{df.isnull().sum()}")
```

#### Lab 1.3: Data Quality Analysis
```python
# notebooks/02_data_quality.ipynb

# Task 1: Check for duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Task 2: Check data types
categorical_cols = df.select_dtypes(include=['object']).columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\nCategorical columns: {list(categorical_cols)}")
print(f"\nNumeric columns: {list(numeric_cols)}")

# Task 3: Detect outliers using IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check outliers for MonthlyCharges
outliers, lower, upper = detect_outliers_iqr(df, 'MonthlyCharges')
print(f"\nMonthlyCharges outliers: {len(outliers)}")
print(f"Bounds: [{lower:.2f}, {upper:.2f}]")

# Task 4: Check unique values for categorical columns
for col in ['gender', 'Partner', 'Dependents', 'Churn']:
    print(f"\n{col} unique values: {df[col].unique()}")
```

#### Lab 1.4: Initial Visualizations
```python
# notebooks/03_initial_viz.ipynb

# Task 1: Churn distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Customer Churn Distribution')
plt.savefig('../outputs/churn_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Task 2: Monthly Charges by Churn
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='Set2')
plt.title('Monthly Charges by Churn Status')
plt.savefig('../outputs/monthly_charges_by_churn.png', dpi=300, bbox_inches='tight')
plt.show()

# Task 3: Tenure distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True)
plt.title('Tenure Distribution by Churn')
plt.savefig('../outputs/tenure_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Task 4: Correlation heatmap (numeric columns only)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.savefig('../outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Independent Work (≤120 min)

#### Task 1: Document Business Understanding
Create `docs/business_understanding.md`:
```markdown
# Business Understanding

## Problem Statement
Predict which customers are likely to churn (cancel service) to enable proactive retention strategies.

## Stakeholders
- Marketing Team: Target retention campaigns
- Customer Success: Identify at-risk customers
- Management: Reduce churn rate

## Success Criteria
- Identify key factors contributing to churn
- Build predictive model with >80% accuracy
- Provide actionable insights for retention

## Business Questions
1. What are the main drivers of customer churn?
2. Which customer segments have highest churn risk?
3. What is the financial impact of churn?
```

#### Task 2: Document Data Understanding
Create `docs/data_understanding.md`:
```markdown
# Data Understanding

## Data Source
Telco Customer Churn Dataset from Kaggle

## Dataset Overview
- **Rows**: 7,043
- **Columns**: 21
- **Target Variable**: Churn (Yes/No)

## Key Features
- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Account Info**: tenure, Contract, PaperlessBilling, PaymentMethod
- **Services**: PhoneService, MultipleLines, InternetService, etc.
- **Charges**: MonthlyCharges, TotalCharges

## Data Quality Issues
- TotalCharges has missing values (need to convert to numeric)
- No duplicate rows found
- Churn rate: ~26.5%

## Initial Insights
1. Customers with month-to-month contracts have higher churn
2. Higher monthly charges correlate with churn
3. Tenure is inversely related to churn
```

#### Task 3: Write Initial Findings
Create `outputs/initial_findings.txt`:
```
Initial Findings - Week 1 EDA

1. CHURN DISTRIBUTION
   - Overall churn rate: 26.5%
   - This is a moderately imbalanced dataset

2. KEY CORRELATIONS
   - Tenure: Negative correlation with churn
   - MonthlyCharges: Positive correlation with churn
   - TotalCharges: Negative correlation with churn

3. DEMOGRAPHIC PATTERNS
   - Senior citizens have higher churn rates
   - Customers without partners/dependents churn more

4. SERVICE PATTERNS
   - Fiber optic internet users churn more
   - Month-to-month contracts have highest churn

5. NEXT STEPS
   - Feature engineering (create tenure groups)
   - Handle TotalCharges missing values
   - Encode categorical variables
   - Build baseline model
```

---

## Deliverable

### EDA Notebook (`notebooks/week01_eda.ipynb`)

**Required Sections**:
1. **Business Understanding**
   - Problem statement
   - Stakeholders
   - Success criteria

2. **Data Understanding**
   - Source and size
   - Column descriptions
   - Data quality assessment

3. **Exploratory Analysis**
   - Descriptive statistics
   - Missing values analysis
   - Outlier detection
   - Initial visualizations

4. **Initial Findings**
   - Key insights
   - Patterns discovered
   - Hypotheses formed

5. **Next Steps**
   - Data preparation tasks
   - Modeling approach
   - Questions to answer

---

## Folder Structure

```
week-01-tools-setup/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── week01_eda.ipynb
├── outputs/
│   ├── data_summary.txt
│   ├── churn_distribution.png
│   ├── monthly_charges_by_churn.png
│   ├── tenure_distribution.png
│   └── correlation_heatmap.png
├── docs/
│   ├── business_understanding.md
│   └── data_understanding.md
└── README.md
```

---

## Submission Checklist

- [ ] Dataset downloaded and placed in `data/`
- [ ] EDA notebook completed with all sections
- [ ] Visualizations saved to `outputs/`
- [ ] Business understanding documented
- [ ] Data understanding documented
- [ ] Initial findings written
- [ ] All code committed with clear messages
- [ ] Branch pushed to GitHub
- [ ] Pull Request created for supervisor review

---

## Commit Message Example
```
week-01: Complete EDA with Telco Customer Churn dataset

- Add data loading and exploration notebook
- Create visualizations for churn distribution
- Document business and data understanding
- Identify key patterns and next steps
```

---

## Resources
- [CRISP-DM Overview](https://www.datascience-pm.com/crisp-dm-2/)
- [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Cursor IDE](https://www.cursor.so/)
