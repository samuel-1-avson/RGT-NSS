# 📊 Month 1: Data Analytics Foundations

> **"Transform raw data into actionable business insights"**

![Data Analytics Banner](https://img.shields.io/badge/Month-1_Data_Analytics-blue?style=for-the-badge)
![Duration](https://img.shields.io/badge/Duration-4_Weeks-green?style=for-the-badge)
![Tools](https://img.shields.io/badge/Tools-Python_|_SQL_|_Looker_Studio-orange?style=for-the-badge)

---

## 🎯 Learning Outcomes

By the end of Month 1, you will be able to:

- ✅ Perform **end-to-end data analysis** using the CRISP-DM methodology
- ✅ Write **complex SQL queries** for business intelligence
- ✅ Build **reusable data pipelines** with Python and pandas
- ✅ Create **interactive dashboards** that tell compelling stories
- ✅ Deliver **professional insights** to stakeholders

---

## 📅 Week-by-Week Breakdown

### 🔍 Week 1: Data Literacy, CRISP-DM & Tools Setup

**Theme:** *Understanding Your Data*

#### What We Covered
- **CRISP-DM Framework**: The industry-standard data mining methodology
  - Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment
- **Tools Setup**: Jupyter, VS Code, Git/GitHub, Python environment
- **Exploratory Data Analysis (EDA)**: First steps with the Telco Customer Churn dataset

#### 📈 Dataset: Telco Customer Churn
| Metric | Value |
|--------|-------|
| **Records** | 7,043 customers |
| **Features** | 21 columns |
| **Target** | Churn (Yes/No) |
| **Churn Rate** | 26.5% |

#### 🎯 Key Insights Discovered
```
🔴 CRITICAL FINDING: Customers with month-to-month contracts 
   have 42% churn rate vs 3% for two-year contracts!

📊 TENURE IMPACT: Average tenure of churned customers 
   is only 18 months vs 38 months for retained

💰 PRICE SENSITIVITY: Churned customers pay $74/month avg 
   vs $61/month for retained
```

#### 🛠️ Deliverables
- ✅ EDA Notebook with 4 visualizations
- ✅ Data quality assessment report
- ✅ Business understanding documentation
- ✅ Git repository initialized

---

### 🗄️ Week 2: SQL for Analytics

**Theme:** *The Language of Data*

#### What We Covered
- **SQL Fundamentals**: SELECT, WHERE, JOIN, GROUP BY, HAVING
- **Advanced Queries**: Subqueries, CTEs (Common Table Expressions)
- **Window Functions**: ROW_NUMBER(), RANK(), LAG(), LEAD()
- **Business Intelligence**: KPI calculation and reporting

#### 📊 Dataset: Synthetic Retail Database
| Table | Records | Description |
|-------|---------|-------------|
| **customers** | 100 | Customer demographics |
| **products** | 50 | Product catalog |
| **orders** | 500 | Transaction records |
| **order_items** | 1,503 | Line items |

#### 🔥 SQL Queries Mastered
```sql
-- Query 1: Customer Lifetime Value
SELECT 
    c.customer_id,
    c.first_name || ' ' || c.last_name as customer_name,
    SUM(oi.quantity * oi.unit_price) as total_revenue,
    RANK() OVER (ORDER BY SUM(oi.quantity * oi.unit_price) DESC) as rank
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY c.customer_id;

-- Query 2: Monthly Revenue Trend with Running Total
SELECT 
    month,
    revenue,
    SUM(revenue) OVER (ORDER BY month) as running_total
FROM monthly_revenue;

-- Query 3: Cohort Analysis (Customer Retention)
SELECT 
    cohort_month,
    periods_since_signup,
    COUNT(DISTINCT customer_id) as active_customers
FROM customer_cohorts
GROUP BY cohort_month, periods_since_signup;
```

#### 📈 Business Intelligence Results
| Metric | Value |
|--------|-------|
| **Total Revenue** | $1,906,019.82 |
| **Top Customer Segment** | Platinum ($3x higher AOV) |
| **Orders by Status** | 80% Completed, 10% Pending, 10% Cancelled |
| **Customer Retention** | ~60% make repeat purchases |

#### 🛠️ Deliverables
- ✅ 16 SQL queries (basic to advanced)
- ✅ 10 exported CSV reports
- ✅ Business insights documentation
- ✅ SQLite database with indexed tables

---

### 🐍 Week 3: Python for Data Analysis

**Theme:** *Building Data Pipelines*

#### What We Covered
- **Pandas Mastery**: DataFrames, filtering, grouping, merging
- **Data Cleaning**: Handling missing values, duplicates, outliers
- **Feature Engineering**: Creating new meaningful variables
- **Pipeline Architecture**: Reusable, testable code

#### 📊 Dataset: Superstore Sales
| Metric | Value |
|--------|-------|
| **Records** | 1,000 orders |
| **Features** | 21 columns |
| **Categories** | Furniture, Office Supplies, Technology |
| **Date Range** | 2024-2026 |

#### 🔄 Data Cleaning Pipeline
```python
# Step 1: Remove Duplicates
df = df.drop_duplicates()

# Step 2: Handle Missing Values
df['Postal Code'] = df['Postal Code'].fillna(0)

# Step 3: Fix Data Types
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Step 4: Feature Engineering
df['Shipping Days'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Profit Margin'] = df['Profit'] / df['Sales']
df['Discount Category'] = pd.cut(df['Discount'], 
                                  bins=[-0.01, 0, 0.2, 0.5, 1.0],
                                  labels=['No Discount', 'Low', 'Medium', 'High'])
```

#### 🎯 Key Findings
```
📦 CATEGORY PERFORMANCE:
   Technology: 37% of sales (highest margin)
   Furniture: 33% of sales
   Office Supplies: 30% of sales

🌍 REGIONAL INSIGHTS:
   West: $108K profit (best performing)
   South: Needs attention (lowest performance)

📈 TRENDS:
   Q4 shows seasonal peaks (holiday shopping)
   Average monthly growth: 3-5%
```

#### 🛠️ Deliverables
- ✅ Data cleaning pipeline (reusable functions)
- ✅ Unit tests (pytest)
- ✅ Analysis notebooks with visualizations
- ✅ Cleaned dataset ready for dashboards

---

### 📈 Week 4: Visualization & Dashboards

**Theme:** *Telling Stories with Data*

#### What We Covered
- **Google Looker Studio**: Connecting data sources
- **KPI Design**: Creating meaningful scorecards
- **Interactive Filters**: Making dashboards dynamic
- **Data Storytelling**: Presenting insights effectively

#### 📊 Dashboard Components
| Component | Purpose |
|-----------|---------|
| **KPI Scorecards** | Total Sales, Profit, Orders, AOV |
| **Line Chart** | Monthly sales trends |
| **Bar Charts** | Sales by category, Profit by region |
| **Pie Chart** | Segment breakdown |
| **Date Filter** | Dynamic time range selection |
| **Category Filter** | Drill-down by product category |

#### 🎯 Dashboard Metrics
| KPI | Value |
|-----|-------|
| **Total Sales** | $2,697,850.40 |
| **Total Profit** | $128,269.92 |
| **Profit Margin** | 4.8% |
| **Order Count** | 999 |
| **Average Order Value** | $2,700.55 |

#### 💡 Dashboard Best Practices Applied
1. **5-Second Rule**: Key metrics visible immediately
2. **Consistent Colors**: Same color = same meaning
3. **Interactive Elements**: Filters for exploration
4. **Mobile Responsive**: Accessible on any device
5. **Export Capability**: Download data for offline analysis

#### 🛠️ Deliverables
- ✅ 4 dashboard-ready CSV files
- ✅ KPI documentation
- ✅ Stakeholder memo
- ✅ Dashboard design guidelines

---

## 🏆 Milestone Project A: Business Insights Pack

**Weight:** 25% of total grade

### Project Overview
Create a comprehensive business insights package using healthcare data (Heart Disease Dataset).

### Deliverables

#### 1️⃣ Data Dictionary
Complete documentation of all 14 features:
- Demographics (age, sex)
- Clinical measurements (blood pressure, cholesterol)
- Test results (ECG, thalassemia)
- Target variable (heart disease presence)

#### 2️⃣ SQL Analysis
12 analytical queries including:
- Disease prevalence by demographic groups
- Risk factor correlation analysis
- Patient segmentation
- High-risk patient identification

#### 3️⃣ Visualizations
| Chart | Insight |
|-------|---------|
| Disease Distribution | 55% disease rate in sample |
| Age Analysis | Higher rates in older patients |
| Cholesterol Levels | Clear correlation with disease |
| Gender Breakdown | 56% female, 54% male disease rates |

#### 4️⃣ Business Recommendations
```
🎯 SCREENING PRIORITY:
   Focus on patients over 50 years old

💊 MONITORING:
   Regular cholesterol checks for high-risk groups

🏥 INTERVENTIONS:
   Lifestyle programs for patients with >2 risk factors
```

### Results
| Metric | Value |
|--------|-------|
| **Patients Analyzed** | 500 |
| **Disease Rate** | 55.0% |
| **Average Age** | 53.0 years |
| **Avg Cholesterol** | 349.1 mg/dl |

---

## 🛠️ Tech Stack Used

| Tool | Purpose |
|------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) | Data processing, EDA, ML |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) | Data manipulation |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) | Interactive notebooks |
| ![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white) | Database management |
| ![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white) | Version control |
| ![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white) | Code repository |
| Google Looker Studio | Dashboards & BI |

---

## 📊 Key Metrics Summary

| Week | Dataset Size | Deliverables | Key Skill |
|------|-------------|--------------|-----------|
| Week 1 | 7,043 rows | 4 visualizations | EDA & Data Understanding |
| Week 2 | 2,153 records | 16 SQL queries | Database & SQL |
| Week 3 | 1,000 rows | Pipeline + Tests | Python & Pandas |
| Week 4 | 1,000 rows | 4 CSV files | Dashboards & BI |
| **Milestone A** | 500 patients | Full report | End-to-end Analysis |

---

## 🎓 Skills Acquired

### Technical Skills
- ✅ Data cleaning and preprocessing
- ✅ SQL query writing (basic to advanced)
- ✅ Python data manipulation (pandas, numpy)
- ✅ Data visualization (matplotlib, seaborn)
- ✅ Dashboard creation (Looker Studio)
- ✅ Git version control

### Soft Skills
- ✅ Business understanding and problem framing
- ✅ Data storytelling and presentation
- ✅ Documentation and reporting
- ✅ Stakeholder communication

---

## 🚀 Next Steps

Month 1 completed! Ready for **Month 2: Applied ML & MLOps** where we'll:
- Build predictive models with scikit-learn
- Deploy models as APIs with FastAPI
- Implement MLOps best practices

---

## 📁 Repository Structure

```
01-month-1-data-analytics/
├── week-01-tools-setup/
│   ├── notebooks/week01_eda.ipynb
│   ├── outputs/
│   │   ├── churn_distribution.png
│   │   ├── correlation_heatmap.png
│   │   └── eda_summary_report.txt
│   └── docs/
│       ├── business_understanding.md
│       └── data_understanding.md
├── week-02-sql-analytics/
│   ├── sql/01_basic_queries.sql
│   ├── sql/02_joins_aggregations.sql
│   ├── sql/03_window_functions.sql
│   ├── sql/04_advanced_analytics.sql
│   └── results/*.csv
├── week-03-python-analysis/
│   ├── scripts/data_cleaning.py
│   ├── tests/test_cleaning.py
│   └── notebooks/02_analysis.ipynb
├── week-04-dashboards/
│   └── data/*.csv
└── milestone-project-a/
    ├── sql/analysis_queries.sql
    ├── docs/data_dictionary.md
    └── outputs/*.png
```

---

## 📝 Resources

- **Dataset**: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Methodology**: [CRISP-DM Guide](https://www.datascience-pm.com/crisp-dm-2/)
- **SQL Practice**: [SQLBolt](https://sqlbolt.com/)
- **Pandas Docs**: [pandas.pydata.org](https://pandas.pydata.org/docs/)
- **Looker Studio**: [Google Analytics Academy](https://analytics.google.com/analytics/academy/course/10)

---

**🔗 Repository:** https://github.com/samuel-1-avson/RGT-NSS

**📅 Completed:** February 2025

---

*This project was completed as part of the RGT 2025 NSP AI/Data Training Program.*
