# Week 4: Visualization & Dashboards (Google Looker Studio)

> **Branch**: `week-04-dashboards` | **Review Required**: Yes  
> **Dataset**: Cleaned Superstore Sales (from Week 3)

---

## Git Workflow
```bash
git checkout main
git pull origin main
git checkout -b week-04-dashboards
git push origin week-04-dashboards
```

---

## Learning Objectives
- Connect data sources to Looker Studio
- Create calculated fields and metrics
- Design interactive dashboards
- Apply data storytelling principles

---

## Dataset

**Name**: Superstore Sales (Cleaned)  
**Source**: Week 3 output  
**File**: `data/superstore_cleaned.csv`

---

## Weekly Structure

### Prep (≤60 min)
- [ ] Complete Looker Studio tutorials
- [ ] Read Storytelling With Data blog posts

### Guided Lab (≤120 min)

#### Lab 4.1: Prepare Data for Looker Studio
```python
# scripts/prep_for_looker.py
import pandas as pd

df = pd.read_csv('../data/superstore_cleaned.csv', parse_dates=['Order Date'])

# Add calculated fields
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Month Name'] = df['Order Date'].dt.month_name()
df['Quarter'] = df['Order Date'].dt.quarter

# Save for Looker Studio
df.to_csv('../data/superstore_for_looker.csv', index=False)
print(f"Prepared data: {len(df)} rows")
```

#### Lab 4.2: Create Looker Studio Dashboard

**Step 1: Connect Data**
1. Go to [Looker Studio](https://lookerstudio.google.com)
2. Click "Create" → "Data Source"
3. Select "File Upload" and upload `superstore_for_looker.csv`
4. Verify data types are correct

**Step 2: Create Report**
1. Click "Create" → "Report"
2. Add your data source

**Step 3: Build Visualizations**

| Visualization | Configuration |
|---------------|---------------|
| **Scorecard 1** | Total Sales (SUM of Sales) |
| **Scorecard 2** | Total Profit (SUM of Profit) |
| **Scorecard 3** | Order Count (COUNT of Order ID) |
| **Scorecard 4** | Avg Order Value (AVG of Sales) |
| **Line Chart** | Sales by Month (Date: Month, Metric: Sales) |
| **Bar Chart** | Sales by Category (Dimension: Category, Metric: Sales) |
| **Pie Chart** | Sales by Region (Dimension: Region, Metric: Sales) |
| **Table** | Top 10 Products (Dimension: Product Name, Metric: Sales) |
| **Scatter Plot** | Profit vs Sales (X: Sales, Y: Profit) |

**Step 4: Add Filters**
- Date Range Filter
- Region Filter
- Category Filter
- Segment Filter

### Independent Work (≤120 min)

#### Task 1: Design Dashboard Layout
- Arrange visualizations logically
- Add titles and descriptions
- Choose consistent color scheme

#### Task 2: Write Stakeholder Memo
```markdown
# Stakeholder Memo: Superstore Sales Dashboard

## Audience
Sales Managers and Executive Team

## Key Insights

### 1. Overall Performance
- Total Sales: $2.3M
- Total Profit: $286K
- Profit Margin: 12.5%

### 2. Category Performance
- Technology leads with 37% of sales
- Furniture has lowest profit margin (8%)

### 3. Regional Insights
- West region has highest sales
- Central region needs attention (lowest profit)

### 4. Trends
- Sales peaked in November/December
- Consistent growth year-over-year

## Recommended Actions
1. Increase Technology inventory
2. Review Furniture pricing strategy
3. Investigate Central region performance
4. Prepare for Q4 seasonal demand

## Dashboard Link
[Looker Studio Dashboard](YOUR_DASHBOARD_URL)
```

---

## Deliverable

**Interactive Dashboard** + **Stakeholder Memo**:
- Looker Studio dashboard (published, shareable link)
- 1-page stakeholder memo (`docs/stakeholder_memo.md`)

---

## Dashboard Requirements
- [ ] 4+ KPI scorecards
- [ ] 5+ visualizations
- [ ] 3+ interactive filters
- [ ] Professional layout
- [ ] Clear titles and labels

---

## Folder Structure
```
week-04-dashboards/
├── data/
│   └── superstore_for_looker.csv
├── scripts/
│   └── prep_for_looker.py
├── docs/
│   └── stakeholder_memo.md
├── outputs/
│   └── dashboard_screenshot.png
└── README.md
```

---

## Commit Message
```
week-04: Add Looker Studio dashboard for Superstore analysis

- Prepare data with calculated fields for Looker Studio
- Create interactive dashboard with 9 visualizations
- Add filters for date, region, category, segment
- Write stakeholder memo with insights and recommendations
```
