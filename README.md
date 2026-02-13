# RGT 2025 NSP AI/Data Training Program

> **8-Week Comprehensive Training in Data Analytics & Machine Learning**  
> **GitHub Repository**: https://github.com/samuel-1-avson/RGT-NSS.git

---

## Program Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | 8 weeks (2 months) |
| **Weekly Commitment** | 10-12 hours/week |
| **Format** | 6 hrs workshops + 4-6 hrs independent study |
| **Data Sources** | Kaggle datasets & Synthetic data only |
| **Tech Stack** | Python, Jupyter/VS Code, SQL, Google Looker Studio, scikit-learn, Git/GitHub, Cursor AI IDE |

---

## Quick Start - Repository Setup

### 1. Clone the Repository
```bash
git clone https://github.com/samuel-1-avson/RGT-NSS.git
cd RGT-NSS
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Git
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

---

## Git Branch Workflow (For Supervisor Review)

### Overview
Every week and milestone has its own branch. All work is pushed to branches and reviewed via Pull Requests before merging to `main`.

```
main ──► week-01-tools-setup ──► PR ──► MERGE (after supervisor approval)
main ──► week-02-sql-analytics ──► PR ──► MERGE
main ──► milestone-project-a ──► PR ──► MERGE
...
```

### Branch Names

| Week | Branch Name | Data Source |
|------|-------------|-------------|
| Week 1 | `week-01-tools-setup` | Kaggle - Customer Churn |
| Week 2 | `week-02-sql-analytics` | Synthetic Retail DB |
| Week 3 | `week-03-python-analysis` | Kaggle - Superstore Sales |
| Week 4 | `week-04-dashboards` | Kaggle - Superstore Sales |
| **Milestone A** | `milestone-project-a` | Kaggle - Healthcare/Retail |
| Week 5 | `week-05-supervised-ml-1` | Kaggle - Customer Churn |
| Week 6 | `week-06-supervised-ml-2` | Kaggle - House Prices |
| Week 7 | `week-07-deployment` | Week 6 Model |
| Week 8 | `week-08-mlops` | Week 7 API |
| **Milestone B** | `milestone-project-b` | Kaggle - Classification Dataset |

---

## Weekly Structure

Each week follows this pattern:

| Component | Time | Description |
|-----------|------|-------------|
| **Prep** | ≤60 min | Watch videos, read documentation |
| **Guided Lab** | ≤120 min | Hands-on exercises with instructor |
| **Independent Work** | ≤120 min | Apply concepts to Kaggle dataset |
| **Deliverable** | - | Submit via Pull Request |

---

## Month 1: Data Analytics Foundations (Weeks 1-4)

### Week 1: Data Literacy, CRISP-DM, Tools Setup
**Branch**: `week-01-tools-setup`

**Dataset**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

#### Prep (≤60 min)
- [ ] Watch CRISP-DM overview video
- [ ] Read GitHub getting started guide
- [ ] Complete Jupyter quickstart tutorial
- [ ] Set up Cursor IDE

#### Guided Lab (≤120 min)
- [ ] Load Customer Churn dataset
- [ ] Generate descriptive statistics (`.describe()`)
- [ ] Identify missing values
- [ ] Detect outliers using IQR method
- [ ] Create initial visualizations

**Lab Code Template**:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Descriptive statistics
print(df.shape)
print(df.describe())

# Missing values
print(df.isnull().sum())

# Outliers
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5*IQR) | (df[column] > Q3 + 1.5*IQR)]
    return outliers

# Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution')
plt.savefig('outputs/churn_distribution.png')
```

#### Independent Work (≤120 min)
- [ ] Document business understanding (problem statement, stakeholders)
- [ ] Document data understanding (source, size, quality issues)
- [ ] Write initial insights and next steps

#### Deliverable
**EDA Notebook** (`notebooks/week01_eda.ipynb`) with:
- Business Understanding section
- Data Understanding section
- Initial findings and recommendations
- Next steps

**Commit Message**: `week-01: Complete EDA with customer churn dataset`

---

### Week 2: SQL for Analytics
**Branch**: `week-02-sql-analytics`

**Dataset**: Synthetic Retail Database (generated via Python)

#### Prep (≤60 min)
- [ ] Complete SQLBolt interactive lessons 1-12
- [ ] Review Mode SQL Tutorial sections 1-4

#### Guided Lab (≤120 min)
- [ ] Generate synthetic retail database
- [ ] Write basic SELECT queries
- [ ] Practice JOINs (INNER, LEFT)
- [ ] Use GROUP BY and aggregations
- [ ] Create window functions

**Lab Code Template**:
```python
import sqlite3
import pandas as pd
import numpy as np
from faker import Faker

# Generate synthetic data
fake = Faker()
np.random.seed(42)

# Create customers table
customers = pd.DataFrame({
    'customer_id': range(1, 101),
    'name': [fake.name() for _ in range(100)],
    'email': [fake.email() for _ in range(100)],
    'city': [fake.city() for _ in range(100)],
    'signup_date': [fake.date_between('-2y') for _ in range(100)]
})

# Create orders table
orders = pd.DataFrame({
    'order_id': range(1, 501),
    'customer_id': np.random.choice(range(1, 101), 500),
    'order_date': [fake.date_between('-1y') for _ in range(500)],
    'amount': np.random.uniform(10, 500, 500).round(2),
    'status': np.random.choice(['completed', 'pending', 'cancelled'], 500)
})

# Create SQLite database
conn = sqlite3.connect('data/retail.db')
customers.to_sql('customers', conn, index=False, if_exists='replace')
orders.to_sql('orders', conn, index=False, if_exists='replace')

# Query 1: Total customers by city
query1 = """
SELECT city, COUNT(*) as customer_count
FROM customers
GROUP BY city
ORDER BY customer_count DESC
LIMIT 10;
"""
pd.read_sql(query1, conn)

# Query 2: Monthly revenue
query2 = """
SELECT 
    strftime('%Y-%m', order_date) as month,
    SUM(amount) as total_revenue,
    COUNT(*) as order_count
FROM orders
WHERE status = 'completed'
GROUP BY month
ORDER BY month;
"""
pd.read_sql(query2, conn)
```

#### Independent Work (≤120 min)
- [ ] Write 10 analytical queries
- [ ] Export results to CSV
- [ ] Document business insights for each query

#### Deliverable
**SQL Report** (`sql/week02_analysis.sql`) containing:
- 10 analytical queries with detailed comments
- Results exported as CSV files (`results/`)
- Business insights narrative (`docs/sql_insights.md`)

**Required Queries**:
1. Customer count by city
2. Monthly revenue trend
3. Top 10 customers by total spend
4. Average order value by month
5. Customer retention rate
6. Orders by status
7. Revenue by customer segment
8. Daily order count
9. Customers with no orders
10. Running total revenue (window function)

**Commit Message**: `week-02: Add 10 SQL queries with retail analysis`

---

### Week 3: Python for Data Analysis
**Branch**: `week-03-python-analysis`

**Dataset**: [Kaggle - Superstore Sales](https://www.kaggle.com/datasets/vivek468/superstore-sales-dataset)

#### Prep (≤60 min)
- [ ] Review pandas user guide (getting started)
- [ ] Read Matplotlib tutorial basics

#### Guided Lab (≤120 min)
- [ ] Load messy Superstore dataset
- [ ] Handle missing values
- [ ] Remove duplicates
- [ ] Fix data types
- [ ] Create reusable cleaning functions

**Lab Code Template**:
```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/superstore.csv')

# Data cleaning pipeline
def clean_superstore_data(df):
    """Clean Superstore dataset."""
    df_clean = df.copy()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Handle missing values
    df_clean['Postal Code'] = df_clean['Postal Code'].fillna(0)
    
    # Fix data types
    df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'])
    df_clean['Ship Date'] = pd.to_datetime(df_clean['Ship Date'])
    
    # Create new features
    df_clean['Shipping Days'] = (df_clean['Ship Date'] - df_clean['Order Date']).dt.days
    df_clean['Profit Margin'] = df_clean['Profit'] / df_clean['Sales']
    
    return df_clean

# Apply cleaning
df_clean = clean_superstore_data(df)

# Save cleaned data
df_clean.to_csv('data/superstore_cleaned.csv', index=False)
```

#### Independent Work (≤120 min)
- [ ] Build complete data cleaning pipeline
- [ ] Create visualizations
- [ ] Write unit tests for helper functions
- [ ] Document pipeline

#### Deliverable
**Cleaned Dataset + Notebook** (`notebooks/week03_cleaning.ipynb`) with:
- Data cleaning pipeline (reusable functions)
- Visualizations (sales trends, profit by category, etc.)
- Unit tests (`tests/test_cleaning.py`)
- Narrative documentation

**Commit Message**: `week-03: Complete data cleaning pipeline for Superstore`

---

### Week 4: Visualization & Dashboards
**Branch**: `week-04-dashboards`

**Dataset**: Cleaned Superstore Sales (from Week 3)

#### Prep (≤60 min)
- [ ] Complete Looker Studio tutorials (Google Analytics Academy)
- [ ] Read Storytelling With Data blog posts

#### Guided Lab (≤120 min)
- [ ] Connect Superstore data to Looker Studio
- [ ] Create KPI scorecards (Total Sales, Total Profit, Order Count)
- [ ] Build charts (line chart for trends, bar chart for categories)
- [ ] Add interactive filters (date range, region, category)

#### Independent Work (≤120 min)
- [ ] Design complete dashboard layout
- [ ] Add calculated fields
- [ ] Write stakeholder memo

#### Deliverable
**Interactive Dashboard** + **Stakeholder Memo**:
- Looker Studio dashboard link (published)
- 1-page stakeholder memo (`docs/stakeholder_memo_week04.md`)

**Dashboard Requirements**:
- Minimum 5 visualizations
- KPI scorecards (Sales, Profit, Orders, Avg Order Value)
- Interactive filters (Date, Region, Category, Segment)
- Drill-down capabilities

**Commit Message**: `week-04: Add Looker Studio dashboard with Superstore analysis`

---

## Milestone Project A: Business Insights Pack
**Branch**: `milestone-project-a` | **Due**: End of Week 4 | **Weight**: 25%

**Dataset**: [Kaggle - Heart Disease UCI](https://www.kaggle.com/datasets/uciml/heart-disease-database) OR [Kaggle - Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Components

#### 1. Cleaned Dataset (25%)
- [ ] Raw data file (`data/raw/`)
- [ ] Cleaned data file (`data/cleaned/`)
- [ ] Data dictionary (`docs/data_dictionary.md`)
- [ ] Cleaning pipeline script (`scripts/data_cleaning.py`)

#### 2. SQL Analysis (25%)
- [ ] 10+ analytical queries (`sql/`)
- [ ] Results exported as CSVs (`results/`)
- [ ] Query documentation with business context

#### 3. Interactive Dashboard (25%)
- [ ] Google Looker Studio dashboard
- [ ] Minimum 5 visualizations
- [ ] Interactive filters and parameters
- [ ] KPI scorecards

#### 4. Documentation & Presentation (25%)
- [ ] Comprehensive README (`README.md`)
- [ ] Business problem statement
- [ ] Methodology explanation
- [ ] Key findings and recommendations
- [ ] 3-minute recorded walkthrough (Loom/Zoom)

### Submission Checklist
- [ ] GitHub repository with all code
- [ ] README.md with clear instructions
- [ ] All datasets (or download links)
- [ ] SQL scripts with comments
- [ ] Dashboard link
- [ ] Recorded walkthrough link
- [ ] Stakeholder memo (PDF)

**Commit Message**: `milestone-a: Complete Business Insights Pack with healthcare analysis`

---

## Month 2: Applied ML & MLOps (Weeks 5-8)

### Week 5: Supervised Learning 1 (scikit-learn)
**Branch**: `week-05-supervised-ml-1`

**Dataset**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

#### Prep (≤60 min)
- [ ] Complete scikit-learn intro tutorial
- [ ] Review ML problem framing guide

#### Guided Lab (≤120 min)
- [ ] Load and prepare churn dataset
- [ ] Implement train/test split (80/20)
- [ ] Train baseline models:
  - Logistic Regression
  - Random Forest
- [ ] Calculate metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

**Lab Code Template**:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

# Load data
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Prepare features (simplified)
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']].fillna(0)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

# Compare results
results_df = pd.DataFrame(results).T
print(results_df)
```

#### Independent Work (≤120 min)
- [ ] Document model selection rationale
- [ ] Create metric comparison table
- [ ] Use Cursor to auto-document experiment logs

#### Deliverable
**ML Notebook** (`notebooks/week05_baseline_models.ipynb`) with:
- Clear problem statement
- Baseline models with metric comparison
- Model selection rationale

**Commit Message**: `week-05: Add baseline ML models for churn prediction`

---

### Week 6: Supervised Learning 2 & Model Interpretability
**Branch**: `week-06-supervised-ml-2`

**Dataset**: [Kaggle - House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

#### Prep (≤60 min)
- [ ] Review scikit-learn pipelines documentation
- [ ] Read about model interpretability techniques

#### Guided Lab (≤120 min)
- [ ] Build complete ML pipeline with preprocessing
- [ ] Implement hyperparameter tuning (GridSearchCV)
- [ ] Compare tuned vs baseline performance
- [ ] Interpret features using permutation importance

**Lab Code Template**:
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import pandas as pd

# Load data
df = pd.read_csv('data/house_prices.csv')

# Define features
numeric_features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt']
categorical_features = ['Neighborhood', 'HouseStyle']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20, None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Feature importance
result = permutation_importance(best_model, X_test, y_test, n_repeats=10)
importance_df = pd.DataFrame({
    'feature': numeric_features + list(best_model.named_steps['preprocessor']
                                       .named_transformers_['cat']
                                       .get_feature_names_out(categorical_features)),
    'importance': result.importances_mean
}).sort_values('importance', ascending=False)
```

#### Independent Work (≤120 min)
- [ ] Complete pipeline with all preprocessing
- [ ] Generate SHAP summary plots (optional)
- [ ] Document feature importance findings

#### Deliverable
**Reproducible Pipeline** (`scripts/week06_ml_pipeline.py`) with:
- Complete preprocessing pipeline
- Tuned model with optimal parameters
- Feature interpretation analysis

**Commit Message**: `week-06: Add ML pipeline with hyperparameter tuning for house prices`

---

### Week 7: Data to Deployment (MLOps Lite)
**Branch**: `week-07-deployment`

**Dataset**: Model from Week 6

#### Prep (≤60 min)
- [ ] Complete FastAPI tutorial
- [ ] Read Docker getting started guide

#### Guided Lab (≤120 min)
- [ ] Package Week 6 model
- [ ] Create FastAPI app with `/predict` endpoint
- [ ] Add input validation with Pydantic
- [ ] Test API locally

**Lab Code Template**:
```python
# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="House Price Prediction API")

# Load model
model = joblib.load('models/house_price_model.pkl')

class PredictionRequest(BaseModel):
    LotArea: float
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    Neighborhood: str
    HouseStyle: str

class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str = "1.0.0"

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Prepare input
    features = np.array([[request.LotArea, request.OverallQual, 
                          request.OverallCond, request.YearBuilt]])
    
    # Predict
    prediction = model.predict(features)[0]
    
    return PredictionResponse(predicted_price=prediction)

# Run: uvicorn api.main:app --reload
```

#### Independent Work (≤120 min)
- [ ] Document API usage
- [ ] Use Cursor to generate API tests
- [ ] Create requirements.txt for API

#### Deliverable
**Local API + README** (`api/`) containing:
- FastAPI application code
- API documentation
- Usage examples
- Test script (`tests/test_api.py`)

**Commit Message**: `week-07: Add FastAPI for house price prediction`

---

### Week 8: MLOps & Model Monitoring
**Branch**: `week-08-mlops`

**Dataset**: Week 7 API

#### Prep (≤60 min)
- [ ] Review ML model cards documentation
- [ ] Read about experiment tracking tools

#### Guided Lab (≤120 min)
- [ ] Create Model Card for your model
- [ ] Set up basic logging
- [ ] Document model limitations

**Model Card Template**:
```markdown
# Model Card: House Price Predictor

## Model Description
- **Type**: Regression
- **Architecture**: Random Forest Regressor
- **Date**: 2025-02-XX

## Intended Use
- **Primary Use**: Predict house prices for real estate analysis
- **Users**: Data analysts, real estate professionals

## Training Data
- **Source**: Kaggle House Prices dataset
- **Size**: 1,460 samples
- **Features**: LotArea, OverallQual, OverallCond, YearBuilt, Neighborhood, HouseStyle

## Performance
- **Metric**: RMSE
- **Value**: 35,000
- **Test Set**: 20% holdout

## Limitations
- Limited to Ames, Iowa housing market
- Does not account for market fluctuations
- Missing features like school district quality

## Ethical Considerations
- Model may perpetuate historical pricing biases
- Should not be used for discriminatory lending practices
```

#### Independent Work (≤120 min)
- [ ] Complete Model Card
- [ ] Prepare for Milestone Project B
- [ ] Practice demo presentation

#### Deliverable
**Model Card** (`docs/model_card.md`) + **Monitoring Setup**

**Commit Message**: `week-08: Add model card and monitoring setup`

---

## Milestone Project B: ML Microservice
**Branch**: `milestone-project-b` | **Due**: End of Week 8 | **Weight**: 25%

**Dataset**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) OR [Kaggle - Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Components

#### 1. Trained Model (20%)
- [ ] Serialized model file (`models/model.pkl`)
- [ ] Preprocessing pipeline (`models/preprocessor.pkl`)
- [ ] Training notebook/script (`notebooks/training.ipynb`)
- [ ] **Model Card** (`docs/model_card.md`)

#### 2. FastAPI Application (30%)
- [ ] `/predict` endpoint (POST)
- [ ] `/health` endpoint (GET)
- [ ] Input validation with Pydantic
- [ ] Error handling
- [ ] Request/response logging

#### 3. Documentation (25%)
- [ ] README with setup instructions
- [ ] API documentation
- [ ] Usage examples (curl, Python)
- [ ] Architecture diagram

#### 4. Testing (15%)
- [ ] Unit tests for API endpoints
- [ ] Integration test script
- [ ] Sample requests/responses

#### 5. Demo (10%)
- [ ] CLI client or notebook demonstrating API usage
- [ ] 5-minute live demo or recorded walkthrough

### Submission Checklist
- [ ] GitHub repository with all code
- [ ] requirements.txt with all dependencies
- [ ] README with clear setup instructions
- [ ] Model file and preprocessing artifacts
- [ ] FastAPI application code
- [ ] Test scripts
- [ ] Model Card (Markdown or PDF)
- [ ] Demo video or live demo scheduled

**Commit Message**: `milestone-b: Complete ML Microservice with fraud detection API`

---

## Data Sources Summary

| Week | Dataset | Source | Link |
|------|---------|--------|------|
| 1 | Customer Churn | Kaggle | [Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| 2 | Retail Database | Synthetic | Generated via Faker |
| 3 | Superstore Sales | Kaggle | [Link](https://www.kaggle.com/datasets/vivek468/superstore-sales-dataset) |
| 4 | Superstore Sales | Kaggle | [Link](https://www.kaggle.com/datasets/vivek468/superstore-sales-dataset) |
| A | Heart Disease / Diabetes | Kaggle | [Link](https://www.kaggle.com/datasets/uciml/heart-disease-database) |
| 5 | Customer Churn | Kaggle | [Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| 6 | House Prices | Kaggle | [Link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| 7 | House Prices Model | Week 6 | Trained model |
| 8 | House Prices API | Week 7 | FastAPI app |
| B | Credit Card Fraud | Kaggle | [Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

---

## Commit Message Standards

### Format
```
[week-XX|milestone-X|capstone]: [action] [description]

[optional details]
```

### Examples
```bash
# Weekly commits
week-01: Add EDA notebook with customer churn analysis
week-02: Implement 10 SQL queries for retail KPIs
week-03: Complete data cleaning pipeline with tests
week-04: Add Looker Studio dashboard link

# Milestone commits
milestone-a: Complete Business Insights Pack
milestone-b: Add ML microservice with FastAPI

# Capstone commits
capstone: Implement RAG pipeline with FAISS
capstone: Add evaluation harness with Ragas
```

---

## Documentation Standards

### Every Commit Must Include
1. **Clear commit message** following format above
2. **Updated README** if adding new features
3. **Inline comments** for complex code
4. **Docstrings** for all functions

### Every Week Must Include
1. **README.md** in week folder with:
   - Week objectives
   - Dataset used
   - Files created
   - How to run
2. **Notebook/script** with clear structure
3. **Output files** (CSV, images, etc.)
4. **Documentation** of findings

### Every Milestone Must Include
1. **Comprehensive README** with setup instructions
2. **requirements.txt** with all dependencies
3. **Model Card** (for ML projects)
4. **API documentation** (for deployment projects)
5. **Demo video** or live demo link

---

## Assessment Rubric

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Technical** | 40% | Correct implementation, reproducible, tested |
| **Analytical** | 30% | Problem framing, metrics, insights |
| **Communication** | 30% | Documentation, presentation, clarity |

---

## Resources

### Kaggle Datasets
- [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Superstore Sales](https://www.kaggle.com/datasets/vivek468/superstore-sales-dataset)
- [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Heart Disease](https://www.kaggle.com/datasets/uciml/heart-disease-database)
- [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Documentation
- [CRISP-DM Guide](https://www.datascience-pm.com/crisp-dm-2/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---


