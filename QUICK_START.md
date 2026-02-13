# Quick Start Guide

## Your GitHub Repository
**URL**: https://github.com/samuel-1-avson/RGT-NSS.git

---

## Setup Instructions

### 1. Clone Your Repository
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

## Weekly Workflow

### Monday: Start New Week
```bash
# Update main branch
git checkout main
git pull origin main

# Create new branch for the week
git checkout -b week-01-tools-setup
```

### Daily: Commit Progress
```bash
# Add and commit your work
git add .
git commit -m "week-XX: Description of changes"
git push origin week-XX-topic-name
```

### Friday: Submit for Review
```bash
# Push final changes
git push origin week-XX-topic-name

# Create Pull Request on GitHub
# - Go to GitHub → Pull Requests → New PR
# - Assign supervisor as reviewer
```

---

## Data Sources by Week

| Week | Dataset | Source | Download Link |
|------|---------|--------|---------------|
| 1 | Customer Churn | Kaggle | [Download](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| 2 | Retail DB | Synthetic | Generated via script |
| 3 | Superstore Sales | Kaggle | [Download](https://www.kaggle.com/datasets/vivek468/superstore-sales-dataset) |
| 4 | Superstore Sales | Week 3 | Use cleaned data |
| A | Heart Disease | Kaggle | [Download](https://www.kaggle.com/datasets/uciml/heart-disease-database) |
| 5 | Customer Churn | Kaggle | [Download](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| 6 | House Prices | Kaggle | [Download](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| B | Credit Card Fraud | Kaggle | [Download](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

---

## Branch Names

```
week-01-tools-setup
week-02-sql-analytics
week-03-python-analysis
week-04-dashboards
milestone-project-a
week-05-supervised-ml-1
week-06-supervised-ml-2
week-07-deployment
week-08-mlops
milestone-project-b
```

---

## Commit Message Format

```
week-XX: Brief description of changes

- Detail 1
- Detail 2
```

**Examples**:
```
week-01: Complete EDA with customer churn dataset

- Add data loading and exploration notebook
- Create visualizations for churn distribution
- Document business and data understanding
```

```
milestone-a: Complete Business Insights Pack

- Add cleaned dataset with data dictionary
- Create 12 SQL queries for risk analysis
- Build Looker Studio dashboard
```

---

## Documentation Requirements

### Every Week Must Include:
1. ✅ README.md with objectives and instructions
2. ✅ Dataset used (Kaggle or synthetic)
3. ✅ Code/notebooks with comments
4. ✅ Output files (CSV, images, etc.)
5. ✅ Clear commit messages

### Every Milestone Must Include:
1. ✅ Comprehensive README
2. ✅ requirements.txt
3. ✅ Model Card (for ML projects)
4. ✅ Demo video link
5. ✅ All components from curriculum

---

## Assessment Weights

| Deliverable | Weight |
|-------------|--------|
| Milestone A | 50% |
| Milestone B | 50% |

---

## Resources

- [Main README](README.md) - Complete program guide
- [GITHUB_SETUP.md](GITHUB_SETUP.md) - Detailed Git workflow
- [Kaggle](https://kaggle.com) - Datasets
- [scikit-learn docs](https://scikit-learn.org) - ML reference

---

## Need Help?

1. Check the week-specific README for detailed instructions
2. Review the lab code templates
3. Ask your supervisor during review
4. Check documentation links in each week's README

---

**Good luck with your training! 🚀**
