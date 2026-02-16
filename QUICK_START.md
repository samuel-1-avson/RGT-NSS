# Quick Start Guide

> **Everything you need to set up and deliver your AI/Data training tasks.**

## 🌐 GitHub Repository

**URL**: https://github.com/samuel-1-avson/RGT-NSS.git

---

## 🛠️ Environment Setup

### 1. Python Environment (Months 1-3)

```bash
# Clone and enter repository
git clone https://github.com/samuel-1-avson/RGT-NSS.git
cd RGT-NSS

# Create and activate virtual environment
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

### 2. Local LLM Setup (Month 3)

Required for Week 9-12 and the Capstone Project.

1. **Install Ollama**: Download from [ollama.com](https://ollama.com)
2. **Pull Required Model**:
   ```bash
   ollama pull llama3
   ```

### 3. Frontend Environment (Capstone)

Required for the Telecom Policy Assistant UI.

1. **Install Node.js**: (Version 20.x or higher)
2. **Install Web Dependencies**:
   ```bash
   cd 03-month-3-llm-genai/capstone-project/web-app
   npm install
   ```

---

## 📅 Branch Schedule

| Week            | Branch Name                | Focus                 |
| --------------- | -------------------------- | --------------------- |
| Week 1          | `week-01-tools-setup`      | EDA & Tools           |
| Week 2          | `week-02-sql-analytics`    | SQL Foundations       |
| Week 3          | `week-03-python-analysis`  | Python/Pandas         |
| Week 4          | `week-04-dashboards`       | Looker Dashboards     |
| **Milestone A** | **`milestone-project-a`**  | **Business Insights** |
| Week 5          | `week-05-supervised-ml-1`  | Baseline ML           |
| Week 6          | `week-06-supervised-ml-2`  | Tuned Pipelines       |
| Week 7          | `week-07-deployment`       | FastAPI Serving       |
| Week 8          | `week-08-mlops`            | MLOps & Monitoring    |
| **Milestone B** | **`milestone-project-b`**  | **ML Microservice**   |
| Week 9          | `week-09-llm-fundamentals` | Prompt Engineering    |
| Week 10         | `week-10-langchain-apps`   | LCEL & Chains         |
| Week 11         | `week-11-rag-vector-db`    | FAISS & RAG           |
| Week 12         | `week-12-evaluation`       | Ragas & QA            |
| **Milestone C** | **`milestone-project-c`**  | **Capstone RAG App**  |

---

## 🔄 Delivery Workflow

### 1. Starting a Task

```bash
git checkout main
git pull origin main
git checkout -b <branch-name>
```

### 2. Committing Progress

```bash
# Standard Format: [week-XX|milestone-X]: [Action] [Description]
git add .
git commit -m "week-09: Add CoT prompting patterns to cookbook"
git push origin <branch-name>
```

---

## 📋 Quality Checklist

### Weekly Deliverables

- ✅ Well-documented Notebook or Python script.
- ✅ README.md explaining the specific approach.
- ✅ Data exports (CSV) or visual outputs (PNG/JPG).

### Milestone Deliverables

- ✅ Production-grade code (FastAPI/Next.js).
- ✅ Model Cards / Evaluation Reports.
- ✅ Recorded Walkthrough Video.
- ✅ Requirements.txt / Package.json maintenance.

---

## 📈 Assessment Weights

| Component   | Weight | Focus              |
| ----------- | ------ | ------------------ |
| Milestone A | 25%    | Analytics & SQL    |
| Milestone B | 25%    | ML & Deployment    |
| Milestone C | 50%    | Capstone (LLM/RAG) |

---

## 🔗 Key Resources

- [Main Project README](README.md) - Full Curriculum
- [GITHUB_SETUP.md](GITHUB_SETUP.md) - Extended Git Guide
- [Month 3 Guide](MONTH_3_LLM_GENAI.md) - LLM Deep-Dive

---

**Happy Coding! 🚀**
