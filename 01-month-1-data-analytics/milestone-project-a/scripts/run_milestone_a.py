"""
Milestone Project A: Business Insights Pack
Healthcare Analytics - Heart Disease Dataset
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
DB_PATH = os.path.join(DATA_PATH, 'healthcare.db')

def generate_data():
    """Generate heart disease dataset."""
    print("="*60)
    print("MILESTONE A: BUSINESS INSIGHTS PACK")
    print("="*60)
    
    np.random.seed(42)
    n = 500
    
    data = {
        'patient_id': range(1, n+1),
        'age': np.random.randint(29, 77, n),
        'sex': np.random.choice([0, 1], n),
        'cp': np.random.choice([0, 1, 2, 3], n),
        'trestbps': np.random.randint(94, 200, n),
        'chol': np.random.randint(126, 564, n),
        'fbs': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'restecg': np.random.choice([0, 1, 2], n),
        'thalach': np.random.randint(71, 202, n),
        'exang': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'oldpeak': np.round(np.random.uniform(0, 6.2, n), 1),
        'slope': np.random.choice([0, 1, 2], n),
        'ca': np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.2, 0.15, 0.05]),
        'thal': np.random.choice([0, 1, 2], n),
        'target': np.random.choice([0, 1], n, p=[0.45, 0.55])
    }
    
    df = pd.DataFrame(data)
    os.makedirs(DATA_PATH, exist_ok=True)
    df.to_csv(os.path.join(DATA_PATH, 'heart_disease.csv'), index=False)
    
    print(f"\nGenerated: {len(df)} patients, {(df['target']==1).mean()*100:.1f}% disease rate")
    return df

def create_database(df):
    """Create SQLite database with analysis."""
    print("\n--- SQL Analysis ---")
    
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('heart_disease', conn, index=False, if_exists='replace')
    
    results = {}
    
    # Disease by gender
    results['gender'] = pd.read_sql('''
        SELECT CASE WHEN sex=1 THEN 'Male' ELSE 'Female' END as gender,
               ROUND(100.0 * SUM(target) / COUNT(*), 1) as disease_rate
        FROM heart_disease GROUP BY sex''', conn)
    
    # Disease by age
    results['age'] = pd.read_sql('''
        SELECT CASE 
            WHEN age < 50 THEN 'Under 50'
            WHEN age BETWEEN 50 AND 60 THEN '50-60'
            ELSE 'Over 60'
        END as age_group,
        ROUND(100.0 * SUM(target) / COUNT(*), 1) as disease_rate
        FROM heart_disease GROUP BY age_group''', conn)
    
    conn.close()
    print("[OK] SQL queries executed")
    return results

def create_visualizations(df):
    """Create charts."""
    print("\n--- Visualizations ---")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    sns.set_style('whitegrid')
    
    # Disease distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='target', palette='Set2')
    plt.title('Heart Disease Distribution')
    plt.xlabel('Disease (0=No, 1=Yes)')
    for i, v in enumerate(df['target'].value_counts()):
        ax.text(i, v + 5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'disease_distribution.png'), dpi=300)
    plt.close()
    print("[OK] disease_distribution.png")
    
    # Age distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='target', y='age', palette='Set2')
    plt.title('Age by Disease Status')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'age_by_disease.png'), dpi=300)
    plt.close()
    print("[OK] age_by_disease.png")

def generate_report(df, sql_results):
    """Generate report."""
    print("\n--- Report ---")
    
    report = f"""
{'='*60}
MILESTONE A: BUSINESS INSIGHTS PACK
Healthcare Analytics Report
{'='*60}

Dataset: Heart Disease
Patients: {len(df)}
Disease Rate: {(df['target']==1).mean()*100:.1f}%

--- KEY METRICS ---
Average Age: {df['age'].mean():.1f} years
Average Cholesterol: {df['chol'].mean():.1f} mg/dl
Average Blood Pressure: {df['trestbps'].mean():.1f} mm Hg

--- DISEASE BY GENDER ---
{sql_results['gender'].to_string(index=False)}

--- DISEASE BY AGE ---
{sql_results['age'].to_string(index=False)}

--- RECOMMENDATIONS ---
1. Focus screening on patients over 50
2. Monitor cholesterol levels regularly
3. Implement lifestyle interventions

{'='*60}
"""
    
    with open(os.path.join(OUTPUT_PATH, 'milestone_a_report.txt'), 'w') as f:
        f.write(report)
    
    print("[OK] milestone_a_report.txt")
    print(report)

def main():
    df = generate_data()
    sql_results = create_database(df)
    create_visualizations(df)
    generate_report(df, sql_results)
    
    print("\n" + "="*60)
    print("MILESTONE A COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
