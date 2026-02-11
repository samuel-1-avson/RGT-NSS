# Data Dictionary: Heart Disease Dataset

## Dataset Overview

**Source**: [Kaggle - Heart Disease UCI](https://www.kaggle.com/datasets/uciml/heart-disease-database)  
**Records**: 303 patients  
**Features**: 13 attributes + target  
**Purpose**: Predict presence of heart disease

## Column Descriptions

### Demographics

| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `age` | Integer | Age in years | 29-77 |
| `sex` | Binary | Gender | 1 = Male, 0 = Female |

### Clinical Measurements

| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `cp` | Categorical | Chest pain type | 0 = Typical angina<br>1 = Atypical angina<br>2 = Non-anginal pain<br>3 = Asymptomatic |
| `trestbps` | Integer | Resting blood pressure (mm Hg) | 94-200 |
| `chol` | Integer | Serum cholesterol (mg/dl) | 126-564 |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dl | 1 = True, 0 = False |
| `restecg` | Categorical | Resting ECG results | 0 = Normal<br>1 = ST-T abnormality<br>2 = LV hypertrophy |
| `thalach` | Integer | Maximum heart rate achieved | 71-202 |
| `exang` | Binary | Exercise induced angina | 1 = Yes, 0 = No |
| `oldpeak` | Float | ST depression induced by exercise | 0.0-6.2 |
| `slope` | Categorical | Slope of peak exercise ST segment | 0 = Upsloping<br>1 = Flat<br>2 = Downsloping |
| `ca` | Integer | Number of major vessels colored by fluoroscopy | 0-3 |
| `thal` | Categorical | Thalassemia | 0 = Normal<br>1 = Fixed defect<br>2 = Reversible defect |

### Target Variable

| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `target` | Binary | Heart disease presence | 0 = No disease<br>1 = Disease present |

## Data Quality Notes

- No missing values in the dataset
- All features are numeric (categorical encoded as integers)
- Target is balanced (~54% positive cases)
- Features are at different scales (require normalization for ML)

## Feature Categories

### Risk Factors
- Age
- Sex
- Cholesterol (chol)
- Blood Pressure (trestbps)
- Blood Sugar (fbs)

### Symptoms
- Chest Pain (cp)
- Exercise Angina (exang)

### Test Results
- ECG (restecg)
- Max Heart Rate (thalach)
- ST Depression (oldpeak)
- ST Slope (slope)
- Vessels (ca)
- Thalassemia (thal)

## Cleaning Applied

1. **No missing values** - Dataset complete
2. **Feature naming** - Standardized column names
3. **Type conversion** - All numeric types verified
4. **Outlier check** - Clinical values within expected ranges

## Usage Notes

- For SQL analysis: All fields can be used directly
- For ML: Consider feature scaling
- For visualization: Target distribution suitable for pie/bar charts
- For dashboard: All features can be used as filters
