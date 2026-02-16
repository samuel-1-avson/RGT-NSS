# Data Dictionary: Heart Disease Dataset

This dictionary describes the clinical features and target variable used in the Heart Disease UCI dataset analysis.

| Column         | Type  | Description                                               | Values / Range                                                         |
| :------------- | :---- | :-------------------------------------------------------- | :--------------------------------------------------------------------- |
| **patient_id** | int   | Unique identifier for each patient record.                | 1 - 500                                                                |
| **age**        | int   | Patient age in years.                                     | 29 - 77                                                                |
| **sex**        | int   | Biological sex of the patient.                            | 0 = Female, 1 = Male                                                   |
| **cp**         | int   | Chest pain type experienced.                              | 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic |
| **trestbps**   | int   | Resting blood pressure (on admission to the hospital).    | 94 - 200 mm Hg                                                         |
| **chol**       | int   | Serum cholesterol level.                                  | 126 - 564 mg/dl                                                        |
| **fbs**        | int   | Fasting blood sugar > 120 mg/dl.                          | 0 = False, 1 = True                                                    |
| **restecg**    | int   | Resting electrocardiographic results.                     | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy   |
| **thalach**    | int   | Maximum heart rate achieved during exercise.              | 71 - 202 bpm                                                           |
| **exang**      | int   | Exercise-induced angina.                                  | 0 = No, 1 = Yes                                                        |
| **oldpeak**    | float | ST depression induced by exercise relative to rest.       | 0.0 - 6.2                                                              |
| **slope**      | int   | The slope of the peak exercise ST segment.                | 0: Upsloping, 1: Flat, 2: Downsloping                                  |
| **ca**         | int   | Number of major vessels colored by flourosopy.            | 0 - 3                                                                  |
| **thal**       | int   | Thalassemia blood disorder status.                        | 0: Normal, 1: Fixed defect, 2: Reversable defect                       |
| **target**     | int   | Diagnosis of heart disease (angiographic disease status). | 0: < 50% diameter narrowing, 1: > 50% diameter narrowing               |
