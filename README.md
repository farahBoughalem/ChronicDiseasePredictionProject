# DEVELOPING AN AI MACHINE LEARNING MODEL FOR EARLY DETECTION OF CHRONIC DISEASES USING PATIENT DATA: A WEB-BASED IMPLEMENTATION

## 📌 Overview
This project implements a **machine-learning–based clinical decision support system** for predicting major chronic diseases using publicly available health survey data.

The system currently supports:
- **Diabetes**
- **Hypertension**
- **Cardiovascular Disease (CVD)**

It follows a complete end-to-end pipeline:
1. Data ingestion and preprocessing
2. Machine learning model training and evaluation
3. Model persistence using `joblib`
4. Deployment via a **Django API**
5. Frontend-ready API consumption (React)

This project was developed as part of a **Engineering’s Thesis** and emphasizes **reproducibility, interpretability, and clinical relevance**.

---

## 🧠 Diseases & Data Sources

### 1️⃣ Diabetes
- **Female model**
- Dataset: Pima Indians Diabetes Dataset (Kaggle)
- **Male model**
- Dataset: NHANES
- Features:
- Age (`RIDAGEYR`)
- HbA1c (`LBXGH`)
- Fasting plasma glucose (`LBXGLU`)

### 2️⃣ Hypertension
- Dataset: NHANES
- Single unified model (men + women)
- Features include:
- Age
- Sex
- Systolic & diastolic blood pressure
- Body Mass Index (BMI)

### 3️⃣ Cardiovascular Disease (CVD)
- Dataset: NHANES
- Composite outcome:
- Coronary heart disease
- Heart attack
- Stroke
- Features:
- Age
- Sex
- Blood pressure
- Total cholesterol
- Smoking status
- Diabetes status

---

## 🧪 Machine Learning Models

For each disease, the following models were trained and compared:
- Logistic Regression
- Random Forest
- XGBoost

Evaluation metrics:
- ROC-AUC (primary)
- Precision, Recall, F1-score

> Logistic Regression was preferred for deployment in some cases due to better sensitivity and interpretability, which is critical in healthcare.

---

## 🏗️ Project Structure

ChronicDiseasesPredictionProject/
│
├── artifacts/ # Trained models (.joblib)
│
├── ml/
│ ├── train_diabetes_female.py
│ ├── train_diabetes_nhanes_male.py
│ ├── train_hypertension_nhanes.py
│ └── train_cvd_nhanes.py
│
├── predictor/ # Django app
│ ├── views.py # API logic
│ ├── urls.py
│ └── __init__.py
│
├── manage.py
├── requirements.txt
└── README.md

---

## 🚀 Installation & Usage

### Install dependencies
pip install -r requirements.txt

### Train a model (example)
python ml/train_cvd_nhanes.py

Trained models are saved to:
artifacts/

---

## 🌐 Django API

### Start the server
python manage.py runserver

### API Endpoints

- /api/diabetes/female/ — Diabetes prediction (female)
- /api/diabetes/male/ — Diabetes prediction (male, NHANES)
- /api/hypertension/ — Hypertension prediction
- /api/cvd/ — Cardiovascular disease prediction

### Example Request (CVD)
POST /api/cvd/
Content-Type: application/json

{
"RIDAGEYR": 55,
"Sex_Male": 1,
"LBXTC": 220,
"SBP_mean": 145,
"DBP_mean": 90,
"EverSmoker": 1,
"Diabetes": 0
}

### Example Response
{
"inputs": {
"RIDAGEYR": 55,
"Sex_Male": 1,
"LBXTC": 220,
"SBP_mean": 145,
"DBP_mean": 90,
"EverSmoker": 1,
"Diabetes": 0
},
"probability": 0.63,
"risk_percent": 63.0,
"risk_band": "high"
}

---

## ⚠️ Important Notes
- Missing values (NaN) are intentionally preserved and handled via SimpleImputer inside ML pipelines.
- Accuracy is not used as the main metric due to class imbalance; ROC-AUC and recall are preferred.
- This system is intended as a clinical decision-support tool, not a diagnostic device.

---

## 📚 Academic Context
- Data source: NHANES (CDC)
- Methodology aligned with recent literature on chronic disease prediction using machine learning
- Focus on interpretability, reproducibility, and clinical relevance

---

## 🔮 Future Work
- Fairness analysis (sex-specific performance)
- Threshold optimization
- Additional chronic diseases (e.g., CKD)
- Authentication and prediction history storage


---

## 👨‍🎓 Author
Developed as part of an Engineering’s Thesis in Computer Engineering - AI specialization