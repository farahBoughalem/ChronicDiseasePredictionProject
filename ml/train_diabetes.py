from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "diabetes_pima_indians.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# STEP 1: Load the dataset into Pandas framework
df = pd.read_csv(DATA_PATH)

# Separate features and target
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

print("--Full Dataset: diabetic vs non-diabetic proportion")
print(y.value_counts(normalize=True)*100) # 0s: 65.10% (majority-class), 1s: 34.89%(minority-class)

# STEP 2: Train/Test split to keep the same proportion of 1 and 0 in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("\n--Training set: diabetic vs non-diabetic proportion")
print(y_train.value_counts(normalize=True)*100) # 0S: 65.14%, 1s: 34.8%
print("\n--Test set: diabetic vs non-diabetic proportion")
print(y_test.value_counts(normalize=True)*100) # 0s: 64.93%, 1s: 35.06%

# STEP 3: Preprocessing - numeric values scaling
preprocess = ColumnTransformer(transformers=[("scale", StandardScaler(), FEATURES)])

# STEP 4: Defining the ML models
logreg_model = LogisticRegression(
    max_iter=300,
    class_weight="balanced"
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

# helper function
def train_evaluate_save(model_name: str, model_obj):
    print(f"Training: {model_name}")

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model_obj)
        ]
    )

    # STEP 5: Model training
    pipe.fit(X_train, y_train)

    # STEP 6: Evaluate the model
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    print(f"{model_name} ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    # STEP 7: Save Pipeline
    model_path = ARTIFACTS_DIR / f"diabetes_{model_name}.joblib"
    joblib.dump(pipe, model_path)
    print(f"Saved: {model_path}")

    return auc


# STEP 5: Models training, evaluating and saving the pipelines
results = {"logreg": train_evaluate_save("logreg", logreg_model),
           "rf": train_evaluate_save("rf", rf_model),
           "xgb": train_evaluate_save("xgb", xgb_model)}

# STEP 6: Printing the results
print("\n=================== SUMMARY ===================")
for name, auc in results.items():
    print(f"{name.upper()} ROC-AUC: {auc:.4f}")

# Training: logreg
# logreg ROC-AUC: 0.8248
#               precision    recall  f1-score   support
#
#            0       0.82      0.75      0.79       100
#            1       0.60      0.70      0.65        54
#
#     accuracy                           0.73       154
#    macro avg       0.71      0.73      0.72       154
# weighted avg       0.75      0.73      0.74       154
#
# Saved: C:\Users\Farah\PycharmProjects\ChronicDiseasesPredictionProject\artifacts\diabetes_logreg.joblib
# Training: rf
# rf ROC-AUC: 0.8268
#               precision    recall  f1-score   support
#
#            0       0.81      0.85      0.83       100
#            1       0.69      0.63      0.66        54
#
#     accuracy                           0.77       154
#    macro avg       0.75      0.74      0.74       154
# weighted avg       0.77      0.77      0.77       154
#
# Saved: C:\Users\Farah\PycharmProjects\ChronicDiseasesPredictionProject\artifacts\diabetes_rf.joblib
# Training: xgb
# xgb ROC-AUC: 0.8319
#               precision    recall  f1-score   support
#
#            0       0.82      0.79      0.81       100
#            1       0.64      0.69      0.66        54
#
#     accuracy                           0.75       154
#    macro avg       0.73      0.74      0.73       154
# weighted avg       0.76      0.75      0.76       154
#
# Saved: C:\Users\Farah\PycharmProjects\ChronicDiseasesPredictionProject\artifacts\diabetes_xgb.joblib
#
# =================== SUMMARY ===================
# LOGREG ROC-AUC: 0.8248
# RF ROC-AUC: 0.8268
# XGB ROC-AUC: 0.8319




