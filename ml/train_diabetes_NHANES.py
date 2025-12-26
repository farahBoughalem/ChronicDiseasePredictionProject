from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
NHANES_DIR = BASE_DIR / "data" / "nhanes_2017_2018"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# NHANES files
DEMO_FILE = NHANES_DIR / "DEMO_J.xpt"
DIQ_FILE = NHANES_DIR / "DIQ_J.xpt"
GHB_FILE = NHANES_DIR / "GHB_J.xpt"
GLU_FILE = NHANES_DIR / "GLU_J.xpt"


# Load NHANES .xpt files
def read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport")

demo = pd.read_sas(DEMO_FILE, format="xport")
diq = pd.read_sas(DIQ_FILE, format="xport")
ghb = pd.read_sas(GHB_FILE, format="xport")
glu = pd.read_sas(GLU_FILE, format="xport")

# Make sure SEQN types match
for d in [demo, diq, ghb, glu]:
    if "SEQN" in d.columns:
        d["SEQN"] = pd.to_numeric(d["SEQN"], errors="coerce").astype("Int64")

# Merge on SEQN
df = demo.merge(diq, on="SEQN", how="left") \
    .merge(ghb, on="SEQN", how="left") \
    .merge(glu, on="SEQN", how="left")

# Filter: males only
# RIAGENDR: 1=Male, 2=Female
df = df[df["RIAGENDR"] == 1].copy()


# DIQ010 values (typical NHANES):
# 1 = Yes, 2 = No, 3 = Borderline, 7 = Refused, 9 = Don't know
# We keep only 1/2
df["DIQ010"] = pd.to_numeric(df["DIQ010"], errors="coerce")
df = df[df["DIQ010"].isin([1, 2])].copy()
df["Outcome"] = (df["DIQ010"] == 1).astype(int)

# =========================
# Features available from your 4 files
# - RIDAGEYR: age
# - LBXGH: HbA1c (from GHB_J)
# - LBXGLU: glucose (from GLU_J)
# =========================
FEATURES = ["RIDAGEYR", "LBXGH", "LBXGLU"]
TARGET = "Outcome"

# Ensure numeric
for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("Final dataset shape (males only, valid DIQ010):", df.shape)
print("Class distribution (Outcome):\n", y.value_counts(dropna=False))

# =========================
# Train/test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Preprocessing:
# - keep NaN, then impute median
# - scale numeric features
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), FEATURES)
    ]
)


# Helper to train/evaluate/save
def train_eval_save(name: str, model):
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    # predict proba for ROC-AUC
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print(f"{name} ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    out_path = ARTIFACTS_DIR / f"diabetes_nhanes_male_{name}.joblib"
    joblib.dump(pipe, out_path)
    print("Saved:", out_path)

    return auc


# Models
logreg = LogisticRegression(max_iter=500, class_weight="balanced")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
)


# Train all + summary
results = {}
results["logreg"] = train_eval_save("logreg", logreg)
results["rf"] = train_eval_save("rf", rf)
results["xgb"] = train_eval_save("xgb", xgb)

print("\n=================== SUMMARY (ROC-AUC) ===================")
for k, v in results.items():
    print(f"{k.upper():6s}: {v:.4f}")


# ============================================================
# rf ROC-AUC: 0.9088
#               precision    recall  f1-score   support
#
#            0       0.96      0.95      0.96       759
#            1       0.64      0.70      0.67        97
#
#     accuracy                           0.92       856
#    macro avg       0.80      0.83      0.81       856
# weighted avg       0.93      0.92      0.92       856
#
# Saved: C:\Users\WH213XR\Downloads\ChronicDiseasesPredictionProject1-master\ChronicDiseasesPredictionProject1-master\artifacts\diabetes_nhanes_male_rf.joblib
#
# ============================================================
# xgb ROC-AUC: 0.9391
#               precision    recall  f1-score   support
#
#            0       0.96      0.98      0.97       759
#            1       0.83      0.67      0.74        97
#
#     accuracy                           0.95       856
#    macro avg       0.90      0.83      0.86       856
# weighted avg       0.94      0.95      0.94       856
#
# Saved: C:\Users\WH213XR\Downloads\ChronicDiseasesPredictionProject1-master\ChronicDiseasesPredictionProject1-master\artifacts\diabetes_nhanes_male_xgb.joblib
#
# =================== SUMMARY (ROC-AUC) ===================
# LOGREG: 0.9392
# RF    : 0.9088
# XGB   : 0.9391
