from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
NHANES_DIR = BASE_DIR / "data" / "nhanes_2017_2018"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

DEMO_FILE = NHANES_DIR / "DEMO_J.xpt"
BMX_FILE = NHANES_DIR / "BMX_J.xpt"
BPQ_FILE = NHANES_DIR / "BPQ_J.xpt"
BPX_FILE = NHANES_DIR / "BPX_J.xpt"


def read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport")


demo = read_xpt(DEMO_FILE)
bmx = read_xpt(BMX_FILE)
bpq = read_xpt(BPQ_FILE)
bpx = read_xpt(BPX_FILE)

# Ensure SEQN matches
for d in [demo, bmx, bpq, bpx]:
    if "SEQN" in d.columns:
        d["SEQN"] = pd.to_numeric(d["SEQN"], errors="coerce").astype("Int64")

df = (
demo.merge(bmx, on="SEQN", how="left")
.merge(bpq, on="SEQN", how="left")
.merge(bpx, on="SEQN", how="left")
)

# BP means (optional: keep for analysis, but DO NOT use in target)
systolic_cols = [c for c in df.columns if c.startswith("BPXSY")]
diastolic_cols = [c for c in df.columns if c.startswith("BPXDI")]

for c in systolic_cols + diastolic_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["SBP_mean"] = df[systolic_cols].mean(axis=1, skipna=True)
df["DBP_mean"] = df[diastolic_cols].mean(axis=1, skipna=True)

# -------------------------
# Target (NO LEAKAGE):
# "Diagnosed hypertension" (self-report OR meds)
# -------------------------
df["BPQ020"] = pd.to_numeric(df.get("BPQ020"), errors="coerce") # told high BP? 1 yes, 2 no
df["BPQ040A"] = pd.to_numeric(df.get("BPQ040A"), errors="coerce") # taking meds? 1 yes, 2 no

self_report_htn = (df["BPQ020"] == 1)
meds_htn = (df["BPQ040A"] == 1)

df["Outcome"] = (self_report_htn | meds_htn).astype(int)

# Keep only rows where target is known (answered at least one of these)
df = df[
df["BPQ020"].isin([1, 2]) | df["BPQ040A"].isin([1, 2])
].copy()

# -------------------------
# Features (risk-like; no BP)
# -------------------------
df["RIDAGEYR"] = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
df["BMXBMI"] = pd.to_numeric(df["BMXBMI"], errors="coerce")
df["RIAGENDR"] = pd.to_numeric(df["RIAGENDR"], errors="coerce") # 1 male, 2 female
df["RIDRETH3"] = pd.to_numeric(df.get("RIDRETH3"), errors="coerce")

features_num = ["RIDAGEYR", "BMXBMI"]
features_cat = ["RIAGENDR", "RIDRETH3"]

FEATURES = features_num + features_cat
TARGET = "Outcome"

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("Final dataset shape:", df.shape)
print("Class distribution:\n", y.value_counts(dropna=False))

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, stratify=y, random_state=42
)

preprocess = ColumnTransformer(
transformers=[
("num", Pipeline(steps=[
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler()),
]), features_num),
("cat", Pipeline(steps=[
("imputer", SimpleImputer(strategy="most_frequent")),
("onehot", OneHotEncoder(handle_unknown="ignore")),
]), features_cat),
]
)

def train_eval_save(name: str, model):
    pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
    ])

    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    print("\n" + "=" * 60)
    print(f"{name} ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    out_path = ARTIFACTS_DIR / f"hypertension_diagnosed_{name}.joblib"
    joblib.dump(pipe, out_path)
    print("Saved:", out_path)
    return auc

logreg = LogisticRegression(max_iter=800, class_weight="balanced")
rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
xgb = XGBClassifier(
n_estimators=600,
learning_rate=0.05,
max_depth=4,
subsample=0.9,
colsample_bytree=0.9,
random_state=42,
eval_metric="logloss",
)

results = {}
results["logreg"] = train_eval_save("logreg", logreg)
results["rf"] = train_eval_save("rf", rf)
results["xgb"] = train_eval_save("xgb", xgb)

print("\n=================== SUMMARY (ROC-AUC) ===================")
for k, v in results.items():
    print(f"{k.upper():6s}: {v:.4f}")

# Final dataset shape: (6151, 99)
# Class distribution:
#  Outcome
# 0    4014
# 1    2137
# Name: count, dtype: int64
#
# ============================================================
# logreg ROC-AUC: 0.8214
#               precision    recall  f1-score   support
#
#            0       0.85      0.73      0.79       803
#            1       0.60      0.76      0.67       428
#
#     accuracy                           0.74      1231
#    macro avg       0.73      0.75      0.73      1231
# weighted avg       0.76      0.74      0.75      1231
#
# Saved: C:\Users\WH213XR\Downloads\ChronicDiseasesPredictionProject1-master\ChronicDiseasesPredictionProject1-master\artifacts\hypertension_diagnosed_logreg.joblib
#
# ============================================================
# rf ROC-AUC: 0.7875
#               precision    recall  f1-score   support
#
#            0       0.78      0.82      0.80       803
#            1       0.63      0.58      0.60       428
#
#     accuracy                           0.73      1231
#    macro avg       0.71      0.70      0.70      1231
# weighted avg       0.73      0.73      0.73      1231
#
# Saved: C:\Users\WH213XR\Downloads\ChronicDiseasesPredictionProject1-master\ChronicDiseasesPredictionProject1-master\artifacts\hypertension_diagnosed_rf.joblib
#
# ============================================================
# xgb ROC-AUC: 0.8103
#               precision    recall  f1-score   support
#
#            0       0.79      0.83      0.81       803
#            1       0.65      0.57      0.61       428
#
#     accuracy                           0.74      1231
#    macro avg       0.72      0.70      0.71      1231
# weighted avg       0.74      0.74      0.74      1231
#
# Saved: C:\Users\WH213XR\Downloads\ChronicDiseasesPredictionProject1-master\ChronicDiseasesPredictionProject1-master\artifacts\hypertension_diagnosed_xgb.joblib
#
# =================== SUMMARY (ROC-AUC) ===================
# LOGREG: 0.8214
# RF    : 0.7875
# XGB   : 0.8103
