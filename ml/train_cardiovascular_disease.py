from pathlib import Path
import joblib
import numpy as np
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
BPX_FILE = NHANES_DIR / "BPX_J.xpt"
TCHOL_FILE = NHANES_DIR / "TCHOL_J.xpt"
SMQ_FILE = NHANES_DIR / "SMQ_J.xpt"
DIQ_FILE = NHANES_DIR / "DIQ_J.xpt"
MCQ_FILE = NHANES_DIR / "MCQ_J.xpt"


def read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport")


demo = read_xpt(DEMO_FILE)
bpx = read_xpt(BPX_FILE)
tchol = read_xpt(TCHOL_FILE)
smq = read_xpt(SMQ_FILE)
diq = read_xpt(DIQ_FILE)
mcq = read_xpt(MCQ_FILE)

# Ensure SEQN matches
for d in [demo, bpx, tchol, smq, diq, mcq]:
    if "SEQN" in d.columns:
        d["SEQN"] = pd.to_numeric(d["SEQN"], errors="coerce").astype("Int64")

df = (
    demo.merge(bpx, on="SEQN", how="left")
    .merge(tchol, on="SEQN", how="left")
    .merge(smq, on="SEQN", how="left")
    .merge(diq, on="SEQN", how="left")
    .merge(mcq, on="SEQN", how="left")
).copy()


# =========================
# Clean special NHANES codes -> NaN for questionnaire items
# Common codes:
# 7 = Refused, 9 = Don't know
# Sometimes 77/99 appear in some variables (less common in these)
# =========================
def clean_q_yesno(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan})
    return s


# =========================
# 1) Outcome label: CVD from MCQ
# MCQ160C = coronary heart disease
# MCQ160E = heart attack
# MCQ160F = stroke
# Usually: 1=Yes, 2=No
# We'll define Outcome=1 if any of these is Yes.
# Keep only rows where outcome can be determined.
# =========================

for col in ["MCQ160C", "MCQ160E", "MCQ160F"]:
    if col not in df.columns:
        raise KeyError(f"{col} not found in MCQ file columns.")
    df[col] = clean_q_yesno(df[col])

# Determine outcome only where we have at least one valid answer
valid_outcome_mask = df[["MCQ160C", "MCQ160E", "MCQ160F"]].notna().any(axis=1)
df = df[valid_outcome_mask].copy()

# Outcome: yes if any == 1
df["Outcome"] = ((df["MCQ160C"] == 1) | (df["MCQ160E"] == 1) | (df["MCQ160F"] == 1)).astype(int)

# =========================
# 2) Features
# Age/Gender from DEMO:
# RIDAGEYR = Age (years)
# RIAGENDR = Gender (1=Male, 2=Female)
# encoding gender as: male=1, female=0
# =========================
if "RIDAGEYR" not in df.columns or "RIAGENDR" not in df.columns:
    raise KeyError("RIDAGEYR or RIAGENDR not found in DEMO file.")

df["RIDAGEYR"] = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
df["RIAGENDR"] = pd.to_numeric(df["RIAGENDR"], errors="coerce")
df["Sex_Male"] = (df["RIAGENDR"] == 1).astype("float")

# Total cholesterol from TCHOL:
# LBXTC = Total cholesterol (mg/dL)
if "LBXTC" not in df.columns:
    raise KeyError("LBXTC not found in TCHOL file.")

df["LBXTC"] = pd.to_numeric(df["LBXTC"], errors="coerce")

# Blood pressure from BPX:
# Typical columns: BPXSY1..BPXSY4 and BPXDI1..BPXDI4
sys_cols = [c for c in df.columns if c.startswith("BPXSY")]
dia_cols = [c for c in df.columns if c.startswith("BPXDI")]

if not sys_cols or not dia_cols:
    raise KeyError("BPXSY* or BPXDI* columns not found in BPX file.")

for c in sys_cols + dia_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["SBP_mean"] = df[sys_cols].mean(axis=1)
df["DBP_mean"] = df[dia_cols].mean(axis=1)

# Smoking from SMQ:
# SMQ020: Smoked at least 100 cigarettes in life (1=Yes, 2=No)
# Encoding: EverSmoker = 1 if yes, 0 if no, NaN otherwise
if "SMQ020" not in df.columns:
    raise KeyError("SMQ020 not found in SMQ file.")

df["SMQ020"] = clean_q_yesno(df["SMQ020"])
df["EverSmoker"] = np.where(df["SMQ020"] == 1, 1.0, np.where(df["SMQ020"] == 2, 0.0, np.nan))

# Diabetes from DIQ:
# DIQ010: Doctor told you have diabetes? (1=Yes, 2=No)
if "DIQ010" not in df.columns:
    raise KeyError("DIQ010 not found in DIQ file.")

df["DIQ010"] = clean_q_yesno(df["DIQ010"])
df["Diabetes"] = np.where(df["DIQ010"] == 1, 1.0, np.where(df["DIQ010"] == 2, 0.0, np.nan))

# Final feature set

FEATURES_CONT = ["RIDAGEYR", "LBXTC", "SBP_mean", "DBP_mean"]
FEATURES_BIN = ["Sex_Male", "EverSmoker", "Diabetes"]
FEATURES = FEATURES_CONT + FEATURES_BIN

TARGET = "Outcome"

# Ensure numeric
for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

print("\nFinal dataset shape:", df.shape)
print("Outcome distribution:\n", y.value_counts(dropna=False))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing
# - Continuous: median impute + scale
# - Binary: most_frequent impute (or constant 0)


preprocess = ColumnTransformer(transformers=[
    ("cont", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
     FEATURES_CONT), ("bin", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ]), FEATURES_BIN), ],
    remainder="drop")


# Helper: train/eval/save
def train_eval_save(name: str, model):
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    print("\n" + "=" * 60)
    print(f"{name} ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    out_path = ARTIFACTS_DIR / f"cvd_nhanes_{name}.joblib"
    joblib.dump(pipe, out_path)
    print("Saved:", out_path)
    return auc


# Models
logreg = LogisticRegression(max_iter=800, class_weight="balanced")
rf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")

xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss"
)

# Train all + summary
results = {}
results["logreg"] = train_eval_save("logreg", logreg)
results["rf"] = train_eval_save("rf", rf)
results["xgb"] = train_eval_save("xgb", xgb)

print("\n=================== SUMMARY (ROC-AUC) ===================")

for k, v in results.items():
    print(f"{k.upper():6s}: {v:.4f}")


# Final dataset shape: (5568, 238)
# Outcome distribution:
#  Outcome
# 0    4996
# 1     572
# Name: count, dtype: int64
#
# ============================================================
# logreg ROC-AUC: 0.7980
#               precision    recall  f1-score   support
#
#            0       0.96      0.70      0.81      1000
#            1       0.22      0.72      0.33       114
#
#     accuracy                           0.71      1114
#    macro avg       0.59      0.71      0.57      1114
# weighted avg       0.88      0.71      0.76      1114
#
# Saved: C:\Users\WH213XR\Downloads\ChronicDiseasesPredictionProject1-master\ChronicDiseasesPredictionProject1-master\artifacts\cvd_nhanes_logreg.joblib
#
# ============================================================
# rf ROC-AUC: 0.7639
#               precision    recall  f1-score   support
#
#            0       0.90      0.98      0.94      1000
#            1       0.21      0.05      0.08       114
#
#     accuracy                           0.88      1114
#    macro avg       0.55      0.51      0.51      1114
# weighted avg       0.83      0.88      0.85      1114
#
# Saved: C:\Users\WH213XR\Downloads\ChronicDiseasesPredictionProject1-master\ChronicDiseasesPredictionProject1-master\artifacts\cvd_nhanes_rf.joblib
#
# ============================================================
# xgb ROC-AUC: 0.7721
#               precision    recall  f1-score   support
#
#            0       0.90      0.98      0.94      1000
#            1       0.09      0.02      0.03       114
#
#     accuracy                           0.88      1114
#    macro avg       0.49      0.50      0.48      1114
# weighted avg       0.81      0.88      0.84      1114
#
# Saved: C:\Users\WH213XR\Downloads\ChronicDiseasesPredictionProject1-master\ChronicDiseasesPredictionProject1-master\artifacts\cvd_nhanes_xgb.joblib
#
# =================== SUMMARY (ROC-AUC) ===================
# LOGREG: 0.7980
# RF    : 0.7639
# XGB   : 0.7721