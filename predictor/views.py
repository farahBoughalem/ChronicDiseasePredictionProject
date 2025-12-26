import json
from pathlib import Path

import joblib
import pandas as pd
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt

BASE_DIR = Path(__file__).resolve().parents[1]

# female diabetes model
FEMALE_DIABETES_MODEL_PATH = BASE_DIR / "artifacts" / "diabetes_xgb.joblib"
female_diabetes_model = joblib.load(FEMALE_DIABETES_MODEL_PATH)
FEMALE_DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# male diabetes model
MALE_DIABETES_MODEL_PATH = BASE_DIR / "artifacts" / "diabetes_nhanes_male_logreg.joblib"
male_diabetes_model = joblib.load(MALE_DIABETES_MODEL_PATH)
MALE_DIABETES_FEATURES = ["RIDAGEYR", "LBXGH", "LBXGLU"]

# hypertension model
HYPERTENSION_MODEL_PATH = BASE_DIR / "artifacts" / "hypertension_diagnosed_logreg.joblib"
hypertension_model = joblib.load(HYPERTENSION_MODEL_PATH)
HYPERTENSION_FEATURES = ["RIDAGEYR", "BMXBMI", "SBP_mean", "DBP_mean", "RIAGENDR", "RIDRETH3"]

# cardiovascular disease
CVD_MODEL_PATH = BASE_DIR / "artifacts" / "cvd_nhanes_logreg.joblib"
cvd_model = joblib.load(CVD_MODEL_PATH)
CVD_FEATURES = ["RIDAGEYR", "LBXTC", "SBP_mean", "DBP_mean", "Sex_Male", "EverSmoker", "Diabetes"]


def _parse_json(request: HttpRequest):
    try:
        body = request.body.decode("utf-8")
        return json.loads(body), None
    except json.JSONDecodeError:
        return None, JsonResponse({"error": "Invalid JSON"}, status=400)


def _validate_numeric_fields(data: dict, features: list[str], allow_null: bool):
    errors = []
    values = {}

    for feature in features:
        if feature not in data:
            errors.append(f"Missing field: {feature}")
            continue

        v = data[feature]
        if allow_null and v is None:
            values[feature] = None
            continue

        try:
            values[feature] = float(v)
        except (ValueError, TypeError):
            errors.append(f"Field {feature} must be a number{' or null' if allow_null else ''}.")
    return values, errors


def _risk_band(risk_pct: float) -> str:
    if risk_pct < 20:
        return "low"
    if risk_pct < 50:
        return "medium"
    return "high"


def _predict_proba_1(model, df: pd.DataFrame) -> float:
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support predict_proba().")
    proba = model.predict_proba(df)[0, 1]
    return float(proba)


# ---------- API VIEWS ----------

@csrf_exempt
def api_predict_diabetes_female(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    data, err = _parse_json(request)
    if err:
        return err

    values, errors = _validate_numeric_fields(data, FEMALE_DIABETES_FEATURES, allow_null=False)
    if errors:
        return JsonResponse({"error": "Invalid input", "details": errors}, status=400)

    df = pd.DataFrame([values], columns=FEMALE_DIABETES_FEATURES)
    proba = _predict_proba_1(female_diabetes_model, df)

    risk_pct = round(proba * 100, 1)
    band = _risk_band(risk_pct)

    return JsonResponse(
        {
            "inputs": values,
            "risk_probability": proba,
            "risk_percent": risk_pct,
            "risk_band": band,
        }
    )


@csrf_exempt
def api_predict_diabetes_male(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    data, err = _parse_json(request)
    if err:
        return err

    values, errors = _validate_numeric_fields(data, MALE_DIABETES_FEATURES, allow_null=True)
    if errors:
        return JsonResponse({"error": "Invalid input", "details": errors}, status=400)

    df = pd.DataFrame([values], columns=MALE_DIABETES_FEATURES)
    proba = _predict_proba_1(male_diabetes_model, df)

    risk_pct = round(proba * 100, 1)
    band = _risk_band(risk_pct)

    return JsonResponse(
        {
            "inputs": values,
            "risk_probability": proba,
            "risk_percent": risk_pct,
            "risk_band": band,
        }
    )


@csrf_exempt
def api_predict_hypertension(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    data, err = _parse_json(request)
    if err:
        return err

    values, errors = _validate_numeric_fields(data, HYPERTENSION_FEATURES, allow_null=True)
    if errors:
        return JsonResponse({"error": "Invalid input", "details": errors}, status=400)

    df = pd.DataFrame([values], columns=HYPERTENSION_FEATURES)
    proba = _predict_proba_1(hypertension_model, df)

    risk_pct = round(proba * 100, 1)
    band = _risk_band(risk_pct)

    return JsonResponse(
        {
            "inputs": values,
            "risk_probability": proba,
            "risk_percent": risk_pct,
            "risk_band": band,
        }
    )

@csrf_exempt
def api_predict_cvd(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    data, err = _parse_json(request)
    if err:
        return err

    values, errors = _validate_numeric_fields(data, CVD_FEATURES, allow_null=True)
    if errors:
        return JsonResponse({"error": "Invalid input", "details": errors}, status=400)

    df = pd.DataFrame([values], columns=CVD_FEATURES)
    proba = _predict_proba_1(cvd_model, df)

    risk_pct = round(proba * 100, 1)
    band = _risk_band(risk_pct)

    return JsonResponse(
        {
            "inputs": values,
            "risk_probability": proba,
            "risk_percent": risk_pct,
            "risk_band": band,
        }
    )