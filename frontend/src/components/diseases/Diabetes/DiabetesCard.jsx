import {useMemo, useState, useEffect} from "react";
import DiabetesFormFemale from "./DiabetesFormFemale";
import DiabetesFormMale from "./DiabetesFormMale";

const FEMALE_INITIAL = {
    Pregnancies: "",
    Glucose: "",
    BloodPressure: "",
    SkinThickness: "",
    Insulin: "",
    Height: "",
    Weight: "",
    DiabetesPedigreeFunction: "",
    Age: "",
};

const MALE_INITIAL = {
    Age: "",
    GlycatedHemoglobin: "",
    FastingBloodGlucose: "",
};

const DIABETES_FEMALE_API_URL = "/api/diabetes/female/";
const DIABETES_MALE_API_URL = "/api/diabetes/male/";

function toNumberOrNull(v) {
    if (v === "" || v === null || v === undefined) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
}

export default function DiabetesCard() {
    const [gender, setGender] = useState("female");
    const [values, setValues] = useState(FEMALE_INITIAL);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    useEffect(() => {
        setResult(null);
        setError("");
        setLoading(false);
        setValues(gender === "female" ? FEMALE_INITIAL : MALE_INITIAL);
    }, [gender]);

    const onChange = (key, value) => {
        setValues((prev) => ({...prev, [key]: value}));
    };

    const bmi = useMemo(() => {
        if (gender !== "female") return null;

        const h = toNumberOrNull(values.Height);
        const w = toNumberOrNull(values.Weight);
        if (!h || !w) return null;

        const meters = h / 100;
        if (meters <= 0) return null;

        const b = w / (meters * meters);
        return Math.round(b * 10) / 10;
    }, [gender, values.Height, values.Weight]);

    const onSubmit = async (e) => {
        e.preventDefault();

        setLoading(true);
        setError("");
        setResult(null);

        try {
            let url = "";
            let payload = {};

            if (gender === "female") {
                if (bmi === null) {
                    setError("Please enter Height and Weight so BMI can be calculated.");
                    setLoading(false);
                    return;
                }

                const required = ["Glucose", "BloodPressure", "Age"];
                for (const k of required) {
                    if (toNumberOrNull(values[k]) === null) {
                        setError(`Please fill: ${k}`);
                        setLoading(false);
                        return;
                    }
                }

                url = DIABETES_FEMALE_API_URL;
                payload = {
                    Pregnancies: toNumberOrNull(values.Pregnancies) ?? 0,
                    Glucose: toNumberOrNull(values.Glucose),
                    BloodPressure: toNumberOrNull(values.BloodPressure),
                    SkinThickness: toNumberOrNull(values.SkinThickness) ?? 0,
                    Insulin: toNumberOrNull(values.Insulin) ?? 0,
                    BMI: bmi,
                    DiabetesPedigreeFunction:
                        toNumberOrNull(values.DiabetesPedigreeFunction) ?? 0.2,
                    Age: toNumberOrNull(values.Age),
                };
            } else {
                const required = ["Age", "GlycatedHemoglobin", "FastingBloodGlucose"];
                for (const k of required) {
                    if (toNumberOrNull(values[k]) === null) {
                        setError(`Please fill: ${k}`);
                        setLoading(false);
                        return;
                    }
                }

                url = DIABETES_MALE_API_URL;

                payload = {
                    RIDAGEYR: toNumberOrNull(values.Age),
                    LBXGH: toNumberOrNull(values.GlycatedHemoglobin),
                    LBXGLU: toNumberOrNull(values.FastingBloodGlucose),
                };
            }

            const res = await fetch(url, {

                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload),
            });

            const data = await res.json();
            if (!res.ok) {
                setError(typeof data === "string" ? data : JSON.stringify(data));
                return;
            }

            setResult(data);
        } catch (err) {
            setError(err?.message || "Something went wrong");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="card shadow-sm">
            <div className="card-body">
                {/* Gender */}
                <div className="row g-3 mb-3">
                    <div className="col-12 col-md-6">
                        <label className="form-label">Gender</label>
                        <div className="btn-group w-100" role="group" aria-label="Gender">
                            <input
                                type="radio"
                                className="btn-check"
                                name="gender"
                                id="female"
                                checked={gender === "female"}
                                onChange={() => setGender("female")}
                            />
                            <label className="btn btn-outline-primary" htmlFor="female">
                                Female
                            </label>

                            <input
                                type="radio"
                                className="btn-check"
                                name="gender"
                                id="male"
                                checked={gender === "male"}
                                onChange={() => setGender("male")}
                            />
                            <label className="btn btn-outline-primary" htmlFor="male">
                                Male
                            </label>
                        </div>
                    </div>
                </div>

                {gender === "female" ? (
                    <DiabetesFormFemale
                        values={values}
                        onChange={onChange}
                        onSubmit={onSubmit}
                        loading={loading}
                        computedBmi={bmi}
                    />
                ) : (
                    <DiabetesFormMale
                        values={values}
                        onChange={onChange}
                        onSubmit={onSubmit}
                        loading={loading}
                    />
                )}

                {error && (
                    <div className="alert alert-danger mt-3 mb-0" role="alert">
                        {error}
                    </div>
                )}

                {result && (
                    <div
                        className={`alert mt-3 mb-0 ${
                            result.risk_band === "high"
                                ? "alert-danger"
                                : result.risk_band === "medium"
                                    ? "alert-warning"
                                    : "alert-success"
                        }`}
                    >
                        <div>
                            <strong>Prediction:</strong>{" "}
                            {result.risk_band === "high"
                                ? "Higher risk"
                                : result.risk_band === "medium"
                                    ? "Medium risk"
                                    : "Low risk"}
                        </div>
                        <div>
                            <strong>Probability:</strong> {result.risk_percent}%
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}