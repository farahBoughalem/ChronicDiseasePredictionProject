import {useMemo, useState} from "react";

const INITIAL = {
    Age: "",
    Height: "",
    Weight: "",
    SystolicBP: "",
    DiastolicBP: "",
    Gender: "",
    Race: "",
};

const API_URL = "/api/hypertension/";

function toNumberOrNull(v) {
    if (v === "" || v === null || v === undefined) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
}

export default function HypertensionCard() {
    const [values, setValues] = useState(INITIAL);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    const onChange = (key, value) => {
        setValues((prev) => ({...prev, [key]: value}));
    };

    const bmi = useMemo(() => {
        const h = toNumberOrNull(values.Height);
        const w = toNumberOrNull(values.Weight);
        if (!h || !w) return null;

        const meters = h / 100;
        if (meters <= 0) return null;

        const b = w / (meters * meters);
        return Math.round(b * 10) / 10;
    }, [values.Height, values.Weight]);

    const onSubmit = async (e) => {
        e.preventDefault();

        setLoading(true);
        setError("");
        setResult(null);

        try {
            const required = ["Age", "SystolicBP", "DiastolicBP"];
            for (const k of required) {
                if (toNumberOrNull(values[k]) === null) {
                    setError(`Please fill: ${k}`);
                    setLoading(false);
                    return;
                }
            }

            if (bmi === null) {
                setError("Please enter Height and Weight so BMI can be calculated.");
                setLoading(false);
                return;
            }

            if (!values.Gender) {
                setError("Please select Gender.");
                setLoading(false);
                return;
            }

            if (!values.Race) {
                setError("Please select Race/Ethnicity.");
                setLoading(false);
                return;
            }

            const payload = {
                RIDAGEYR: toNumberOrNull(values.Age),
                BMXBMI: bmi,
                SBP_mean: toNumberOrNull(values.SystolicBP),
                DBP_mean: toNumberOrNull(values.DiastolicBP),
                RIAGENDR: values.Gender === "male" ? 1 : 2,
                RIDRETH3: toNumberOrNull(values.Race),
            };

            const res = await fetch(API_URL, {
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
                <form onSubmit={onSubmit}>
                    <div className="row g-3">
                        {/* Age */}
                        <div className="col-12">
                            <label className="form-label">Age</label>
                            <input
                                className="form-control"
                                type="number"
                                value={values.Age}
                                onChange={(e) => onChange("Age", e.target.value)}
                                placeholder="e.g. 45"
                                min="0"
                                step="1"
                            />
                        </div>

                        {/* Height / Weight (BMI computed) */}
                        <div className="col-12 col-md-6">
                            <label className="form-label">Height (cm)</label>
                            <input
                                className="form-control"
                                type="number"
                                value={values.Height}
                                onChange={(e) => onChange("Height", e.target.value)}
                                placeholder="e.g. 165"
                                min="0"
                                step="1"
                            />
                        </div>

                        <div className="col-12 col-md-6">
                            <label className="form-label">Weight (kg)</label>
                            <input
                                className="form-control"
                                type="number"
                                value={values.Weight}
                                onChange={(e) => onChange("Weight", e.target.value)}
                                placeholder="e.g. 65"
                                min="0"
                                step="0.1"
                            />
                        </div>

                        {/* BP */}
                        <div className="col-12 col-md-6">
                            <label className="form-label">Systolic Blood Pressure (mmHg)</label>
                            <input
                                className="form-control"
                                type="number"
                                value={values.SystolicBP}
                                onChange={(e) => onChange("SystolicBP", e.target.value)}
                                placeholder="e.g. 120"
                                min="0"
                                step="1"
                            />
                        </div>

                        <div className="col-12 col-md-6">
                            <label className="form-label">Diastolic Blood Pressure (mmHg)</label>
                            <input
                                className="form-control"
                                type="number"
                                value={values.DiastolicBP}
                                onChange={(e) => onChange("DiastolicBP", e.target.value)}
                                placeholder="e.g. 80"
                                min="0"
                                step="1"
                            />
                        </div>

                        <div className="col-12">
                            <label className="form-label">Gender</label>
                            <div className="btn-group w-100" role="group" aria-label="Gender">
                                <button type="button"
                                        className={`btn ${values.Gender === "male" ? "btn-primary" : "btn-outline-primary"}`}
                                        onClick={() => onChange("Gender", "male")}>Male
                                </button>
                                <button type="button"
                                        className={`btn ${values.Gender === "female" ? "btn-primary" : "btn-outline-primary"}`}
                                        onClick={() => onChange("Gender", "female")}>Female
                                </button>
                            </div>
                        </div>

                        {/* Race / Ethnicity */}
                        <div className="col-12">
                            <label className="form-label">Race / Ethnicity</label>

                            <div className="race-grid">
                                {[
                                    {label: "White", value: "3"},
                                    {label: "Black", value: "4"},
                                    {label: "Asian", value: "6"},
                                    {label: "Mexican American", value: "1"},
                                    {label: "Other Hispanic", value: "2"},
                                    {label: "Other / Mixed", value: "7"},
                                ].map((opt) => (
                                    <button
                                        key={opt.value}
                                        type="button"
                                        className={`race-btn ${
                                            values.Race === opt.value ? "active" : ""
                                        }`}
                                        onClick={() => onChange("Race", opt.value)}
                                    >
                                        {opt.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="col-12 mt-3">
                            <button className="btn btn-primary w-100" type="submit" disabled={loading}>
                                {loading ? "Predicting..." : "Predict Hypertension Risk"}
                            </button>
                        </div>
                    </div>
                </form>

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
                        {"risk_percent" in result && (
                            <div>
                                <strong>Probability:</strong> {result.risk_percent}%
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}