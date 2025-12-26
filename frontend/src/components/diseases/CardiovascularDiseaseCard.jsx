import {useMemo, useState} from "react";

const INITIAL = {
    Age: "",
    TotalCholesterol: "",
    SystolicBP: "",
    DiastolicBP: "",
    Gender: "",
    Smoking: "",
    Diabetes: "",
};

const API_URL = "http://localhost:8000/api/cvd/";

function toNumberOrNull(v) {
    if (v === "" || v === null || v === undefined) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
}

function riskBand(riskPct) {
    if (riskPct < 20) return "low";
    if (riskPct < 50) return "medium";
    return "high";
}

function bandLabel(band) {
    if (band === "high") return "Higher risk";
    if (band === "medium") return "Medium risk";
    return "Low risk";
}

export default function CardiovascularDiseaseCard() {
    const [values, setValues] = useState(INITIAL);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    const onChange = (key, value) => {
        setValues((prev) => ({...prev, [key]: value}));
    };

    const payload = useMemo(() => {
        const age = toNumberOrNull(values.Age);
        const tc = toNumberOrNull(values.TotalCholesterol);
        const sbp = toNumberOrNull(values.SystolicBP);
        const dbp = toNumberOrNull(values.DiastolicBP);

        const sexMale =
            values.Gender === "male" ? 1 : values.Gender === "female" ? 0 : null;

        const everSmoker =
            values.Smoking === "yes" ? 1 : values.Smoking === "no" ? 0 : null;

        const diabetes =
            values.Diabetes === "yes" ? 1 : values.Diabetes === "no" ? 0 : null;

        return {
            RIDAGEYR: age,
            LBXTC: tc,
            SBP_mean: sbp,
            DBP_mean: dbp,
            Sex_Male: sexMale,
            EverSmoker: everSmoker,
            Diabetes: diabetes,
        };
    }, [values]);

    const onSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError("");
        setResult(null);

        try {
            const requiredNumeric = ["Age", "TotalCholesterol", "SystolicBP", "DiastolicBP"];
            for (const k of requiredNumeric) {
                if (toNumberOrNull(values[k]) === null) {
                    setError(`Please fill: ${k}`);
                    setLoading(false);
                    return;
                }
            }

            if (!values.Gender) {
                setError("Please select Gender.");
                setLoading(false);
                return;
            }
            if (!values.Smoking) {
                setError("Please select Ever smoking.");
                setLoading(false);
                return;
            }
            if (!values.Diabetes) {
                setError("Please select Diabetes.");
                setLoading(false);
                return;
            }

            const age = toNumberOrNull(values.Age);
            if (age < 0 || age > 120) {
                setError("Age looks invalid.");
                setLoading(false);
                return;
            }

            const tc = toNumberOrNull(values.TotalCholesterol);
            if (tc < 50 || tc > 600) {
                setError("Total cholesterol looks invalid.");
                setLoading(false);
                return;
            }

            const sbp = toNumberOrNull(values.SystolicBP);
            const dbp = toNumberOrNull(values.DiastolicBP);
            if (sbp < 50 || sbp > 300 || dbp < 30 || dbp > 200) {
                setError("Blood pressure values look invalid.");
                setLoading(false);
                return;
            }

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

            let final = data;
            if (final && typeof final.risk_probability === "number" && !("risk_percent" in final)) {
                const pct = Math.round(final.risk_probability * 1000) / 10;
                final = {
                    ...final,
                    risk_percent: pct,
                    risk_band: final.risk_band || riskBand(pct),
                };
            }

            setResult(final);
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

                        {/* Total Cholesterol */}
                        <div className="col-12">
                            <label className="form-label">Total Cholesterol (mg/dL)</label>
                            <input
                                className="form-control"
                                type="number"
                                value={values.TotalCholesterol}
                                onChange={(e) => onChange("TotalCholesterol", e.target.value)}
                                placeholder="e.g. 190"
                                min="0"
                                step="1"
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

                        {/* Gender bar */}
                        <div className="col-12">
                            <label className="form-label">Gender</label>
                            <div className="btn-group w-100" role="group" aria-label="Gender">
                                <button
                                    type="button"
                                    className={`btn ${values.Gender === "male" ? "btn-primary" : "btn-outline-primary"}`}
                                    onClick={() => onChange("Gender", "male")}
                                >
                                    Male
                                </button>
                                <button
                                    type="button"
                                    className={`btn ${values.Gender === "female" ? "btn-primary" : "btn-outline-primary"}`}
                                    onClick={() => onChange("Gender", "female")}
                                >
                                    Female
                                </button>
                            </div>
                        </div>

                        {/* Ever smoking bar */}
                        <div className="col-12">
                            <label className="form-label">Ever smoking</label>
                            <div className="btn-group w-100" role="group" aria-label="Ever smoking">
                                <button
                                    type="button"
                                    className={`btn ${values.Smoking === "yes" ? "btn-primary" : "btn-outline-primary"}`}
                                    onClick={() => onChange("Smoking", "yes")}
                                >
                                    Yes
                                </button>
                                <button
                                    type="button"
                                    className={`btn ${values.Smoking === "no" ? "btn-primary" : "btn-outline-primary"}`}
                                    onClick={() => onChange("Smoking", "no")}
                                >
                                    No
                                </button>
                            </div>
                        </div>

                        {/* Diabetes bar */}
                        <div className="col-12">
                            <label className="form-label">Diabetes</label>
                            <div className="btn-group w-100" role="group" aria-label="Diabetes">
                                <button
                                    type="button"
                                    className={`btn ${values.Diabetes === "yes" ? "btn-primary" : "btn-outline-primary"}`}
                                    onClick={() => onChange("Diabetes", "yes")}
                                >
                                    Yes
                                </button>
                                <button
                                    type="button"
                                    className={`btn ${values.Diabetes === "no" ? "btn-primary" : "btn-outline-primary"}`}
                                    onClick={() => onChange("Diabetes", "no")}
                                >
                                    No
                                </button>
                            </div>
                        </div>

                        <div className="col-12 mt-3">
                            <button className="btn btn-primary w-100" type="submit" disabled={loading}>
                                {loading ? "Predicting..." : "Predict CVD Risk"}
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
                            <strong>Prediction:</strong> {bandLabel(result.risk_band)}
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