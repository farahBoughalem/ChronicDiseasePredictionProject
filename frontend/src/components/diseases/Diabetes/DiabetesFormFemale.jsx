import {useMemo, useState} from "react";

function Field({label, value, onChange, placeholder = "", type = "number", min, step}) {
    return (
        <div className="col-12 col-md-6">
            <label className="form-label">{label}</label>
            <input
                className="form-control"
                type={type}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                min={min}
                step={step}
            />
        </div>
    );
}

export default function DiabetesFormFemale({values, onChange, onSubmit, loading, computedBmi}) {
    const [showExactSkin, setShowExactSkin] = useState(false);
    const [showExactDpf, setShowExactDpf] = useState(false);

    const skinChoices = useMemo(
        () => [
            {label: "I don’t know", value: "0"},
            {label: "Very lean", value: "10"},
            {label: "Average", value: "20"},
            {label: "More body fat", value: "30"},
        ],
        []
    );

    const dpfChoices = useMemo(
        () => [
            {label: "No", value: "0.20"},
            {label: "Yes, distant relatives", value: "0.45"},
            {label: "Yes, parent or sibling", value: "0.70"},
            {label: "Yes, multiple close relatives", value: "1.10"},
        ],
        []
    );

    return (
        <form onSubmit={onSubmit}>
            <div className="row g-3">
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

                <Field
                    label="Pregnancies"
                    value={values.Pregnancies}
                    onChange={(v) => onChange("Pregnancies", v)}
                    placeholder="e.g. 2"
                    min="0"
                    step="1"
                />

                <Field
                    label="Glucose (mg/dL)"
                    value={values.Glucose}
                    onChange={(v) => onChange("Glucose", v)}
                    placeholder="e.g. 120"
                    min="0"
                    step="1"
                />

                <Field
                    label="Blood Pressure (diastolic) (mmHg)"
                    value={values.BloodPressure}
                    onChange={(v) => onChange("BloodPressure", v)}
                    placeholder="e.g. 70"
                    min="0"
                    step="1"
                />

                <Field
                    label="Insulin (2-hour) (mg/dL)"
                    value={values.Insulin}
                    onChange={(v) => onChange("Insulin", v)}
                    placeholder="e.g. 80 (or 0 if unknown)"
                    min="0"
                    step="1"
                />

                <Field
                    label="Height (cm)"
                    value={values.Height}
                    onChange={(v) => onChange("Height", v)}
                    placeholder="e.g. 165"
                    min="0"
                    step="0.1"
                />

                <Field
                    label="Weight (kg)"
                    value={values.Weight}
                    onChange={(v) => onChange("Weight", v)}
                    placeholder="e.g. 65"
                    min="0"
                    step="0.1"
                />


                {/* SkinThickness */}
                <div className="col-12 mt-2">
                    <div className="p-3 border rounded">
                        <div className="mb-2">
                            <div className="fw-semibold">How would you describe your body fat level?</div>
                            <div className="text-muted small">
                                This replaces “SkinThickness” (triceps skinfold in mm) with simple options.
                            </div>
                        </div>

                        <div className="btn-group w-100 d-flex" role="group" aria-label="SkinThickness choices">
                            {skinChoices.map((c) => (
                                <button
                                    key={c.value}
                                    type="button"
                                    className={`btn ${
                                        values.SkinThickness === c.value ? "btn-primary" : "btn-outline-primary"
                                    } flex-fill`}
                                    onClick={() => {
                                        onChange("SkinThickness", c.value);
                                        setShowExactSkin(false);
                                    }}
                                >
                                    {c.label}
                                </button>
                            ))}
                        </div>

                        <div className="mt-3">
                            <button
                                type="button"
                                className="btn btn-sm btn-link p-0"
                                onClick={() => setShowExactSkin((s) => !s)}
                            >
                                {showExactSkin ? "Hide exact measurement" : "I have an exact measurement (mm)"}
                            </button>
                        </div>

                        {showExactSkin && (
                            <div className="mt-2">
                                <label className="form-label">SkinThickness (mm)</label>
                                <input
                                    className="form-control"
                                    type="number"
                                    value={values.SkinThickness}
                                    onChange={(e) => onChange("SkinThickness", e.target.value)}
                                    placeholder="e.g. 20"
                                    min="0"
                                    step="0.1"
                                />
                                <div className="text-muted small mt-1">
                                    If unknown, you can keep it 0.
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Diabetes Pedigree Function */}
                <div className="col-12 mt-3">
                    <div className="p-3 border rounded">
                        <div className="mb-2">
                            <div className="fw-semibold">Do any of your close relatives have diabetes?</div>
                            <div className="text-muted small">
                                This approximates the “Diabetes Pedigree Function” feature used by the model.
                            </div>
                        </div>

                        <div className="btn-group w-100 d-flex" role="group" aria-label="DPF choices">
                            {dpfChoices.map((opt) => (
                                <button
                                    key={opt.value}
                                    type="button"
                                    className={`btn ${
                                        values.DiabetesPedigreeFunction === opt.value ? "btn-primary" : "btn-outline-primary"
                                    } flex-fill`}
                                    onClick={() => {
                                        onChange("DiabetesPedigreeFunction", opt.value);
                                        setShowExactDpf(false);
                                    }}
                                >
                                    {opt.label}
                                </button>
                            ))}
                        </div>

                        <div className="mt-3">
                            <button
                                type="button"
                                className="btn btn-sm btn-link p-0"
                                onClick={() => setShowExactDpf((s) => !s)}
                            >
                                {showExactDpf ? "Hide exact value" : "I want to enter an exact value"}
                            </button>
                        </div>

                        {showExactDpf && (
                            <div className="mt-2">
                                <label className="form-label">Diabetes Pedigree Function</label>
                                <input
                                    className="form-control"
                                    type="number"
                                    value={values.DiabetesPedigreeFunction}
                                    onChange={(e) => onChange("DiabetesPedigreeFunction", e.target.value)}
                                    placeholder="e.g. 0.47"
                                    min="0"
                                    step="0.01"
                                />
                                <div className="text-muted small mt-1">
                                    If you don’t know, choose an option above.
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                <div className="col-12">
                    <button className="btn btn-primary w-100" type="submit" disabled={loading}>
                        {loading ? "Predicting..." : "Predict Diabetes Risk"}
                    </button>
                </div>
            </div>
        </form>
    );
}
