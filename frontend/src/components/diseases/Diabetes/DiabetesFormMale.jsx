function Field({label, value, onChange, placeholder = "", type = "number", min, step}) {
    return (
        <div className="col-12 mb-3">
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

export default function DiabetesFormMale({values, onChange, onSubmit, loading}) {

    return (
        <form onSubmit={onSubmit}>
            <div className="row g-3">
                <Field
                    label="Age"
                    value={values.Age}
                    onChange={(v) => onChange("Age", v)}
                    placeholder="e.g. 34"
                    min="0"
                    step="1"
                />

                <Field
                    label="Glycated Hemoglobin (%)"
                    value={values.GlycatedHemoglobin}
                    onChange={(v) => onChange("GlycatedHemoglobin", v)}
                    placeholder="e.g. 6"
                    min="0"
                    step="1"
                />

                <Field
                    label="Fasting blood glucose (mg/dL)"
                    value={values.FastingBloodGlucose}
                    onChange={(v) => onChange("FastingBloodGlucose", v)}
                    placeholder="e.g. 72"
                    min="0"
                    step="1"
                />

                <div className="col-12">
                    <button className="btn btn-primary w-100" type="submit" disabled={loading}>
                        {loading ? "Predicting..." : "Predict Diabetes Risk"}
                    </button>
                </div>
            </div>
        </form>
    );
}
