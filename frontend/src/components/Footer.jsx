import "../styles/footer.css";

export default function Footer() {
    const year = new Date().getFullYear();

    return (
        <footer className="app-footer mt-5">
            <div className="container py-4">
                <div className="row gy-3">
                    <div className="col-12 col-md-6">
                        <h6 className="footer-title">Chronic Disease Risk Prediction</h6>
                        <p className="footer-text">
                            AI-assisted risk estimation tools based on population data (NHANES & Pima).
                            These tools are intended for educational and research purposes only.
                        </p>
                    </div>

                    <div className="col-12 col-md-6 text-md-end">
                        <p className="footer-text mb-1">
                            © {year} Chronic Diseases Prediction
                        </p>
                        <p className="footer-text small">
                            Not a medical diagnosis · Consult a healthcare professional
                        </p>
                    </div>
                </div>
            </div>
        </footer>
    );
}