import { useState } from "react";
import DiabetesCard from "./Diabetes/DiabetesCard";
import HypertensionCard from "./HypertensionCard";
import CardiovascularDiseaseCard from "./CardiovascularDiseaseCard";

export default function DiseaseTabs() {
    const [active, setActive] = useState("diabetes");

    return (
        <div className="card shadow">
            <div className="card-header p-0">
                <ul className="nav nav-tabs card-header-tabs px-2 pt-2">
                    <li className="nav-item">
                        <button className={`nav-link ${active === "diabetes" ? "active" : ""}`} onClick={() => setActive("diabetes")}>Diabetes</button>
                    </li>
                    <li className="nav-item">
                        <button className={`nav-link ${active === "hypertension" ? "active" : ""}`} onClick={() => setActive("hypertension")}>Hypertension</button>
                    </li>
                    <li className="nav-item">
                        <button className={`nav-link ${active === "cardiovascular disease" ? "active" : ""}`} onClick={() => setActive("cardiovascular disease")}>Cardiovascular Disease</button>
                    </li>
                </ul>
            </div>

            <div className="card-body">
                {active === "diabetes" && <DiabetesCard />}
                {active === "hypertension" && <HypertensionCard />}
                {active === "cardiovascular disease" && <CardiovascularDiseaseCard />}
            </div>
        </div>
    )
}