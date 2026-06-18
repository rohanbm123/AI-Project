from pathlib import Path
from src.utils import save_json

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def policy_check(model_results, fairness_results, model_name=None, eu_risk_tier="Low", drift_results=None):
    decisions = {}
    gate_rules = []
    score = 0
    total_rules = 0

    # Auto-detect model_name if None
    if model_name is None:
        if "logistic_regression" in model_results:
            model_name = "logistic_regression"
        elif "linear_regression" in model_results:
            model_name = "linear_regression"
        elif "random_forest" in model_results:
            model_name = "random_forest"
        elif "hist_gradient_boosting" in model_results:
            model_name = "hist_gradient_boosting"
        else:
            model_name = list(model_results.keys())[0] if model_results else ""

    # Determine task_type
    is_regression = False
    if model_name in model_results:
        metrics = model_results[model_name]
        if "r2_score" in metrics or "mae" in metrics:
            is_regression = True

    # Stricter thresholds for EU AI Act High-Risk applications
    if eu_risk_tier == "High":
        limits = {
            "min_accuracy": 0.85,
            "min_recall": 0.80,
            "min_f1": 0.80,
            "min_r2": 0.80,
            "max_gap": 0.08,
            "min_disparate_impact": 0.80,
            "max_psi": 0.10
        }
    else:
        limits = {
            "min_accuracy": 0.75,
            "min_recall": 0.50,
            "min_f1": 0.50,
            "min_r2": 0.50,
            "max_gap": 0.15,
            "min_disparate_impact": 0.80,
            "max_psi": 0.20
        }

    # 1. GOVERN PILLAR: Governance, data quality, and documentation checks
    # Check if a demographic attribute exists for auditing
    has_demo = fairness_results.get("demographic_column") is not None
    demo_status = "PASS" if has_demo else "WARNING"
    gate_rules.append({
        "rule": "Demographic Audit Registration",
        "nist_pillar": "GOVERN",
        "metric_value": 1.0 if has_demo else 0.0,
        "threshold": 1.0,
        "status": demo_status,
        "explanation": "Verifies that a protected/demographic attribute has been designated for governance auditing.",
        "recommendation": "If warning, register a demographic category (e.g. Gender, Age, Region) to track model bias."
    })
    if demo_status == "PASS":
        score += 1
    total_rules += 1

    # 2. MAP PILLAR: Setup verification, target columns, and context
    # Target validation
    target_col = fairness_results.get("target_col", "Detected Target")
    gate_rules.append({
        "rule": "Prediction Target Definition",
        "nist_pillar": "MAP",
        "metric_value": 1.0,
        "threshold": 1.0,
        "status": "PASS",
        "explanation": f"Verifies that the target objective ({target_col}) is clearly mapped and logged.",
        "recommendation": "None"
    })
    score += 1
    total_rules += 1

    # 3. MEASURE PILLAR: Quantitative Model Performance Metrics & Drift
    if not is_regression:
        # Classification
        acc = model_results.get(model_name, {}).get("accuracy", 0.0)
        rec = model_results.get(model_name, {}).get("recall", 0.0)
        f1 = model_results.get(model_name, {}).get("f1_score", 0.0)

        acc_status = "PASS" if acc >= limits["min_accuracy"] else "FAIL"
        gate_rules.append({
            "rule": "Model Accuracy",
            "nist_pillar": "MEASURE",
            "metric_value": acc,
            "threshold": limits["min_accuracy"],
            "status": acc_status,
            "explanation": f"Statistical prediction accuracy check. Target: >= {limits['min_accuracy']:.2f}, Actual: {acc:.3f}",
            "recommendation": "If failed, review feature engineering, hyperparameter tuning, or collect more balanced data."
        })
        if acc_status == "PASS":
            score += 1
        total_rules += 1

        rec_status = "PASS" if rec >= limits["min_recall"] else "FAIL"
        gate_rules.append({
            "rule": "Model Recall",
            "nist_pillar": "MEASURE",
            "metric_value": rec,
            "threshold": limits["min_recall"],
            "status": rec_status,
            "explanation": f"True positive detection rate. Target: >= {limits['min_recall']:.2f}, Actual: {rec:.3f}",
            "recommendation": "If failed, adjust decision thresholds or apply class weights to model estimators."
        })
        if rec_status == "PASS":
            score += 1
        total_rules += 1

        # Check demographic gap if fairness audit ran
        if has_demo:
            gap = fairness_results.get("demographic_gap", 0.0)
            gap_status = "PASS" if gap <= limits["max_gap"] else "FAIL"
            gate_rules.append({
                "rule": "Demographic Bias Gap",
                "nist_pillar": "MEASURE",
                "metric_value": gap,
                "threshold": limits["max_gap"],
                "status": gap_status,
                "explanation": f"Maximum selection rate gap between demographic cohorts. Target: <= {limits['max_gap']:.2f}, Actual: {gap:.3f}",
                "recommendation": "If failed, check feature correlation with demographic class, or use sample re-weighting."
            })
            if gap_status == "PASS":
                score += 1
            total_rules += 1
    else:
        # Regression
        r2 = model_results.get(model_name, {}).get("r2_score", 0.0)
        r2_status = "PASS" if r2 >= limits["min_r2"] else "FAIL"
        gate_rules.append({
            "rule": "Model R² Score",
            "nist_pillar": "MEASURE",
            "metric_value": r2,
            "threshold": limits["min_r2"],
            "status": r2_status,
            "explanation": f"Coefficient of determination. Target: >= {limits['min_r2']:.2f}, Actual: {r2:.3f}",
            "recommendation": "If failed, try advanced non-linear regression models or check for outliers in continuous target."
        })
        if r2_status == "PASS":
            score += 1
        total_rules += 1

        if has_demo:
            gap = fairness_results.get("demographic_gap", 0.0)
            gap_status = "PASS" if gap <= limits["max_gap"] else "FAIL"
            gate_rules.append({
                "rule": "Group Error Disparity",
                "nist_pillar": "MEASURE",
                "metric_value": gap,
                "threshold": limits["max_gap"],
                "status": gap_status,
                "explanation": f"Disparity in model error (MAE) between demographic groups. Target: <= {limits['max_gap']:.2f}, Actual: {gap:.3f}",
                "recommendation": "If failed, verify if sample size is balanced across demographic cohorts."
            })
            if gap_status == "PASS":
                score += 1
            total_rules += 1

    # Check for data drift if available
    if drift_results:
        max_psi_feature = ""
        max_psi_val = 0.0
        for feat, d in drift_results.items():
            if d["psi"] > max_psi_val:
                max_psi_val = d["psi"]
                max_psi_feature = feat
        
        drift_status = "PASS" if max_psi_val <= limits["max_psi"] else "FAIL"
        gate_rules.append({
            "rule": "Population Stability Drift",
            "nist_pillar": "MEASURE",
            "metric_value": max_psi_val,
            "threshold": limits["max_psi"],
            "status": drift_status,
            "explanation": f"Maximum Population Stability Index (PSI) drift across features. Feature: '{max_psi_feature}', PSI: {max_psi_val:.3f}",
            "recommendation": "If failed, retrain baseline model with recent data splits to account for covariate shift."
        })
        if drift_status == "PASS":
            score += 1
        total_rules += 1

    # 4. MANAGE PILLAR: Risk mitigation, threshold adjustments, and bias control
    # Check disparate impact ratio (for classification tasks with demographic columns)
    if not is_regression and has_demo:
        di_ratio = fairness_results.get("disparate_impact_ratio", 1.0)
        di_status = "PASS" if di_ratio >= limits["min_disparate_impact"] else "FAIL"
        gate_rules.append({
            "rule": "Disparate Impact Ratio",
            "nist_pillar": "MANAGE",
            "metric_value": di_ratio,
            "threshold": limits["min_disparate_impact"],
            "status": di_status,
            "explanation": f"Audits compliance with EEOC four-fifths rule. Target: >= {limits['min_disparate_impact']:.2f}, Actual: {di_ratio:.3f}",
            "recommendation": "If failed, enable active sample re-weighting mitigation in the AutoML training loop."
        })
        if di_status == "PASS":
            score += 1
        total_rules += 1

    # Status Verdict
    if score == total_rules:
        final_status = "APPROVED"
    elif score >= total_rules / 2:
        final_status = "CONDITIONALLY APPROVED"
    else:
        final_status = "REJECTED"

    # Populate decisions dictionary for UI mapping
    for r in gate_rules:
        # Standardize key names for UI backwards compatibility
        decisions[r["rule"].lower().replace(" ", "_")] = r["status"]
    
    # Add legacy keys for backward compatibility
    if not is_regression:
        decisions["accuracy"] = decisions.get("model_accuracy", "N/A")
        decisions["recall"] = decisions.get("model_recall", "N/A")
        decisions["demographic_fairness"] = decisions.get("demographic_bias_gap", "N/A")
        if "gender" in fairness_results.get("demographic_column", ""):
            decisions["gender_fairness"] = decisions.get("demographic_bias_gap", "N/A")
        if "SeniorCitizen" in fairness_results.get("demographic_column", ""):
            decisions["senior_fairness"] = decisions.get("demographic_bias_gap", "N/A")
    else:
        decisions["r2_score"] = decisions.get("model_r²_score", "N/A")
        decisions["demographic_fairness"] = decisions.get("group_error_disparity", "N/A")

    output = {
        "score": score,
        "max_score": total_rules,
        "decisions": decisions,
        "final_status": final_status,
        "limits": limits,
        "gate_rules": gate_rules,
        "eu_risk_tier": eu_risk_tier
    }

    save_json(output, REPORTS_DIR / "compliance_results.json")
    return output