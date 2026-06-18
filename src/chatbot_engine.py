import os
import json
import urllib.request

def call_gemini_api(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode("utf-8"), 
        headers=headers, 
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error contacting Gemini API: {e}"

def generate_governance_explanation(user_query: str, policy_results_json: dict) -> str:
    """
    Backend router that accepts a user query and a context dictionary of policy run results,
    returning a markdown explanation. Uses Gemini API if GEMINI_API_KEY is present;
    otherwise falls back to an intelligent, context-aware local query analyzer.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        prompt = f"""You are the Anchor AI Compliance Assistant. You have access to the underlying pipeline audit JSON: {json.dumps(policy_results_json, indent=2)}.

Your behavior must adapt dynamically to the user's input:
- IF the user asks a foundational ML or compliance question (e.g., "What is accuracy?"), give a clear executive definition, reference their current dashboard metrics, and stop there.
- IF the user asks for a structural diagnostic (e.g., "Why is the score low?" or "What failed?"), drill directly into the specific failing fields inside the JSON data.
- Maintain an conversational, professional auditor tone. Never dump the entire boilerplate framework matrix unless asked to 'Provide a full framework audit summary'.

User Query: "{user_query}"
"""
        return call_gemini_api(prompt, api_key)
    
    # ── Context-Aware Local Compliance Auditor Fallback ──
    q = user_query.lower()
    
    # Extract details from JSON for easy dynamic tie-backs
    gate_rules = policy_results_json.get("gate_rules", [])
    limits = policy_results_json.get("limits", {})
    final_status = policy_results_json.get("final_status", "UNKNOWN")
    score = policy_results_json.get("score", 0)
    max_score = policy_results_json.get("max_score", 4)
    
    # Find accuracy metric values
    acc_rule = next((r for r in gate_rules if "accuracy" in r.get("rule", "").lower()), None)
    rec_rule = next((r for r in gate_rules if "recall" in r.get("rule", "").lower()), None)
    f1_rule = next((r for r in gate_rules if "f1" in r.get("rule", "").lower()), None)
    r2_rule = next((r for r in gate_rules if "r²" in r.get("rule", "").lower() or "r2" in r.get("rule", "").lower()), None)
    drift_rule = next((r for r in gate_rules if "drift" in r.get("rule", "").lower() or "psi" in r.get("rule", "").lower()), None)
    bias_rule = next((r for r in gate_rules if "disparate" in r.get("rule", "").lower() or "bias" in r.get("rule", "").lower()), None)

    # 1. ACCURACY QUERY
    if "accuracy" in q:
        definition = "**Model Accuracy** is the ratio of correctly predicted instances to the total instances in the evaluation dataset."
        if acc_rule:
            val = acc_rule.get("metric_value", 0.0)
            thresh = acc_rule.get("threshold", 0.75)
            status = acc_rule.get("status", "FAIL")
            return f"{definition}\n\nIn your current model, accuracy is at **{val * 100:.1f}%**, which **{ 'passes' if status == 'PASS' else 'fails' }** your configured threshold of **{thresh * 100:.0f}%**."
        elif r2_rule:
            val = r2_rule.get("metric_value", 0.0)
            thresh = r2_rule.get("threshold", 0.50)
            status = r2_rule.get("status", "FAIL")
            return f"This is a regression task. Instead of classification accuracy, we track the **R² Score** (Coefficient of Determination), which measures the proportion of target variance explained by features.\n\nIn your current model, the R² score is **{val:.4f}**, which **{ 'passes' if status == 'PASS' else 'fails' }** your configured threshold of **{thresh:.2f}**."
        return f"{definition}\n\nNo active accuracy check was registered in this pipeline execution."

    # 2. RECALL QUERY
    elif "recall" in q or "sensitivity" in q:
        definition = "**Model Recall** (or sensitivity) measures the proportion of actual positives that were correctly identified by the model."
        if rec_rule:
            val = rec_rule.get("metric_value", 0.0)
            thresh = rec_rule.get("threshold", 0.50)
            status = rec_rule.get("status", "FAIL")
            return f"{definition}\n\nIn your current model, recall is at **{val * 100:.1f}%**, which **{ 'passes' if status == 'PASS' else 'fails' }** your configured threshold of **{thresh * 100:.0f}%**."
        return f"{definition}\n\nNo active recall check was registered in this pipeline execution."

    # 3. PRECISION QUERY
    elif "precision" in q:
        definition = "**Model Precision** measures the proportion of positive identifications that were actually correct (reducing false alarms)."
        # Precision is part of F1, tie it back to metrics if F1 is available
        if f1_rule:
            return f"{definition}\n\nPrecision and Recall are balanced via your F1 score. Your current model's F1 score is **{f1_rule.get('metric_value', 0.0) * 100:.1f}%** against a threshold of **{f1_rule.get('threshold', 0.50) * 100:.0f}%**."
        return f"{definition}\n\nPrecision details are monitored via the Model Performance tab."

    # 4. F1 SCORE QUERY
    elif "f1" in q:
        definition = "The **F1 Score** is the harmonic mean of precision and recall, providing a balanced metric for uneven class distributions."
        if f1_rule:
            val = f1_rule.get("metric_value", 0.0)
            thresh = f1_rule.get("threshold", 0.50)
            status = f1_rule.get("status", "FAIL")
            return f"{definition}\n\nIn your current model, the F1 Score is **{val * 100:.1f}%**, which **{ 'passes' if status == 'PASS' else 'fails' }** your configured threshold of **{thresh * 100:.0f}%**."
        return f"{definition}\n\nNo active F1 check was registered in this pipeline execution."

    # 5. SHAP / EXPLAINABILITY QUERY
    elif "shap" in q or "explain" in q or "feature importance" in q:
        return "**SHAP** (SHapley Additive exPlanations) is a game-theoretic approach that explains individual predictions by calculating the contribution of each feature to the final output.\n\nIn your current model, you can view the top global contributors under the **Explainability** tab, which lists features ranked by mean absolute SHAP values."

    # 6. DRIFT QUERY
    elif "drift" in q or "psi" in q or "stability" in q:
        definition = "**Data Drift** represents a change in feature distributions over time. We calculate it using the **Population Stability Index (PSI)**."
        if drift_rule:
            val = drift_rule.get("metric_value", 0.0)
            thresh = drift_rule.get("threshold", 0.20)
            status = drift_rule.get("status", "FAIL")
            status_text = "stable (no significant drift)" if status == "PASS" else "drifted (requires retraining)"
            return f"{definition}\n\nYour maximum PSI drift metric is **{val:.4f}** (Threshold: `<= {thresh:.2f}`). Feature distributions are currently **{status_text}**."
        return f"{definition}\n\nNo active data drift checking was registered in this pipeline execution."

    # 7. BIAS / FAIRNESS QUERY
    elif "bias" in q or "fair" in q or "disparate" in q:
        definition = "**Disparate Impact** audits selection rate gaps between sensitive groups, enforcing the EEOC's four-fifths rule."
        if bias_rule:
            val = bias_rule.get("metric_value", 1.0)
            thresh = bias_rule.get("threshold", 0.80)
            status = bias_rule.get("status", "FAIL")
            status_text = "compliant" if status == "PASS" else "non-compliant"
            return f"{definition}\n\nYour current Disparate Impact Ratio is **{val:.4f}** (Threshold: `>= {thresh:.2f}`). Your demographic selection is currently **{status_text}**."
        return f"{definition}\n\nNo active fairness check was registered in this pipeline execution."

    # 8. DIAGNOSTIC QUERY ("Why is score low", "What failed")
    elif any(x in q for x in ["why", "low", "failed", "fail", "broken", "gate", "policy"]):
        failed_gates = [g for g in gate_rules if g.get("status") == "FAIL"]
        warn_gates = [g for g in gate_rules if g.get("status") == "WARNING"]
        
        if not failed_gates and not warn_gates:
            return f"Your governance score is high (**{score} / {max_score}** passed) because all policy gates met their threshold checks. The verdict is **{final_status}**."
        
        res = "### 🔍 Diagnostic Auditor Findings\n\n"
        if failed_gates:
            res += "The following specific checks **failed**:\n"
            for g in failed_gates:
                res += f"- **{g.get('rule')}** ({g.get('nist_pillar')}): Actual value `{g.get('metric_value'):.4f}` violated the limit threshold of `{g.get('threshold'):.2f}`.\n"
                res += f"  *Remediation:* {g.get('recommendation')}\n\n"
        if warn_gates:
            res += "The following warning indicators were flagged:\n"
            for g in warn_gates:
                res += f"- **{g.get('rule')}** ({g.get('nist_pillar')}): {g.get('explanation')}\n"
                res += f"  *Remediation:* {g.get('recommendation')}\n\n"
        return res

    # 9. EXPLICIT FULL SUMMARY QUERY
    elif "full" in q or "summary" in q or "overview" in q or "matrix" in q:
        res = f"### ⚓ Anchor AI Compliance Ledger Summary\n\n"
        res += f"**Verdict:** {final_status} ({score} of {max_score} rules passed)\n"
        res += f"**Risk Tier:** EU AI Act {policy_results_json.get('eu_risk_tier', 'Low')}-Risk sector\n\n"
        res += "#### NIST AI RMF 1.0 Pillar Summary:\n"
        res += "- **GOVERN:** Schema verification and column exclusions.\n"
        res += "- **MAP:** Context registration and risk category assignment.\n"
        res += "- **MEASURE:** ML validation metrics and PSI feature drift.\n"
        res += "- **MANAGE:** Group fairness selection rates and re-weighting mitigation.\n"
        return res

    # 10. DEFAULT AUDITOR CONVERSATIONAL RESPONDER
    else:
        return (
            "Hi, I'm the Anchor AI compliance assistant. Ask me questions about model parameters, "
            "such as:\n"
            "- *'What is accuracy?'* or *'What is F1 score?'* (ML Definitions)\n"
            "- *'Why is the score low?'* or *'What failed?'* (Structural diagnostics)\n"
            "- *'Provide a full framework audit summary'* (Full NIST map overview)"
        )
