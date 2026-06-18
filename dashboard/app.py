import sys
import textwrap
from pathlib import Path

# Add project root to python path to resolve 'src' imports when run via Streamlit
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.utils import load_json
from src.auto_pipeline import AutoMLPipeline
from src.pdf_generator import generate_pdf_report

# ---------- Force non-interactive matplotlib backend ----------
matplotlib.use("Agg")

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

# ---------- Page config ----------
st.set_page_config(
    page_title="Anchor AI",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

def dark_residuals_plot(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor("#0c1222")
    ax.set_facecolor("#0c1222")
    ax.scatter(y_test, y_pred, color="rgba(129, 140, 248, 0.6)", edgecolors="rgba(99, 102, 241, 0.8)", alpha=0.7, s=25)
    min_val = min(float(y_test.min()), float(y_pred.min()))
    max_val = max(float(y_test.max()), float(y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=1.5, label="Perfect Fit")
    ax.set_title("Actual vs. Predicted", fontsize=14, color="#e2e8f0", fontweight=800, pad=14)
    ax.set_xlabel("Actual Values", fontsize=11, color="#94a3b8", fontweight=600)
    ax.set_ylabel("Predicted Values", fontsize=11, color="#94a3b8", fontweight=600)
    ax.tick_params(colors="#94a3b8", labelsize=10)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=9)
    ax.grid(color="#1e293b", linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    return fig

# ---------- Load data ----------
if "automl_pipeline" not in st.session_state:
    serialized_path = MODELS_DIR / "serialized_pipeline.joblib"
    if serialized_path.exists():
        try:
            saved_data = joblib.load(serialized_path)
            pipeline = AutoMLPipeline(
                target_col=saved_data.get("target_col"),
                demographic_col=saved_data.get("demographic_col"),
                task_type_override=saved_data.get("task_type")
            )
            pipeline.task_type = saved_data.get("task_type")
            pipeline.features_schema = saved_data.get("features_schema", {})
            pipeline.numeric_cols = saved_data.get("numeric_cols", [])
            pipeline.categorical_cols = saved_data.get("categorical_cols", [])
            pipeline.feature_columns = saved_data.get("feature_columns", [])
            pipeline.linear_pipeline = saved_data.get("linear")
            pipeline.ensemble_pipeline = saved_data.get("ensemble")
            pipeline.metrics = saved_data.get("metrics", {})
            pipeline.fairness_report = saved_data.get("fairness_report", {})
            pipeline.drift_report = saved_data.get("drift_report", {})
            pipeline.explainability_report = saved_data.get("explainability_report", {})
            pipeline.compliance_report = saved_data.get("compliance_report", {})
            pipeline.data_quality = saved_data.get("data_quality", {})
            pipeline.metadata = saved_data.get("metadata", {})
            
            pipeline.X_train = saved_data.get("X_train")
            pipeline.X_test = saved_data.get("X_test")
            pipeline.y_train = saved_data.get("y_train")
            pipeline.y_test = saved_data.get("y_test")
            
            st.session_state["automl_pipeline"] = pipeline
            st.session_state["is_custom_run"] = True
            st.session_state["custom_filename"] = pipeline.metadata.get("filename", "Customer-Churn.csv")
        except Exception as e:
            pass

model_metrics = load_json(REPORTS_DIR / "model_metrics.json")
quality_report = load_json(REPORTS_DIR / "quality_report.json")
fairness_report = load_json(REPORTS_DIR / "fairness_report.json")
explainability_report = load_json(REPORTS_DIR / "explainability_report.json")
compliance_report = load_json(REPORTS_DIR / "compliance_results.json")
metadata_report = load_json(BASE_DIR / "metadata" / "dataset_catalog.json")
lineage_log = load_json(BASE_DIR / "metadata" / "lineage_log.json", default=[])
shap_img = REPORTS_DIR / "shap_summary.png"

# ---------- AutoML Session State Override ----------
is_custom = st.session_state.get("is_custom_run", False) and "automl_pipeline" in st.session_state

if is_custom:
    pipeline = st.session_state["automl_pipeline"]
    model_metrics = pipeline.metrics
    fairness_report = pipeline.fairness_report
    explainability_report = pipeline.explainability_report
    compliance_report = pipeline.compliance_report
    
    total_cells = pipeline.data_quality["total_rows"] * pipeline.data_quality["total_columns"]
    missing_pct = pipeline.data_quality["total_missing_values"] / max(1, total_cells)
    quality_score = int((1.0 - missing_pct) * 100)
    if pipeline.data_quality["duplicate_rows"] > 0:
        quality_score = max(10, quality_score - 15)
        
    quality_report = {
        "num_rows": pipeline.data_quality["total_rows"],
        "num_columns": pipeline.data_quality["total_columns"],
        "total_missing_values": pipeline.data_quality["total_missing_values"],
        "duplicate_rows": pipeline.data_quality["duplicate_rows"],
        "quality_score": quality_score,
        "status": "PASS" if quality_score >= 80 else "FAIL"
    }
    
    metadata_report = {
        "dataset_name": st.session_state.get("custom_filename", "Custom Dataset"),
        "num_rows": pipeline.data_quality["total_rows"],
        "num_columns": pipeline.data_quality["total_columns"],
        "target_column": pipeline.target_col,
        "owner": "Current Session User",
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if pipeline.metadata.get("data_only", False):
        lineage_log = [
            {"step": "ingestion", "status": "COMPLETED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": f"Loaded CSV file: {st.session_state.get('custom_filename')}"},
            {"step": "data_quality", "status": "COMPLETED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": f"Inspected: missing={pipeline.data_quality['total_missing_values']}, duplicates={pipeline.data_quality['duplicate_rows']}"},
            {"step": "metadata", "status": "COMPLETED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": f"Registered target: {pipeline.target_col}"},
            {"step": "preprocessing", "status": "SKIPPED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": "Skipped (AI Pipeline disabled)"},
            {"step": "model_training", "status": "SKIPPED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": "Skipped (AI Pipeline disabled)"},
            {"step": "fairness_check", "status": "SKIPPED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": "Skipped (AI Pipeline disabled)"},
            {"step": "explainability", "status": "SKIPPED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": "Skipped (AI Pipeline disabled)"},
            {"step": "policy_engine", "status": "SKIPPED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": "Skipped (AI Pipeline disabled)"}
        ]
    else:
        lineage_log = [
            {"step": "ingestion", "status": "COMPLETED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": f"Loaded CSV file: {st.session_state.get('custom_filename')}"},
            {"step": "data_quality", "status": "COMPLETED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": f"Inspected: missing={pipeline.data_quality['total_missing_values']}, duplicates={pipeline.data_quality['duplicate_rows']}"},
            {"step": "metadata", "status": "COMPLETED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": f"Registered target: {pipeline.target_col}"},
            {"step": "preprocessing", "status": "COMPLETED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": f"Encoded categoricals. Scaled {len(pipeline.numeric_cols)} numeric columns."},
            {"step": "model_training", "status": "COMPLETED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": f"Trained linear & forest models for {pipeline.task_type}."},
            {"step": "fairness_check", "status": "COMPLETED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": f"Audited demographic gap on column: {pipeline.demographic_col}"},
            {"step": "explainability", "status": "COMPLETED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": "Extracted coefficients & SHAP values"},
            {"step": "policy_engine", "status": "COMPLETED", "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "details": f"Final Governance decision computed."}
        ]

# ---------- Derived values ----------
if is_custom and pipeline.metadata.get("data_only", False):
    final_status = "DATA HEALTH ONLY"
else:
    final_status = compliance_report.get("final_status", "N/A") if compliance_report else "N/A"
score = compliance_report.get("score", 0) if compliance_report else 0
max_score = compliance_report.get("max_score", 4) if compliance_report else 4
decisions = compliance_report.get("decisions", {}) if compliance_report else {}

if is_custom:
    if pipeline.task_type == "classification":
        log_model = model_metrics.get("logistic_regression", {})
        rf_model = model_metrics.get("random_forest", {})
    else:
        log_model = model_metrics.get("linear_regression", {})
        rf_model = model_metrics.get("random_forest", {})
else:
    log_model = model_metrics.get("logistic_regression", {}) if model_metrics else {}
    rf_model = model_metrics.get("random_forest", {}) if model_metrics else {}

# Get only the latest pipeline run from lineage (last 8 steps)
PIPELINE_STEPS = ["ingestion", "data_quality", "metadata", "preprocessing",
                  "model_training", "fairness_check", "explainability", "policy_engine"]
latest_lineage = lineage_log[-len(PIPELINE_STEPS):] if lineage_log else []


# ---------- Sandbox Data Loading ----------
@st.cache_data
def load_sandbox_data():
    import pandas as pd
    try:
        X_test = pd.read_csv(BASE_DIR / "data" / "processed" / "X_test.csv", index_col=0)
        y_test = pd.read_csv(BASE_DIR / "data" / "processed" / "y_test.csv", index_col=0).squeeze()
        raw_df = pd.read_csv(BASE_DIR / "data" / "raw" / "Customer-Churn.csv")
        gender_test = raw_df.loc[X_test.index, "gender"]
        senior_test = raw_df.loc[X_test.index, "SeniorCitizen"]
        return X_test, y_test, gender_test, senior_test
    except Exception as e:
        st.error(f"Error loading sandbox data: {e}")
        return None, None, None, None


def predict_single_customer(inputs, selected_model_name):
    import joblib
    from pathlib import Path
    from src.auto_pipeline import AutoMLPipeline
    
    serialized_path = Path(__file__).resolve().parent.parent / "models" / "serialized_pipeline.joblib"
    if not serialized_path.exists():
        return 0.0, [{"name": "error", "label": "No serialized model found. Please train a pipeline.", "contribution": 0.0}]
        
    try:
        saved_data = joblib.load(serialized_path)
        pipeline = AutoMLPipeline(
            target_col=saved_data.get("target_col"),
            demographic_col=saved_data.get("demographic_col"),
            task_type_override=saved_data.get("task_type")
        )
        pipeline.task_type = saved_data.get("task_type")
        pipeline.features_schema = saved_data.get("features_schema", {})
        pipeline.numeric_cols = saved_data.get("numeric_cols", [])
        pipeline.categorical_cols = saved_data.get("categorical_cols", [])
        pipeline.feature_columns = saved_data.get("feature_columns", [])
        pipeline.linear_pipeline = saved_data.get("linear")
        pipeline.ensemble_pipeline = saved_data.get("ensemble")
        pipeline.metadata = saved_data.get("metadata", {})
        
        # Check task type to unpack accordingly
        if pipeline.task_type == "classification":
            proba, factors, _ = pipeline.predict_single(inputs, selected_model_name)
            return proba, factors
        else:
            pred_val, factors, _ = pipeline.predict_single(inputs, selected_model_name)
            return pred_val, factors
    except Exception as e:
        return 0.0, [{"name": "error", "label": f"Error loading models: {e}", "contribution": 0.0}]


def render_risk_factors(factors):
    if not factors:
        return ""
    
    pos_factors = sorted([f for f in factors if f['contribution'] > 0.01], key=lambda x: x['contribution'], reverse=True)[:3]
    neg_factors = sorted([f for f in factors if f['contribution'] < -0.01], key=lambda x: x['contribution'])[:3]
    
    pos_html = ""
    for f in pos_factors:
        pos_html += f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding: 10px 14px; background: rgba(239,68,68,0.06); border-radius: 12px; border-left: 3px solid #ef4444;">
            <span style="font-size: 13px; font-weight: 600; color: #fecaca; font-family: 'Outfit', sans-serif;">{f['label']}</span>
            <span style="font-size: 11px; font-family: monospace; color: #f87171; font-weight: 800;">+{f['contribution']:.2f}</span>
        </div>
        """
        
    neg_html = ""
    for f in neg_factors:
        neg_html += f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding: 10px 14px; background: rgba(16,185,129,0.06); border-radius: 12px; border-left: 3px solid #10b981;">
            <span style="font-size: 13px; font-weight: 600; color: #a7f3d0; font-family: 'Outfit', sans-serif;">{f['label']}</span>
            <span style="font-size: 11px; font-family: monospace; color: #34d399; font-weight: 800;">{f['contribution']:.2f}</span>
        </div>
        """
        
    return strip_indent(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
        <div>
            <div style="font-size: 11px; font-weight: 800; text-transform: uppercase; color: #ef4444; letter-spacing: 0.1em; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">🔴 Top Risk Boosters</div>
            {pos_html if pos_html else '<div style="color: #64748b; font-size: 12px; font-style: italic;">No significant boosters found.</div>'}
        </div>
        <div>
            <div style="font-size: 11px; font-weight: 800; text-transform: uppercase; color: #10b981; letter-spacing: 0.1em; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">🟢 Top Risk Reducers</div>
            {neg_html if neg_html else '<div style="color: #64748b; font-size: 12px; font-style: italic;">No significant reducers found.</div>'}
        </div>
    </div>
    """)


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def strip_indent(text):
    import re
    return re.sub(r'^\s+', '', text, flags=re.MULTILINE)


def status_color(status):
    m = {"APPROVED": "#10b981", "CONDITIONALLY APPROVED": "#f59e0b", "REJECTED": "#ef4444"}
    return m.get(status, "#64748b")

def status_icon(status):
    m = {"APPROVED": "✅", "CONDITIONALLY APPROVED": "⚠️", "REJECTED": "❌"}
    return m.get(status, "❔")

def risk_level(s):
    return "Low" if s == 4 else ("Medium" if s >= 2 else "High")

def risk_color(s, max_s=4):
    ratio = s / max(1, max_s)
    return "#10b981" if ratio >= 0.85 else ("#f59e0b" if ratio >= 0.50 else "#ef4444")

def step_icon(step_name):
    icons = {
        "ingestion": "📥", "data_quality": "🔍", "metadata": "📋",
        "preprocessing": "⚙️", "model_training": "🧠", "fairness_check": "⚖️",
        "explainability": "💡", "policy_engine": "🛡️",
    }
    return icons.get(step_name, "▶️")

def step_label(step_name):
    labels = {
        "ingestion": "Data Ingestion", "data_quality": "Quality Checks", "metadata": "Metadata Catalog",
        "preprocessing": "Preprocessing", "model_training": "Model Training", "fairness_check": "Fairness Audit",
        "explainability": "Explainability", "policy_engine": "Policy Gate",
    }
    return labels.get(step_name, step_name.replace("_", " ").title())


def render_metric_card(label, value, subtitle="", glow="#6366f1", icon="📊"):
    return strip_indent(f"""
    <div class="metric-card" style="--glow-color: {glow}; border-left: 3px solid {glow};">
        <div class="metric-card-glow"></div>
        <div class="metric-card-header">{icon}&ensp;{label}</div>
        <div class="metric-card-value">{value}</div>
        <div class="metric-card-subtitle">{subtitle}</div>
    </div>
    """)

def render_circle_gauge(label, val_pct, subtitle="", color="#6366f1", icon="📊"):
    pct = val_pct * 100.0 if val_pct <= 1.0 else val_pct
    pct = min(100.0, max(0.0, pct))
    val_str = f"{val_pct:.1%}" if val_pct <= 1.0 else f"{val_pct:.1f}%"
    return strip_indent(f"""
    <div class="metric-card" style="--glow-color: {color}; border-left: 3px solid {color}; display: flex; align-items: center; justify-content: space-between; gap: 16px;">
        <div class="metric-card-glow"></div>
        <div style="flex: 1;">
            <div class="metric-card-header">{icon}&ensp;{label}</div>
            <div class="metric-card-value">{val_str}</div>
            <div class="metric-card-subtitle">{subtitle}</div>
        </div>
        <div style="position: relative; width: 62px; height: 62px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
            <svg style="transform: rotate(-90deg); width: 62px; height: 62px;">
                <circle cx="31" cy="31" r="26" stroke="rgba(255,255,255,0.03)" stroke-width="4.5" fill="transparent" />
                <circle cx="31" cy="31" r="26" stroke="{color}" stroke-width="4.5" fill="transparent" 
                        stroke-dasharray="163.3" stroke-dashoffset="{163.3 - (163.3 * pct / 100)}" 
                        style="transition: stroke-dashoffset 0.8s ease-in-out; filter: drop-shadow(0 0 3px {color}aa);" />
            </svg>
            <div style="position: absolute; font-size: 10px; font-weight: 850; color: #f8fafc; font-family: 'SF Mono', monospace;">
                {int(pct)}%
            </div>
        </div>
    </div>
    """)

def render_gauge_card(label, val, threshold, icon="👥", color="#10b981"):
    pct = min(100.0, max(0.0, (val / threshold) * 100.0)) if threshold > 0 else 0
    status_label = "ACCEPTABLE" if val <= threshold else "EXCEEDS THRESHOLD"
    return strip_indent(f"""
    <div class="metric-card" style="--glow-color: {color}; border-left: 3px solid {color}; display: flex; align-items: center; justify-content: space-between; gap: 16px;">
        <div class="metric-card-glow"></div>
        <div style="flex: 1;">
            <div class="metric-card-header">{icon}&ensp;{label}</div>
            <div class="metric-card-value">{val:.4f}</div>
            <div style="font-size: 11px; color: #475569; font-weight: 600; margin-top: 4px;">Limit: ≤ {threshold:.2f} · <span style="color: {color}; font-weight: 800;">{status_label}</span></div>
        </div>
        <div style="position: relative; width: 62px; height: 62px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
            <svg style="transform: rotate(-90deg); width: 62px; height: 62px;">
                <circle cx="31" cy="31" r="26" stroke="rgba(255,255,255,0.03)" stroke-width="4.5" fill="transparent" />
                <circle cx="31" cy="31" r="26" stroke="{color}" stroke-width="4.5" fill="transparent" 
                        stroke-dasharray="163.3" stroke-dashoffset="{163.3 - (163.3 * pct / 100)}" 
                        style="transition: stroke-dashoffset 0.8s ease-in-out; filter: drop-shadow(0 0 3px {color}aa);" />
            </svg>
            <div style="position: absolute; font-size: 10px; font-weight: 850; color: #f8fafc; font-family: 'SF Mono', monospace;">
                {int(pct)}%
            </div>
        </div>
    </div>
    """)

def render_badge(text, bg, icon=""):
    return strip_indent(f"""
    <div style="
        display: inline-flex; align-items: center; gap: 8px;
        padding: 10px 24px; border-radius: 999px;
        background: linear-gradient(135deg, {bg}ee, {bg}99);
        color: white; font-weight: 800; font-size: 13px; font-family: \'Outfit\', sans-serif; text-transform: uppercase; letter-spacing: 0.05em;
        box-shadow: 0 4px 24px {bg}55, inset 0 1px 0 rgba(255,255,255,0.1);
        letter-spacing: 0.02em;
        backdrop-filter: blur(8px);
    ">
        {icon} {text}
    </div>
    """)

def render_compliance_item(name, status, description=""):
    is_pass = status == "PASS"
    icon = "✅" if is_pass else "❌"
    bg = "rgba(16,185,129,0.04)" if is_pass else "rgba(239,68,68,0.04)"
    border = "rgba(16,185,129,0.15)" if is_pass else "rgba(239,68,68,0.15)"
    tc = "#34d399" if is_pass else "#f87171"
    pill_bg = "rgba(16,185,129,0.1)" if is_pass else "rgba(239,68,68,0.1)"
    return strip_indent(f"""
    <div class="compliance-item" style="
        display: flex; align-items: center; justify-content: space-between;
        padding: 16px 20px; margin-bottom: 10px;
        background: {bg}; border: 1px solid {border}; border-radius: 18px;
        transition: all 0.25s ease;
    ">
        <div style="display: flex; align-items: center; gap: 14px;">
            <span style="font-size: 18px;">{icon}</span>
            <div>
                <div style="font-family: \'Outfit\', sans-serif; font-size: 15px; font-weight: 800; color: #ffffff; letter-spacing: -0.01em;">{name}</div>
                <div style="font-size: 11.5px; color: #64748b; margin-top: 3px;">{description}</div>
            </div>
        </div>
        <div style="
            font-size: 11px; font-weight: 800; color: {tc};
            padding: 5px 18px; border-radius: 999px; background: {pill_bg};
            letter-spacing: 0.1em;
        ">{status}</div>
    </div>
    """)

def render_signal_banner(text, level="warn"):
    colors = {
        "good": ("#064e3b", "#059669", "#10b981"),
        "warn": ("#78350f", "#d97706", "#f59e0b"),
        "bad":  ("#7f1d1d", "#dc2626", "#ef4444"),
    }
    c1, c2, glow = colors.get(level, colors["warn"])
    return strip_indent(f"""
    <div style="
        padding: 20px 26px; border-radius: 20px;
        background: linear-gradient(135deg, {c1}ee, {c2}cc);
        color: #f8fafc; font-weight: 600; font-size: 14.5px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.08), 0 0 20px {glow}15;
        line-height: 1.6; backdrop-filter: blur(12px);
        border: 1px solid {glow}33;
    ">{text}</div>
    """)

def render_lineage_timeline(entries):
    """Renders a beautiful vertical pipeline timeline."""
    html = '<div style="padding: 12px 0;">'
    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        sname = entry.get("step", "unknown")
        icon = step_icon(sname)
        label = step_label(sname)
        desc = entry.get("description", "")
        ts = entry.get("timestamp", "")
        
        # Determine color of glow based on step state
        glow = "#10b981" if i < len(entries) - 1 else "#6366f1"
        pulse_style = "animation: pulseGlowGreen 2s infinite;" if i < len(entries) - 1 else "animation: pulseGlow 2s infinite;"
        
        html += strip_indent(f"""
        <div style="display: flex; gap: 24px; position: relative; min-height: 80px;">
            <!-- Vertical line & node -->
            <div style="display: flex; flex-direction: column; align-items: center; width: 44px; flex-shrink: 0;">
                <div style="
                    width: 40px; height: 40px; border-radius: 14px;
                    background: linear-gradient(135deg, {glow}15, {glow}35);
                    border: 2px solid {glow}88;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 18px; box-shadow: 0 0 20px {glow}33;
                    flex-shrink: 0; z-index: 1;
                    {pulse_style}
                ">{icon}</div>
                {f'<div style="flex: 1; width: 2px; background: linear-gradient(180deg, {glow}44, rgba(99,102,241,0.05)); margin-top: 6px; margin-bottom: 6px;"></div>' if not is_last else ''}
            </div>
            <!-- Content -->
            <div style="flex: 1; padding-bottom: 20px; padding-top: 2px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                    <span style="font-family: \'Outfit\', sans-serif; font-size: 16px; font-weight: 800; color: #ffffff; letter-spacing: -0.01em;">{label}</span>
                    {f'<span style="font-size: 9px; padding: 2.5px 8px; border-radius: 999px; background: rgba(99,102,241,0.15); color: #a5b4fc; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">LATEST</span>' if is_last else ''}
                </div>
                <div style="font-size: 12.5px; color: #94a3b8; line-height: 1.4; margin-bottom: 4px;">{desc}</div>
                <div style="font-size: 11px; color: #64748b; font-family: \'SF Mono\', \'Fira Code\', monospace; font-weight: 600;">{ts}</div>
            </div>
        </div>
        """)
    html += '</div>'
    return html

def render_html_table(df):
    """Renders a beautiful custom glassmorphic HTML table instead of standard streamlit dataframe."""
    html = """
    <div style="overflow-x: auto; border-radius: 18px; border: 1px solid rgba(255,255,255,0.06); background: rgba(10,15,30,0.3); backdrop-filter: blur(12px); box-shadow: 0 10px 30px rgba(0,0,0,0.25);">
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-family: 'Inter', sans-serif; font-size: 13px; color: #cbd5e1;">
            <thead>
                <tr style="background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.08);">
    """
    html += f'<th style="font-family: \'Outfit\', sans-serif; padding: 16px 20px; font-weight: 800; color: #cbd5e1; text-transform: uppercase; font-size: 11px; letter-spacing: 0.08em;">{df.index.name or "Rank"}</th>'
    for col in df.columns:
        html += f'<th style="font-family: \'Outfit\', sans-serif; padding: 16px 20px; font-weight: 800; color: #cbd5e1; text-transform: uppercase; font-size: 11px; letter-spacing: 0.08em;">{col}</th>'
    html += "</tr></thead><tbody>"
    for idx, row in df.iterrows():
        html += '<tr class="table-row" style="border-bottom: 1px solid rgba(255,255,255,0.04); transition: all 0.2s;">'
        html += f'<td style="padding: 16px 20px; font-weight: 700; color: #f1f5f9;">{idx}</td>'
        for val in row:
            val_str = f"{val:.5f}" if isinstance(val, float) else str(val)
            html += f'<td style="padding: 16px 20px; color: #cbd5e1; font-weight: 500;">{val_str}</td>'
        html += "</tr>"
    html += "</tbody></table></div>"
    return html

def render_key_value_list(data):
    """Renders a beautiful glassmorphic metadata summary list, handling nested lists/dicts gracefully."""
    html = '<div style="display: flex; flex-direction: column; gap: 14px; padding: 6px 0;">'
    for k, v in data.items():
        label = k.replace("_", " ").title()
        
        if isinstance(v, list):
            tag_html = '<div style="display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; max-width: 75%;">'
            for item in v:
                tag_html += f'<span style="background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.15); padding: 3px 8px; border-radius: 6px; color: #c7d2fe; font-size: 11px; font-weight: 600; font-family: \'SF Mono\', monospace; white-space: nowrap;">{item}</span>'
            tag_html += '</div>'
            value_block = tag_html
        elif isinstance(v, dict):
            tag_html = '<div style="display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; max-width: 75%;">'
            for dict_k, dict_v in v.items():
                tag_html += f'<span style="background: rgba(139,92,246,0.06); border: 1px solid rgba(139,92,246,0.15); padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; white-space: nowrap;"><strong style="color: #c084fc; font-family: \'Outfit\', sans-serif;">{dict_k}</strong>: <span style="color: #94a3b8; font-family: \'SF Mono\', monospace;">{dict_v}</span></span>'
            tag_html += '</div>'
            value_block = tag_html
        else:
            value_block = f'<span style="font-size: 13px; color: #f1f5f9; font-weight: 700; font-family: \'SF Mono\', \'Fira Code\', monospace;">{v}</span>'
            
        html += strip_indent(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.03);">
            <span style="font-family: \'Outfit\', sans-serif; font-size: 13px; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; min-width: 20%;">{label}</span>
            {value_block}
        </div>
        """)
    html += '</div>'
    return html

def render_progress_bar(pct, color="#10b981"):
    """Renders a custom neon progress bar."""
    return strip_indent(f"""
    <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.04); border-radius: 99px; overflow: hidden; margin-top: 14px; border: 1px solid rgba(255,255,255,0.02);">
        <div style="width: {pct}%; height: 100%; background: linear-gradient(90deg, {color}aa, {color}); border-radius: 99px; box-shadow: 0 0 12px {color}aa;"></div>
    </div>
    """)


# ── Dark-themed chart functions ──

def dark_bar_chart(labels, values, color="#818cf8", title="", horizontal=False, value_fmt=".3f"):
    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.6) if horizontal else 4.8))
    fig.patch.set_facecolor("#0c1222")
    ax.set_facecolor("#0c1222")
    positions = np.arange(len(labels))

    if horizontal:
        bars = ax.barh(positions, values, color=color, height=0.55, edgecolor="none", alpha=0.92)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=11, color="#cbd5e1", fontweight=600)
        ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.025, bar.get_y() + bar.get_height() / 2,
                    f"{val:{value_fmt}}", va="center", ha="left", fontsize=10, color="#94a3b8", fontweight=700)
    else:
        bars = ax.bar(positions, values, color=color, width=0.5, edgecolor="none", alpha=0.92)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=11, color="#cbd5e1", fontweight=600)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.025,
                    f"{val:{value_fmt}}", ha="center", va="bottom", fontsize=10, color="#94a3b8", fontweight=700)

    if title:
        ax.set_title(title, fontsize=14, color="#e2e8f0", fontweight=800, pad=18, loc="left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1e293b")
    ax.spines["bottom"].set_color("#1e293b")
    ax.tick_params(colors="#475569", labelsize=10)
    ax.grid(axis="x" if horizontal else "y", color="#1e293b", linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    return fig


def dark_grouped_bar_chart(labels, g1_vals, g2_vals, g1_name, g2_name,
                           c1="#818cf8", c2="#475569", title=""):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0c1222")
    ax.set_facecolor("#0c1222")
    x = np.arange(len(labels))
    w = 0.30
    bars1 = ax.bar(x - w / 2, g1_vals, w, label=g1_name, color=c1, edgecolor="none", alpha=0.92)
    bars2 = ax.bar(x + w / 2, g2_vals, w, label=g2_name, color=c2, edgecolor="none", alpha=0.92)
    for bar, val in zip(bars1, g1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#a5b4fc", fontweight=700)
    for bar, val in zip(bars2, g2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#94a3b8", fontweight=700)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, color="#cbd5e1", fontweight=600)
    if title:
        ax.set_title(title, fontsize=14, color="#e2e8f0", fontweight=800, pad=18, loc="left")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1e293b")
    ax.spines["bottom"].set_color("#1e293b")
    ax.tick_params(colors="#475569", labelsize=10)
    ax.grid(axis="y", color="#1e293b", linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    return fig


def dark_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor("#0c1222")
    ax.set_facecolor("#0c1222")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, color="#e2e8f0", fontweight=800, pad=14)
    ax.set_xlabel("Predicted", fontsize=11, color="#94a3b8", fontweight=600)
    ax.set_ylabel("Actual", fontsize=11, color="#94a3b8", fontweight=600)
    ax.tick_params(colors="#94a3b8", labelsize=10)
    for t in ax.texts:
        t.set_color("#f1f5f9")
        t.set_fontweight(800)
        t.set_fontsize(18)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS — GOD-LEVEL AESTHETICS
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800;900&display=swap');

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    h1, h2, h3, .outfit-font {
        font-family: 'Outfit', sans-serif !important;
    }

    /* ── Hide default streamlit elements ── */
    header { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }

    /* ── Animated Background - God Level ── */
    .stApp {
        background: #030509; /* Ultra deep dark base */
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.08), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.08), transparent 25%),
            radial-gradient(circle at 50% 80%, rgba(16, 185, 129, 0.05), transparent 20%),
            linear-gradient(180deg, #030509 0%, #060a12 50%, #030509 100%);
        background-attachment: fixed;
        animation: bgShift 20s ease-in-out infinite alternate;
    }

    @keyframes bgShift {
        0% { background-position: 0% 0%; }
        100% { background-position: 100% 100%; }
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
    }

    /* ── Sidebar — Ultra Dark Glass ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(3,5,9,0.95) 0%, rgba(6,10,18,0.98) 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.03) !important;
        box-shadow: 10px 0 50px rgba(0,0,0,0.8) !important;
        backdrop-filter: blur(20px) !important;
    }

    section[data-testid="stSidebar"] * { color: #94a3b8 !important; }

    /* ── Sidebar Radio Navigation ── */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        background: rgba(10, 15, 30, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 8px !important;
        gap: 6px !important;
        margin-top: 10px !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02) !important;
    }

    /* Hide Streamlit default radio check circles */
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin: 0 !important;
        width: 100% !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
    }

    /* Hover state for inactive options */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.03) !important;
        border-color: rgba(255, 255, 255, 0.02) !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover p {
        color: #f1f5f9 !important;
    }

    /* Active selected state (matching the purple/indigo theme) */
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: rgba(99, 102, 241, 0.12) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.12) !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #a5b4fc !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(165, 180, 252, 0.15) !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
        color: #94a3b8 !important;
        letter-spacing: 0.02em !important;
        margin: 0 !important;
    }

    /* Hide standard streamlit metric styling */
    div[data-testid="stMetric"] { display: none !important; }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(145deg, rgba(10,14,24,0.7), rgba(6,9,16,0.9));
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 24px;
        padding: 24px 28px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.02);
        backdrop-filter: blur(24px);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .metric-card:hover {
        transform: translateY(-6px) scale(1.02);
        border-color: var(--glow-color) !important;
        box-shadow: 
            0 20px 50px rgba(0, 0, 0, 0.6),
            0 0 30px var(--glow-color)33,
            inset 0 1px 0 rgba(255,255,255,0.06);
    }

    .metric-card-glow {
        position: absolute;
        top: -40px;
        right: -40px;
        width: 140px;
        height: 140px;
        background: radial-gradient(circle, var(--glow-color)15, transparent 65%);
        border-radius: 50%;
        pointer-events: none;
        transition: all 0.4s ease;
    }

    .metric-card:hover .metric-card-glow {
        background: radial-gradient(circle, var(--glow-color)25, transparent 70%);
        transform: scale(1.2);
    }

    .metric-card-header {
        font-family: 'Outfit', sans-serif;
        font-size: 12px;
        color: #94a3b8;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .metric-card-value {
        font-family: 'Outfit', sans-serif;
        font-size: 36px;
        font-weight: 900;
        color: #ffffff;
        line-height: 1;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }

    .metric-card-subtitle {
        font-size: 12px;
        color: #64748b;
        font-weight: 500;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        border-radius: 32px;
        padding: 48px 56px;
        background: linear-gradient(135deg, rgba(10,15,30,0.8) 0%, rgba(4,6,12,0.95) 100%) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        box-shadow: 
            0 30px 100px rgba(0,0,0,0.5), 
            inset 0 1px 0 rgba(255,255,255,0.04),
            inset 0 0 40px rgba(99,102,241,0.05) !important;
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }

    .hero-banner::before {
        content: ''; position: absolute; top: -50%; right: -15%;
        width: 600px; height: 600px;
        background: radial-gradient(circle, rgba(99,102,241,0.1), transparent 60%);
        border-radius: 50%; filter: blur(40px); animation: float 10s ease-in-out infinite alternate;
    }

    .hero-banner::after {
        content: ''; position: absolute; bottom: -30%; left: -10%;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(139,92,246,0.08), transparent 60%);
        border-radius: 50%; filter: blur(40px); animation: float 8s ease-in-out infinite alternate-reverse;
    }

    @keyframes float {
        0% { transform: translateY(0) scale(1); }
        100% { transform: translateY(-20px) scale(1.05); }
    }

    .hero-banner h1 {
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        font-size: 3.2rem;
        line-height: 1.1;
        margin-bottom: 16px;
        font-weight: 900;
        letter-spacing: -0.04em;
        position: relative;
        background: linear-gradient(135deg, #ffffff 20%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 20px rgba(99,102,241,0.2);
    }

    .hero-banner p {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 28px;
        max-width: 720px;
        line-height: 1.7;
        position: relative;
        font-weight: 400;
    }

    /* ── General Glass Cards ── */
    .glass-card {
        background: linear-gradient(180deg, rgba(10,14,24,0.7), rgba(6,9,16,0.9));
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 24px;
        padding: 32px 36px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.02);
        margin-bottom: 24px;
        backdrop-filter: blur(24px);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(255,255,255,0.08);
        box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .glass-card .card-title {
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }

    .glass-card .card-subtitle {
        color: #64748b;
        margin-bottom: 24px;
        font-size: 0.9rem;
        font-weight: 400;
    }

    /* ── Buttons & Downloads ── */
    .stButton > button, .stDownloadButton > button {
        border-radius: 16px !important;
        padding: 0.8rem 1.4rem !important;
        background: rgba(255,255,255,0.03) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        background: rgba(99,102,241,0.15) !important;
        border-color: rgba(99,102,241,0.4) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 24px rgba(99,102,241,0.2), inset 0 1px 0 rgba(255,255,255,0.1) !important;
        transform: translateY(-2px) !important;
    }

    /* ── Section Dividers ── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 36px 0;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #030509; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #334155; }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulseGlowGreen {
        0%, 100% { box-shadow: 0 0 10px rgba(16,185,129,0.4), inset 0 0 8px rgba(16,185,129,0.2); }
        50% { box-shadow: 0 0 24px rgba(16,185,129,0.8), inset 0 0 12px rgba(16,185,129,0.4); }
    }
    @keyframes pulseGlowAmber {
        0%, 100% { box-shadow: 0 0 10px rgba(245,158,11,0.4), inset 0 0 8px rgba(245,158,11,0.2); }
        50% { box-shadow: 0 0 24px rgba(245,158,11,0.8), inset 0 0 12px rgba(245,158,11,0.4); }
    }
    @keyframes pulseGlowRed {
        0%, 100% { box-shadow: 0 0 10px rgba(239,68,68,0.4), inset 0 0 8px rgba(239,68,68,0.2); }
        50% { box-shadow: 0 0 24px rgba(239,68,68,0.8), inset 0 0 12px rgba(239,68,68,0.4); }
    }

    .pulse-green { animation: pulseGlowGreen 2.5s infinite; }
    .pulse-amber { animation: pulseGlowAmber 2.5s infinite; }
    .pulse-red { animation: pulseGlowRed 2.5s infinite; }

    .animate-in { animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; }

    /* Tables */
    .table-row:hover { background: rgba(255,255,255,0.03) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 8px;">
        <div style="
            width: 44px; height: 44px;
            background: linear-gradient(135deg, #1e1b4b, #312e81);
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 4px 20px rgba(99,102,241,0.35), inset 0 1px 0 rgba(255,255,255,0.15);
            border: 1px solid rgba(139,92,246,0.3);
        ">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 5V21M12 21C7.58 21 4 17.42 4 13M12 21C16.42 21 20 17.42 20 13M8 9H16" stroke="url(#anchorGrad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="12" cy="4" r="2" stroke="#a5b4fc" stroke-width="2"/>
                <path d="M4 13L2 12.5M20 13L22 12.5" stroke="#8b5cf6" stroke-width="2" stroke-linecap="round"/>
                <defs>
                    <linearGradient id="anchorGrad" x1="4" y1="5" x2="20" y2="21" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#818cf8"/>
                        <stop offset="1" stop-color="#4f46e5"/>
                    </linearGradient>
                </defs>
            </svg>
        </div>
        <div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 20px; font-weight: 900; color: #ffffff !important; letter-spacing: -0.01em;">Anchor AI</div>
            <div style="font-size: 10px; color: #a5b4fc !important; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em;">Trust & Stability</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.12), transparent); margin: 18px 0;"></div>', unsafe_allow_html=True)

    if is_custom:
        pipeline = st.session_state.get("automl_pipeline")
        is_data_only = pipeline.metadata.get("data_only", False) if pipeline else False
        default_idx = 7 if is_data_only else 0
        page = st.radio(
            "Navigation",
            ["Overview", "Model Card", "Data Card", "Risk Sandbox", "Risk & Fairness", "Model Performance",
             "Explainability", "Data Health", "Lineage", "Artifacts"],
            index=default_idx,
            label_visibility="collapsed",
        )
        st.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.12), transparent); margin: 18px 0;"></div>', unsafe_allow_html=True)

        # ── Custom Dataset Uploader in Sidebar (Only when custom dataset is active) ──
        st.markdown('<div style="font-size: 11px; color: #818cf8; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 10px;">Dataset Control</div>', unsafe_allow_html=True)
        uploaded_file_sidebar = st.file_uploader("Upload CSV Dataset", type=["csv"], key="sidebar_uploader", help="Upload any tabular CSV file to dynamically train models, audit fairness, and view explanations.")
        
        if uploaded_file_sidebar is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file_sidebar)
                st.session_state["custom_filename"] = uploaded_file_sidebar.name
                cols = list(df_uploaded.columns)
                
                # Select target & demographic columns
                target_col = st.selectbox(
                    "Prediction Target", 
                    options=cols, 
                    index=len(cols)-1, 
                    key="sidebar_target",
                    help="Select the column/variable you want the AI models to predict."
                )
                
                # Auto detect demographic column
                temp_pipeline = AutoMLPipeline(target_col=target_col)
                detected_demo = temp_pipeline.auto_detect_demographic(df_uploaded)
                default_demo_idx = cols.index(detected_demo) if detected_demo in cols else 0
                demographic_col = st.selectbox(
                    "Demographic Column", 
                    options=cols, 
                    index=default_demo_idx, 
                    key="sidebar_demo",
                    help="Select a column representing sensitive groups (e.g. Gender, Age, Race, Region) to audit the model for algorithmic bias and group fairness."
                )
                
                sidebar_run_mode = st.radio(
                    "Execution Mode",
                    ["Full AI Pipeline & Audits", "Data Profiling & Health Only"],
                    index=0,
                    key="sidebar_run_mode",
                    help="Choose whether to train machine learning models and audit compliance gates, or only generate data health metrics."
                )
                
                # Exclude ID Columns
                cols_excluding_target_demo = [c for c in cols if c != target_col and c != demographic_col]
                id_cols = st.multiselect(
                    "Exclude ID Columns",
                    options=cols_excluding_target_demo,
                    default=[],
                    key="sidebar_id_cols"
                )
                
                # Task Override
                task_override = st.selectbox(
                    "Task Type Override",
                    options=["Auto-Detect", "Classification", "Regression"],
                    index=0,
                    key="sidebar_task_override"
                )
                
                # Positive Class
                is_classif = False
                if task_override == "Classification":
                    is_classif = True
                elif task_override == "Auto-Detect":
                    if df_uploaded[target_col].nunique() == 2 or not pd.api.types.is_numeric_dtype(df_uploaded[target_col].dtype):
                        is_classif = True
                positive_class = None
                if is_classif:
                    unique_targets = sorted(list(df_uploaded[target_col].dropna().astype(str).unique()))
                    if len(unique_targets) > 0:
                        positive_class = st.selectbox(
                            "Positive Class",
                            options=unique_targets,
                            index=len(unique_targets)-1,
                            key="sidebar_positive_class"
                        )

                # Schema Gate Table
                sidebar_schema_data = []
                for col in df_uploaded.columns:
                    missing_pct = df_uploaded[col].isnull().mean() * 100
                    if col == target_col:
                        col_role = "🎯 Target"
                    elif col == demographic_col:
                        col_role = "⚖️ Demographic"
                    elif col in id_cols:
                        col_role = "🚫 Excluded"
                    elif pd.api.types.is_numeric_dtype(df_uploaded[col]):
                        col_role = "🔢 Numeric"
                    else:
                        col_role = "🔤 Categorical"
                    sidebar_schema_data.append({
                        "Column": col,
                        "Role": col_role,
                        "Missing": f"{missing_pct:.1f}%"
                    })
                with st.expander("📋 Verify Inferred Schema Gate", expanded=False):
                    st.dataframe(pd.DataFrame(sidebar_schema_data), use_container_width=True)

                # EU AI Act Compliance Wizard
                st.markdown('<div style="font-size: 11px; color: #818cf8; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; margin-top: 10px;">EU AI Act Risk Wizard</div>', unsafe_allow_html=True)
                sidebar_eu_sector = st.selectbox(
                    "Deployment Sector",
                    options=[
                        "General Low-Risk Commercial Use",
                        "Employment & Worker Management",
                        "Education & Vocational Training",
                        "Essential Private & Public Services",
                        "Law Enforcement & Justice System",
                        "Critical Infrastructure"
                    ],
                    index=0,
                    key="sidebar_eu_sector"
                )
                
                if sidebar_eu_sector != "General Low-Risk Commercial Use":
                    computed_risk_sidebar = "High"
                    st.warning("⚠️ High-Risk. Strict limits apply.")
                else:
                    computed_risk_sidebar = "Low"
                    st.info("🟢 Low-Risk.")

                # Confirm checkbox
                sidebar_schema_confirmed = st.checkbox("Verify Schema & Risk Class", key="sidebar_schema_confirmed")

                # Run button
                if st.button("⚡ Run Pipeline", use_container_width=True, key="sidebar_run", disabled=not sidebar_schema_confirmed):
                    is_data_only = (sidebar_run_mode == "Data Profiling & Health Only")
                    with st.spinner("Processing dataset..."):
                        pipeline = AutoMLPipeline(
                            target_col=target_col,
                            demographic_col=demographic_col,
                            task_type_override=task_override if task_override != "Auto-Detect" else None,
                            positive_class=positive_class,
                            id_columns_override=id_cols
                        )
                        pipeline.run_pipeline(df_uploaded, custom_limits=st.session_state.get("custom_limits"), data_only=is_data_only, eu_risk_tier=computed_risk_sidebar, filename=uploaded_file_sidebar.name)
                        st.session_state["automl_pipeline"] = pipeline
                        st.session_state["is_custom_run"] = True
                        st.success("Pipeline Run Success!")
                        st.rerun()
            except Exception as e:
                st.error(f"Error parsing file: {e}")
                
        if st.button("Upload New Dataset", use_container_width=True, key="sidebar_reset"):
            if "automl_pipeline" in st.session_state:
                del st.session_state["automl_pipeline"]
            st.session_state["is_custom_run"] = False
            st.rerun()

        # ---------- Chatbot Auditor ----------
        st.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.12), transparent); margin: 18px 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 11px; color: #818cf8; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 10px;">💬 Governance Chatbot Auditor</div>', unsafe_allow_html=True)
        
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [
                {"role": "assistant", "content": "Hello! I am your AI compliance auditor. How can I help you analyze this model's policy gates, data drift, or bias reports?"}
            ]
            
        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if prompt := st.chat_input("Ask auditor...", key="chatbot_input"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            
            from src.chatbot_engine import generate_governance_explanation
            response = generate_governance_explanation(prompt, compliance_report)
            
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state["chat_messages"].append({"role": "assistant", "content": response})


# ══════════════════════════════════════════════════════════════
#  LANDING PAGE (IF NO CUSTOM DATASET YET)
# ══════════════════════════════════════════════════════════════

if not is_custom:
    st.markdown("""
<div style="text-align: center; font-family: 'Outfit', sans-serif; margin-top: 40px; margin-bottom: 30px;">
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <div style="
            width: 64px; height: 64px;
            background: linear-gradient(135deg, #1e1b4b, #312e81);
            border-radius: 18px;
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 8px 32px rgba(99,102,241,0.35), inset 0 1px 0 rgba(255,255,255,0.15);
            border: 1px solid rgba(139,92,246,0.3);
        ">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 5V21M12 21C7.58 21 4 17.42 4 13M12 21C16.42 21 20 17.42 20 13M8 9H16" stroke="url(#landingAnchorGrad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="12" cy="4" r="2" stroke="#a5b4fc" stroke-width="2"/>
                <path d="M4 13L2 12.5M20 13L22 12.5" stroke="#8b5cf6" stroke-width="2" stroke-linecap="round"/>
                <defs>
                    <linearGradient id="landingAnchorGrad" x1="4" y1="5" x2="20" y2="21" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#818cf8"/>
                        <stop offset="1" stop-color="#4f46e5"/>
                    </linearGradient>
                </defs>
            </svg>
        </div>
    </div>
    <h1 style="font-size: 44px; font-weight: 900; background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 12px; letter-spacing: -0.02em;">Anchor AI</h1>
    <p style="font-size: 18px; color: #94a3b8; font-weight: 400; max-width: 600px; margin: 0 auto; line-height: 1.6;">
        Automated Model Validation & Governance Engine. Upload a CSV dataset to train models, evaluate biases, and inspect compliance policies in real-time.
    </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; max-width: 900px; margin: 0 auto 40px auto; font-family: 'Outfit', sans-serif;">
    <div style="background: rgba(30, 41, 59, 0.25); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 24px; box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);">
        <div style="font-size: 11px; color: #818cf8; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Step 1</div>
        <h3 style="font-size: 16px; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">Upload Dataset</h3>
        <p style="font-size: 13px; color: #64748b; line-height: 1.5; margin: 0;">Drag and drop any tabular CSV file containing your features and target variables.</p>
    </div>
    <div style="background: rgba(30, 41, 59, 0.25); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 24px; box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);">
        <div style="font-size: 11px; color: #818cf8; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Step 2</div>
        <h3 style="font-size: 16px; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">Configure Target & Bias</h3>
        <p style="font-size: 13px; color: #64748b; line-height: 1.5; margin: 0;">Select your prediction target and the demographic column for compliance audits.</p>
    </div>
    <div style="background: rgba(30, 41, 59, 0.25); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 24px; box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);">
        <div style="font-size: 11px; color: #818cf8; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Step 3</div>
        <h3 style="font-size: 16px; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">Inspect Governance</h3>
        <p style="font-size: 13px; color: #64748b; line-height: 1.5; margin: 0;">Unlock performance metrics, SHAP explainers, fairness audits, and the interactive sandbox.</p>
    </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
<div style="text-align: center; margin-bottom: 20px; font-family: 'Outfit', sans-serif;">
    <h3 style="font-size: 20px; font-weight: 800; color: #ffffff;">Let's get started</h3>
    <p style="font-size: 14px; color: #64748b; margin: 4px 0 0 0;">Upload your CSV file below to initialize the automated pipeline</p>
</div>
""", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], key="landing_uploader", label_visibility="collapsed")
        
        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                st.session_state["custom_filename"] = uploaded_file.name
                cols = list(df_uploaded.columns)
                
                sel1, sel2 = st.columns(2)
                with sel1:
                    target_col = st.selectbox(
                        "Prediction Target", 
                        options=cols, 
                        index=len(cols)-1, 
                        key="landing_target",
                        help="Select the column/variable you want the AI models to predict."
                    )
                with sel2:
                    temp_pipeline = AutoMLPipeline(target_col=target_col)
                    detected_demo = temp_pipeline.auto_detect_demographic(df_uploaded)
                    default_demo_idx = cols.index(detected_demo) if detected_demo in cols else 0
                    demographic_col = st.selectbox(
                        "Demographic Column", 
                        options=cols, 
                        index=default_demo_idx, 
                        key="landing_demo",
                        help="Select a column representing sensitive groups (e.g. Gender, Age, Race, Region) to audit the model for algorithmic bias and group fairness."
                    )
                
                st.markdown("""
<div style="font-size: 12px; color: #94a3b8; line-height: 1.5; margin-top: 10px; font-family: 'Outfit', sans-serif; text-align: left;">
    <strong>💡 Demographic Column:</strong> Used to audit algorithmic fairness. If your dataset does not contain human traits, select any categorical column to evaluate group-level prediction gaps.
</div>
""", unsafe_allow_html=True)
                
                # Advanced configuration expander
                with st.expander("⚙️ Advanced Pipeline Configuration", expanded=False):
                    cols_excluding_target_demo = [c for c in cols if c != target_col and c != demographic_col]
                    id_cols = st.multiselect(
                        "Exclude ID/Key Columns from Training",
                        options=cols_excluding_target_demo,
                        default=[],
                        key="landing_id_cols",
                        help="Select columns to ignore during training (e.g. ID, names, primary keys, timestamps)."
                    )
                    
                    task_override = st.selectbox(
                        "Task Type Override",
                        options=["Auto-Detect", "Classification", "Regression"],
                        index=0,
                        key="landing_task_override",
                        help="Override task auto-detection if needed."
                    )
                    
                    # Positive Class (only if classification is likely or selected)
                    is_classif = False
                    if task_override == "Classification":
                        is_classif = True
                    elif task_override == "Auto-Detect":
                        if df_uploaded[target_col].nunique() == 2 or not pd.api.types.is_numeric_dtype(df_uploaded[target_col].dtype):
                            is_classif = True
                            
                    positive_class = None
                    if is_classif:
                        unique_targets = sorted(list(df_uploaded[target_col].dropna().astype(str).unique()))
                        if len(unique_targets) > 0:
                            positive_class = st.selectbox(
                                "Positive Class Label",
                                options=unique_targets,
                                index=len(unique_targets)-1,
                                key="landing_positive_class",
                                help="Select the class value that represents the event of interest (e.g. 'Yes', 1, 'Survived')."
                            )
                
                # HIML Schema Gate Table
                schema_data = []
                for col in df_uploaded.columns:
                    missing_pct = df_uploaded[col].isnull().mean() * 100
                    unique_vals = df_uploaded[col].nunique()
                    
                    if col == target_col:
                        col_type = "🎯 Prediction Target"
                    elif col == demographic_col:
                        col_type = "⚖️ Protected Demographic"
                    elif col in id_cols:
                        col_type = "🚫 Ignored (Excluded)"
                    elif pd.api.types.is_numeric_dtype(df_uploaded[col]):
                        col_type = "🔢 Numeric Feature"
                    else:
                        col_type = "🔤 Categorical Feature"
                        
                    if pd.api.types.is_numeric_dtype(df_uploaded[col]):
                        range_details = f"Min: {df_uploaded[col].min()}, Max: {df_uploaded[col].max()}"
                    else:
                        range_details = f"Unique values: {unique_vals}"
                        
                    schema_data.append({
                        "Column Name": col,
                        "Data Type Status": col_type,
                        "Missing %": f"{missing_pct:.1f}%",
                        "Details / Range": range_details
                    })
                
                st.markdown('<div style="font-size: 14px; font-weight: 700; color: #818cf8; margin-top: 15px; margin-bottom: 8px; font-family: \'Outfit\', sans-serif;">📋 Inferred Column Schemas & Role Mapping</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(schema_data), use_container_width=True)

                # EU AI Act Compliance Risk Wizard
                st.markdown('<div style="font-size: 14px; font-weight: 700; color: #818cf8; margin-top: 15px; margin-bottom: 8px; font-family: \'Outfit\', sans-serif;">🇪🇺 EU AI Act Compliance Risk Wizard</div>', unsafe_allow_html=True)
                eu_sector = st.selectbox(
                    "Select System Deployment Sector/Domain",
                    options=[
                        "General Low-Risk Commercial Use (e.g., e-commerce recommendation, movie recommendations)",
                        "Employment, Worker Management (e.g., CV screening, hiring, promotion scoring)",
                        "Education & Vocational Training (e.g., student admissions, grading)",
                        "Essential Private & Public Services (e.g., credit risk assessment, loan eligibility, utility access)",
                        "Law Enforcement & Justice System (e.g., criminal risk profiling, judge decision aids)",
                        "Critical Infrastructure Operations (e.g., road traffic systems, water/power supply networks)",
                        "Other Custom High-Risk Category"
                    ],
                    index=0,
                    key="landing_eu_sector",
                    help="Under the EU AI Act, systems deployed in critical areas are designated as High-Risk, requiring stricter governance thresholds."
                )

                if eu_sector != "General Low-Risk Commercial Use (e.g., e-commerce recommendation, movie recommendations)":
                    computed_risk = "High"
                    info_msg = "⚠️ **High-Risk Application Detected**: Under the EU AI Act, this sector is classified as High-Risk. Strict threshold gates will be automatically enforced (e.g. Min Accuracy ≥ 0.85, Max Demographic Bias Gap ≤ 0.08, Max PSI Drift ≤ 0.10)."
                else:
                    computed_risk = "Low"
                    info_msg = "🟢 **Low-Risk Application**: Standard commercial use. Normal governance thresholds will apply (e.g. Min Accuracy ≥ 0.75, Max Demographic Bias Gap ≤ 0.15, Max PSI Drift ≤ 0.20)."

                st.markdown(f'<div style="padding: 12px; background-color: rgba(99, 102, 241, 0.08); border-left: 4px solid {"#ef4444" if computed_risk == "High" else "#10b981"}; border-radius: 4px; font-size: 13px; color: #e2e8f0; margin-bottom: 15px;">{info_msg}</div>', unsafe_allow_html=True)

                landing_run_mode = st.radio(
                    "Execution Mode",
                    ["Full AI Pipeline & Audits", "Data Profiling & Health Only"],
                    index=0,
                    key="landing_run_mode",
                    horizontal=True,
                    help="Choose whether to train ML models and run audits, or only generate data quality stats."
                )
                
                # Confirm checkbox
                schema_confirmed = st.checkbox("I verify the dataset schema, column roles, and EU AI Act classification.", key="landing_schema_confirmed")

                st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
                if st.button("⚡ Run Pipeline", use_container_width=True, key="landing_run", disabled=not schema_confirmed):
                    is_data_only = (landing_run_mode == "Data Profiling & Health Only")
                    with st.spinner("Processing dataset..."):
                        pipeline = AutoMLPipeline(
                            target_col=target_col,
                            demographic_col=demographic_col,
                            task_type_override=task_override if task_override != "Auto-Detect" else None,
                            positive_class=positive_class,
                            id_columns_override=id_cols
                        )
                        pipeline.run_pipeline(df_uploaded, custom_limits=st.session_state.get("custom_limits"), data_only=is_data_only, eu_risk_tier=computed_risk, filename=uploaded_file.name)
                        st.session_state["automl_pipeline"] = pipeline
                        st.session_state["is_custom_run"] = True
                        st.success("Pipeline Run Success!")
                        st.rerun()
            except Exception as e:
                st.error(f"Error parsing file: {e}")
        else:
            st.markdown('<div style="text-align: center; margin: 15px 0; color: #64748b;">— OR —</div>', unsafe_allow_html=True)
            if st.button("📊 Load Default Churn Demo Dataset", use_container_width=True, key="landing_demo_btn"):
                try:
                    df_uploaded = pd.read_csv(BASE_DIR / "data" / "raw" / "Customer-Churn.csv")
                    st.session_state["custom_filename"] = "Customer-Churn.csv"
                    with st.spinner("Loading demo dataset..."):
                        pipeline = AutoMLPipeline(target_col="Churn", demographic_col="gender")
                        pipeline.run_pipeline(df_uploaded, custom_limits=st.session_state.get("custom_limits"), data_only=False)
                        st.session_state["automl_pipeline"] = pipeline
                        st.session_state["is_custom_run"] = True
                        st.success("Demo loaded successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading demo: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="hero-banner animate-in">
    <h1>Anchor AI</h1>
    <p>
        Production-grade decision support for churn prediction —
        performance, fairness, explainability &amp; policy approval in one view.
    </p>
    {render_badge(f"{status_icon(final_status)}  {final_status}", status_color(final_status))}
</div>
""", unsafe_allow_html=True)

# ── KPI strip ──
if is_custom and pipeline.metadata.get("data_only", False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_circle_gauge("Data Quality Score", quality_report.get("quality_score", 0), "Completeness & duplicate check", "#10b981" if quality_report.get("quality_score", 0) >= 80 else "#ef4444", "🎯"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("Total Rows", f"{quality_report.get('num_rows', 0):,}", "Dataset row count", "#818cf8", "📋"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card("Total Columns", f"{quality_report.get('num_columns', 0):,}", "Dataset column count", "#38bdf8", "📊"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card("Missing Values", f"{quality_report.get('total_missing_values', 0):,}", "Empty data cells", "#f43f5e" if quality_report.get('total_missing_values', 0) > 0 else "#10b981", "🔍"), unsafe_allow_html=True)
elif compliance_report and model_metrics and fairness_report:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_circle_gauge("Governance Score", score / max(1, max_score), "Policy gates passed", risk_color(score, max_score), "🎯"), unsafe_allow_html=True)
    
    if is_custom:
        pipeline = st.session_state["automl_pipeline"]
        if pipeline.task_type == "classification":
            with c2:
                st.markdown(render_circle_gauge("Model Accuracy", log_model.get('accuracy', 0), "Logistic Regression", "#818cf8", "📈"), unsafe_allow_html=True)
            with c3:
                dg = fairness_report.get("demographic_gap", 0)
                st.markdown(render_gauge_card(f"{pipeline.demographic_col.title()} Gap", dg, 0.10, "👥", "#10b981" if dg <= 0.10 else "#ef4444"), unsafe_allow_html=True)
            with c4:
                st.markdown(render_circle_gauge("F1 Score", log_model.get('f1_score', 0), "Logistic Regression", "#38bdf8", "⚡"), unsafe_allow_html=True)
        else:
            with c2:
                st.markdown(render_circle_gauge("Model R² Score", log_model.get('r2_score', 0), "Ridge Regression", "#818cf8", "📈"), unsafe_allow_html=True)
            with c3:
                dg = fairness_report.get("demographic_gap", 0)
                st.markdown(render_gauge_card(f"{pipeline.demographic_col.title()} Gap", dg, 0.10, "👥", "#10b981" if dg <= 0.10 else "#ef4444"), unsafe_allow_html=True)
            with c4:
                st.markdown(render_circle_gauge("Alternative R²", rf_model.get('r2_score', 0), "Random Forest Regressor", "#475569", "🌲"), unsafe_allow_html=True)
    else:
        with c2:
            st.markdown(render_circle_gauge("Model Accuracy", log_model.get('accuracy', 0), "Logistic Regression", "#818cf8", "📈"), unsafe_allow_html=True)
        with c3:
            gg = fairness_report.get("gender_gap", 1)
            st.markdown(render_gauge_card("Gender Gap", gg, 0.05, "👥", "#10b981" if gg <= 0.05 else "#ef4444"), unsafe_allow_html=True)
        with c4:
            sg = fairness_report.get("senior_gap", 1)
            st.markdown(render_gauge_card("Senior Gap", sg, 0.15, "👴", "#10b981" if sg <= 0.15 else "#ef4444"), unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════

# Check if running in Data Profiling mode and trying to view AI-related pages
if is_custom and st.session_state.get("automl_pipeline"):
    pipeline = st.session_state["automl_pipeline"]
    if pipeline.metadata.get("data_only", False) and page in ["Overview", "Model Card", "Risk Sandbox", "Risk & Fairness", "Model Performance", "Explainability", "Artifacts"]:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 50px 20px; border: 1px solid rgba(139, 92, 246, 0.2); background: rgba(15, 23, 42, 0.4);">
            <div style="font-size: 50px; margin-bottom: 20px;">🔒</div>
            <h3 style="color: #ffffff; font-family: 'Outfit', sans-serif; font-size: 24px; font-weight: 800; margin-bottom: 12px; background: linear-gradient(135deg, #ffffff, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI Engine is Offline</h3>
            <p style="color: #94a3b8; font-size: 15px; max-width: 500px; margin: 0 auto 24px auto; line-height: 1.6;">
                This page requires trained models, SHAP values, and algorithmic fairness audits. You uploaded this dataset in <strong>Data Profiling & Health Only</strong> mode.
            </p>
            <div style="font-size: 13px; color: #818cf8; font-weight: 600;">
                💡 To unlock, switch the Execution Mode to "Full AI Pipeline & Audits" in the sidebar and rerun.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

# ── OVERVIEW ──
if page == "Overview":
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("""
        <div class="glass-card animate-in">
            <div class="card-title">Executive Summary</div>
            <div class="card-subtitle">Single-screen digest for leadership and compliance reviewers.</div>
        </div>
        """, unsafe_allow_html=True)

        if is_custom:
            pipeline = st.session_state["automl_pipeline"]
            desc_text = (
                f"This **AutoML pipeline** evaluates predictive modeling for target column **'{pipeline.target_col}'** "
                f"using custom policy constraints. It checks statistical performance indicators and audits algorithmic "
                f"group parity across the demographic dimension **'{pipeline.demographic_col}'** to establish organizational compliance."
            )
        else:
            desc_text = (
                "This **governed ML system** validates production-readiness for customer churn. The engine audits precision, "
                "recall, and demographic fairness gaps (gender and age) to ensure compliance with enterprise risk management "
                "standards before controlled production deployment."
            )
            
        st.markdown(f"""
        <div style="font-size: 14px; line-height: 1.6; color: #cbd5e1; margin-bottom: 20px; font-family: 'Outfit', sans-serif;">
            {desc_text}
        </div>
        """, unsafe_allow_html=True)

        if model_metrics:
            m1, m2, m3 = st.columns(3)
            if is_custom and pipeline.task_type == "regression":
                with m1:
                    st.markdown(render_circle_gauge("Model R² Score", log_model.get('r2_score', 0), "Predictive utility", "#a78bfa", "🎯"), unsafe_allow_html=True)
                with m2:
                    st.markdown(render_metric_card("Mean Absolute Error (MAE)", f"{log_model.get('mae', 0):.3f}", "Lower is better", "#f472b6", "🔍"), unsafe_allow_html=True)
                with m3:
                    st.markdown(render_metric_card("Root Mean Squared Error (RMSE)", f"{log_model.get('rmse', 0):.3f}", "Lower is better", "#38bdf8", "⚡"), unsafe_allow_html=True)
            else:
                with m1:
                    st.markdown(render_circle_gauge("Recall", log_model.get('recall', 0), "Churn detection rate", "#a78bfa", "🎯"), unsafe_allow_html=True)
                with m2:
                    st.markdown(render_circle_gauge("Precision", log_model.get('precision', 0), "Prediction quality", "#f472b6", "🔍"), unsafe_allow_html=True)
                with m3:
                    st.markdown(render_circle_gauge("F1 Score", log_model.get('f1_score', 0), "Balanced metric", "#38bdf8", "⚡"), unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        if is_custom:
            if final_status == "CONDITIONALLY APPROVED":
                st.markdown(render_signal_banner("⚠️ <strong>Compliance Warning</strong> — The model has been conditionally approved. Predictive utility is acceptable, but fairness metrics are near threshold limits. Monitor closely for demographic drift.", "warn"), unsafe_allow_html=True)
            elif final_status == "APPROVED":
                st.markdown(render_signal_banner("✅ <strong>Governance Cleared</strong> — All gates successfully passed. The model satisfies accuracy and algorithmic parity boundaries. Ready for production deployment.", "good"), unsafe_allow_html=True)
            else:
                st.markdown(render_signal_banner("❌ <strong>Deployment Prohibited</strong> — Critical compliance failure. The model has failed to meet the mandatory accuracy or fairness thresholds. Remediation is required.", "bad"), unsafe_allow_html=True)
        else:
            if final_status == "CONDITIONALLY APPROVED":
                st.markdown(render_signal_banner("⚠️ <strong>Equity Disparity Detected</strong> — The model shows strong predictive capability, but the fairness audit detected a significant senior citizen gap. Review recommended before production release.", "warn"), unsafe_allow_html=True)
            elif final_status == "APPROVED":
                st.markdown(render_signal_banner("✅ <strong>Compliance Integrity Passed</strong> — Model performance satisfies quality thresholds and exhibits no substantial demographic gaps. Ready for rollout.", "good"), unsafe_allow_html=True)
            else:
                st.markdown(render_signal_banner("❌ <strong>Compliance Failure</strong> — The model failed to pass the required accuracy gates or contains unacceptable demographic disparities. Deployment suspended.", "bad"), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card animate-in">
            <div class="card-title">Policy Gates</div>
            <div class="card-subtitle">Rule-by-rule compliance checklist.</div>
        </div>
        """, unsafe_allow_html=True)

        if compliance_report:
            if is_custom:
                pipeline = st.session_state["automl_pipeline"]
                if pipeline.task_type == "classification":
                    st.markdown(render_compliance_item("Model Accuracy", decisions.get("model_accuracy", decisions.get("accuracy", "N/A")), f"Threshold ≥ {pipeline.compliance_report['limits']['min_accuracy']:.2f} · Actual: {log_model.get('accuracy', 0):.3f}"), unsafe_allow_html=True)
                    st.markdown(render_compliance_item("Model Recall", decisions.get("model_recall", decisions.get("recall", "N/A")), f"Threshold ≥ {pipeline.compliance_report['limits']['min_recall']:.2f} · Actual: {log_model.get('recall', 0):.3f}"), unsafe_allow_html=True)
                    st.markdown(render_compliance_item(f"{pipeline.demographic_col.title()} Fairness", decisions.get("demographic_bias_gap", decisions.get("demographic_fairness", "N/A")), f"Gap ≤ {pipeline.compliance_report['limits']['max_gap']:.2f} · Actual: {fairness_report.get('demographic_gap', 0):.3f}"), unsafe_allow_html=True)
                else:
                    st.markdown(render_compliance_item("Model R² Score", decisions.get("model_r²_score", decisions.get("r2_score", "N/A")), f"Threshold ≥ {pipeline.compliance_report['limits']['min_r2']:.2f} · Actual: {log_model.get('r2_score', 0):.3f}"), unsafe_allow_html=True)
                    st.markdown(render_compliance_item(f"{pipeline.demographic_col.title()} Fairness", decisions.get("group_error_disparity", decisions.get("demographic_fairness", "N/A")), f"Gap ≤ {pipeline.compliance_report['limits']['max_gap']:.2f} · Actual: {fairness_report.get('demographic_gap', 0):.3f}"), unsafe_allow_html=True)
            else:
                st.markdown(render_compliance_item("Model Accuracy", decisions.get("accuracy", "N/A"), f"Threshold ≥ 0.75 · Actual: {log_model.get('accuracy', 0):.3f}"), unsafe_allow_html=True)
                st.markdown(render_compliance_item("Model Recall", decisions.get("recall", "N/A"), f"Threshold ≥ 0.50 · Actual: {log_model.get('recall', 0):.3f}"), unsafe_allow_html=True)
                st.markdown(render_compliance_item("Gender Fairness", decisions.get("gender_fairness", "N/A"), f"Gap ≤ 0.05 · Actual: {fairness_report.get('gender_gap', 0):.3f}"), unsafe_allow_html=True)
                st.markdown(render_compliance_item("Senior Fairness", decisions.get("senior_fairness", "N/A"), f"Gap ≤ 0.15 · Actual: {fairness_report.get('senior_gap', 0):.3f}"), unsafe_allow_html=True)

        st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">Recommended Action</div>
            <div class="card-subtitle">Next step for governance stakeholders.</div>
        </div>
        """, unsafe_allow_html=True)

        if is_custom:
            if final_status == "APPROVED":
                st.markdown(render_signal_banner("🚀 **Enterprise-Ready Deployment**: The candidate model has met or exceeded all statistical accuracy constraints and algorithmic fairness checks. It is cleared for automated production execution under active monitoring.", "good"), unsafe_allow_html=True)
            elif final_status == "CONDITIONALLY APPROVED":
                st.markdown(render_signal_banner("🔍 **Conditional Authorization**: The model demonstrates solid predictive utility but shows marginal compliance metrics. Deployment is authorized with strict guardrails: human-in-the-loop review, low traffic volume, and daily demographic drift analysis.", "warn"), unsafe_allow_html=True)
            else:
                st.markdown(render_signal_banner("❌ **Remediation Required**: The candidate model failed critical governance thresholds. Do not deploy. Please inspect model feature coefficients, review data profiling for historical bias, and adjust preprocessing transformations before retraining.", "bad"), unsafe_allow_html=True)
        else:
            if fairness_report and fairness_report.get("senior_gap", 0) > 0.15:
                st.markdown(render_signal_banner("🔍 **Equity Disparity Lock**: Deploy forbidden. Serious validation failure. The bias gap for the senior citizen cohort is outside permissible corporate safety boundaries. Refit model pipeline or review feature correlations.", "bad"), unsafe_allow_html=True)
            else:
                st.markdown(render_signal_banner("🚀 **Production Clearance**: The churn validation pipeline indicates optimal precision/recall balance and satisfies demographic parity rules across all audited groups. Clear for release to production servers.", "good"), unsafe_allow_html=True)

# ── MODEL CARD ──
elif page == "Model Card":
    st.markdown("""
    <div class="glass-card animate-in">
        <div class="card-title">Model Card</div>
        <div class="card-subtitle">Standardized governance documentation for model transparency and accountability.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if is_custom:
        pipeline = st.session_state["automl_pipeline"]
        task_type = pipeline.task_type
        target = pipeline.target_col
        demo = pipeline.demographic_col
        metrics = pipeline.metrics
        status = pipeline.compliance_report.get("final_status", "N/A")
    else:
        task_type = "classification"
        target = "Churn"
        demo = "gender"
        metrics = model_metrics
        status = compliance_report.get("final_status", "N/A")
        
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">Model Details</div>
            <table style="width:100%; border-collapse: collapse; font-family:'Inter',sans-serif; color:#cbd5e1; font-size:14px;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0; font-weight:700; color:#818cf8;">Model Family</td><td style="padding:10px 0;">{"Logistic Regression & Random Forest" if task_type == "classification" else "Ridge Regression & Random Forest Regressor"}</td></tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0; font-weight:700; color:#818cf8;">Task Type</td><td style="padding:10px 0;">{task_type.title()}</td></tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0; font-weight:700; color:#818cf8;">Target Variable</td><td style="padding:10px 0;">{target}</td></tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0; font-weight:700; color:#818cf8;">Governance Status</td><td style="padding:10px 0; font-weight:700; color:#10b981;">{status}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">Intended Use &amp; Limitations</div>
            <p style="font-size:14px; line-height:1.6; color:#cbd5e1; margin-bottom:10px;">
                <strong>Intended Use:</strong> This model is intended for predicting {target} for validation and governance check. It is built as a candidate to audit performance and compliance benchmarks.
            </p>
            <p style="font-size:14px; line-height:1.6; color:#cbd5e1; margin-bottom:0;">
                <strong>Limitations:</strong> The model training is deterministic with fixed seed. Extreme target imbalances or missing demographic subgroups may bias model parameters. Continuous drift audits are recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.write("### 📊 Metrics Summary")
    if metrics:
        df_metrics = pd.DataFrame(metrics).T
        st.dataframe(df_metrics.style.format(precision=4), use_container_width=True)

# ── DATA CARD ──
elif page == "Data Card":
    st.markdown("""
    <div class="glass-card animate-in">
        <div class="card-title">Data Card</div>
        <div class="card-subtitle">Dataset specifications, profiling, schema overview, and data quality check results.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if is_custom:
        pipeline = st.session_state["automl_pipeline"]
        dq = pipeline.data_quality
        meta = pipeline.metadata
        dropped = dq.get("dropped_columns", [])
        num_cols = dq.get("total_columns", 0)
        num_rows = dq.get("total_rows", 0)
        missing = dq.get("total_missing_values", 0)
        dupes = dq.get("duplicate_rows", 0)
        features_schema = meta.get("features_schema", {})
    else:
        dq = quality_report
        meta = metadata_report
        dropped = ["customerID"]
        num_cols = dq.get("num_columns", 21)
        num_rows = dq.get("num_rows", 7043)
        missing = dq.get("total_missing_values", 0)
        dupes = dq.get("duplicate_rows", 0)
        features_schema = {}
        
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">Dataset Overview</div>
            <table style="width:100%; border-collapse: collapse; font-family:'Inter',sans-serif; color:#cbd5e1; font-size:14px;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0; font-weight:700; color:#818cf8;">Total Rows</td><td style="padding:10px 0;">{num_rows}</td></tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0; font-weight:700; color:#818cf8;">Total Columns</td><td style="padding:10px 0;">{num_cols}</td></tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0; font-weight:700; color:#818cf8;">Missing Values</td><td style="padding:10px 0;">{missing}</td></tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0; font-weight:700; color:#818cf8;">Duplicate Rows</td><td style="padding:10px 0;">{dupes}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">Schema &amp; Feature Actions</div>
            <p style="font-size:14px; line-height:1.6; color:#cbd5e1; margin-bottom:10px;">
                <strong>Excluded ID Columns:</strong> {', '.join(dropped) if dropped else 'None'}
            </p>
            <p style="font-size:14px; line-height:1.6; color:#cbd5e1; margin-bottom:0;">
                <strong>Protected Demographic:</strong> {meta.get('demographic_column', 'gender')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    if features_schema:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.write("### 📋 Columns Data Profiling")
        schema_rows = []
        for col_name, col_info in features_schema.items():
            col_type = col_info.get("type", "unknown")
            if col_type == "numeric":
                details = f"Range: [{col_info.get('min', 0):.2f}, {col_info.get('max', 0):.2f}] · Median: {col_info.get('median', 0):.2f}"
            else:
                details = f"Categories: {', '.join(map(str, col_info.get('categories', [])))[:80]}..."
            schema_rows.append({"Column Name": col_name, "Type": col_type, "Details": details})
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

# ── RISK SANDBOX ──
elif page == "Risk Sandbox":
    st.markdown("""
    <div class="glass-card animate-in">
        <div class="card-title">Risk Sandbox &amp; Interactive Simulator</div>
        <div class="card-subtitle">Run real-time what-if predictions, simulate decision thresholds, and explore policy compliance.</div>
    </div>
    """, unsafe_allow_html=True)

    if is_custom:
        pipeline = st.session_state["automl_pipeline"]
        tab_list = ["🔮 Custom Predictor", "🎛️ Policy Simulator"]
        tabs = st.tabs(tab_list)
        
        with tabs[0]:
            st.write(f"### 🔮 Individual {pipeline.target_col.title()} Predictor")
            st.write("Adjust the attributes below to estimate target prediction for a single instance.")
            
            c_left, c_mid, c_right = st.columns(3)
            inputs = {}
            all_feats = [(feat, "num") for feat in pipeline.numeric_cols] + [(feat, "cat") for feat in pipeline.categorical_cols]
            
            for idx, (feat, feat_type) in enumerate(all_feats):
                col_target = c_left if idx % 3 == 0 else (c_mid if idx % 3 == 1 else c_right)
                with col_target:
                    if feat_type == "num":
                        schema = pipeline.features_schema[feat]
                        val = col_target.slider(
                            feat.replace("_", " ").title(),
                            min_value=float(schema["min"]),
                            max_value=float(schema["max"]),
                            value=float(schema.get("mean", schema.get("min", 0.0))),
                            key=f"custom_in_{feat}"
                        )
                        inputs[feat] = val
                    else:
                        schema = pipeline.features_schema[feat]
                        cats = schema["categories"]
                        val = col_target.selectbox(
                            feat.replace("_", " ").title(),
                            options=cats,
                            index=cats.index(schema["mode"]) if schema["mode"] in cats else 0,
                            key=f"custom_in_{feat}"
                        )
                        inputs[feat] = val
            
            with c_right:
                model_lbl = "Logistic Regression" if pipeline.task_type == "classification" else "Linear/Ridge Regression"
                selected_model = st.selectbox("Select Model for Prediction", [model_lbl, "Random Forest"], index=0, key="custom_sandbox_model")
                submit = st.button("Calculate Prediction", use_container_width=True, key="custom_sandbox_calc")
                
            if submit:
                pred_val, factors, _ = pipeline.predict_single(inputs, selected_model)
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                r_left, r_right = st.columns([1, 1.8])
                
                with r_left:
                    st.markdown("""
                    <div class="glass-card" style="text-align: center;">
                        <div class="card-title">Prediction Result</div>
                        <div class="card-subtitle">Estimated model output.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if pipeline.task_type == "classification":
                        risk_color_hex = "#ef4444" if pred_val >= 0.7 else ("#f59e0b" if pred_val >= 0.4 else "#10b981")
                        st.markdown(render_circle_gauge("Probability", pred_val, "Model prediction probability", risk_color_hex, "🔮"), unsafe_allow_html=True)
                    else:
                        st.markdown(render_metric_card(f"Predicted {pipeline.target_col.title()}", f"{pred_val:.4f}", "Model continuous output", "#818cf8", "🔮"), unsafe_allow_html=True)
                        
                with r_right:
                    st.markdown("""
                    <div class="glass-card">
                        <div class="card-title">Key Prediction Drivers</div>
                        <div class="card-subtitle">Local explanations for this prediction.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if factors:
                        st.markdown(render_risk_factors(factors), unsafe_allow_html=True)
                    else:
                        st.info("Feature importance contribution maps are calculated for the Linear model. For Random Forest, global importances apply.")
                        
        with tabs[1]:
            saved_limits = st.session_state.get("custom_limits", {})
            
            if pipeline.task_type == "classification":
                st.write("### 🎛️ Interactive Threshold & Policy Simulator")
                st.write("Adjust the decision threshold and safety gate constraints to see live performance and policy compliance updates on the test dataset.")
                
                p1, p2 = st.columns([1, 2])
                with p1:
                    st.markdown('<div style="font-size: 13px; font-weight: 700; color: #818cf8; margin-bottom: 8px; font-family: \'Outfit\', sans-serif;">Simulation Parameters</div>', unsafe_allow_html=True)
                    sim_model = st.selectbox("Simulation Model", ["Logistic Regression", "Random Forest"], index=0, key="custom_sim_model")
                    sim_threshold = st.slider("Classification Threshold", 0.05, 0.95, 0.50, step=0.05, key="custom_sim_thresh")
                    
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<div style="font-size: 13px; font-weight: 700; color: #818cf8; margin-bottom: 8px; font-family: \'Outfit\', sans-serif;">Compliance Policy Limits</div>', unsafe_allow_html=True)
                    limit_acc = st.slider("Min Accuracy", 0.50, 0.95, saved_limits.get("min_accuracy", 0.75), step=0.05, key="custom_sim_l_acc")
                    limit_rec = st.slider("Min Recall", 0.40, 0.90, saved_limits.get("min_recall", 0.50), step=0.05, key="custom_sim_l_rec")
                    limit_gap = st.slider("Max Demographic Bias Gap", 0.01, 0.30, saved_limits.get("max_gap", 0.10), step=0.01, key="custom_sim_l_gap")
                    
                    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
                    if st.button("🔒 Lock & Apply Policy Globally", use_container_width=True, key="lock_policy_classification"):
                        limits = {
                            "min_accuracy": limit_acc,
                            "min_recall": limit_rec,
                            "max_gap": limit_gap
                        }
                        st.session_state["custom_limits"] = limits
                        pipeline.run_compliance_gate(limits)
                        st.success("Compliance policy updated globally!")
                        st.rerun()
                        
                with p2:
                    model_obj = pipeline.linear_model if sim_model == "Logistic Regression" else pipeline.forest_model
                    y_proba = model_obj.predict_proba(pipeline.X_test)[:, 1]
                    y_pred = (y_proba >= sim_threshold).astype(int)
                    
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    acc = accuracy_score(pipeline.y_test, y_pred)
                    prec = precision_score(pipeline.y_test, y_pred, zero_division=0)
                    rec = recall_score(pipeline.y_test, y_pred, zero_division=0)
                    f1 = f1_score(pipeline.y_test, y_pred, zero_division=0)
                    
                    demo_series = pipeline.original_df.loc[pipeline.X_test.index, pipeline.demographic_col]
                    df_fair = pd.DataFrame({"predicted": y_pred, "demographic": demo_series})
                    if pd.api.types.is_numeric_dtype(df_fair["demographic"]):
                        try:
                            df_fair["demographic"] = pd.qcut(df_fair["demographic"], q=3, labels=["Low", "Mid", "High"])
                        except ValueError:
                            df_fair["demographic"] = pd.cut(df_fair["demographic"], bins=3, labels=["Low", "Mid", "High"])
                            
                    rates = df_fair.groupby("demographic")["predicted"].mean().to_dict()
                    gap = abs(max(rates.values()) - min(rates.values())) if rates else 0.0
                    
                    dec_acc = "PASS" if acc >= limit_acc else "FAIL"
                    dec_rec = "PASS" if rec >= limit_rec else "FAIL"
                    dec_gap = "PASS" if gap <= limit_gap else "FAIL"
                    passed_gates = [dec_acc, dec_rec, dec_gap].count("PASS")
                    
                    if passed_gates == 3:
                        sim_status = "APPROVED"
                    elif dec_acc == "PASS" and dec_rec == "PASS":
                        sim_status = "CONDITIONALLY APPROVED"
                    else:
                        sim_status = "REJECTED"
                        
                    fc_sim = status_color(sim_status)
                    pulse_class_sim = "pulse-green" if sim_status == "APPROVED" else ("pulse-amber" if sim_status == "CONDITIONALLY APPROVED" else "pulse-red")
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(145deg, rgba(10,14,24,0.8), rgba(6,9,16,0.9));
                        border: 1px solid rgba(255,255,255,0.04);
                        border-radius: 20px; padding: 24px;
                        margin-bottom: 24px;
                        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
                        display: flex; align-items: center; justify-content: space-between;
                    ">
                        <div>
                            <div style="font-size: 11px; color: #64748b !important; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">Simulated Status</div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <div class="{pulse_class_sim}" style="width: 10px; height: 10px; border-radius: 50%; background: {fc_sim}; box-shadow: 0 0 14px {fc_sim};"></div>
                                <span style="font-size: 18px; font-weight: 900; color: {fc_sim} !important; font-family: 'Outfit', sans-serif;">{sim_status}</span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 11px; color: #64748b !important; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">Compliance Score</div>
                            <div style="font-size: 24px; font-weight: 900; color: #ffffff !important; font-family: 'Outfit', sans-serif;">{passed_gates}/3 <span style="font-size: 14px; color: #475569; font-weight: 600;">Gates Passed</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    sm1, sm2, sm3, sm4 = st.columns(4)
                    with sm1:
                        st.markdown(render_circle_gauge("Simulated Accuracy", acc, f"Limit: ≥ {limit_acc:.2f}", "#10b981" if dec_acc == "PASS" else "#ef4444", "📈"), unsafe_allow_html=True)
                    with sm2:
                        st.markdown(render_circle_gauge("Simulated Recall", rec, f"Limit: ≥ {limit_rec:.2f}", "#10b981" if dec_rec == "PASS" else "#ef4444", "🎯"), unsafe_allow_html=True)
                    with sm3:
                        st.markdown(render_circle_gauge("Simulated Precision", prec, "Quality rate", "#f472b6", "🔍"), unsafe_allow_html=True)
                    with sm4:
                        st.markdown(render_circle_gauge("Simulated F1 Score", f1, "Balanced metric", "#38bdf8", "⚡"), unsafe_allow_html=True)
                        
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    
                    fc_left, fc_right = st.columns(2)
                    with fc_left:
                        st.markdown("""
                        <div class="glass-card">
                            <div class="card-title">Simulated Bias Gap</div>
                            <div class="card-subtitle">Demographic prediction disparity.</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(render_gauge_card(f"{pipeline.demographic_col.title()} Gap", gap, limit_gap, "👥", "#10b981" if dec_gap == "PASS" else "#ef4444"), unsafe_allow_html=True)
                        
                    with fc_right:
                        st.markdown("""
                        <div class="glass-card">
                            <div class="card-title">Compliance Gate Checklist</div>
                            <div class="card-subtitle">Real-time simulated check rules.</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(render_compliance_item("Model Accuracy", dec_acc, f"Threshold ≥ {limit_acc:.2f} · Actual: {acc:.3f}"), unsafe_allow_html=True)
                        st.markdown(render_compliance_item("Model Recall", dec_rec, f"Threshold ≥ {limit_rec:.2f} · Actual: {rec:.3f}"), unsafe_allow_html=True)
                        st.markdown(render_compliance_item(f"{pipeline.demographic_col.title()} Fairness", dec_gap, f"Gap ≤ {limit_gap:.2f} · Actual: {gap:.3f}"), unsafe_allow_html=True)
            
            else:  # regression
                st.write("### 🎛️ Interactive Policy Simulator (Regression)")
                st.write("Adjust the compliance bounds for your continuous target model to see policy updates.")
                
                p1, p2 = st.columns([1, 2])
                with p1:
                    st.markdown('<div style="font-size: 13px; font-weight: 700; color: #818cf8; margin-bottom: 8px; font-family: \'Outfit\', sans-serif;">Simulation Parameters</div>', unsafe_allow_html=True)
                    sim_model = st.selectbox("Simulation Model", ["Ridge Regression", "Random Forest"], index=0, key="custom_sim_model_reg")
                    
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<div style="font-size: 13px; font-weight: 700; color: #818cf8; margin-bottom: 8px; font-family: \'Outfit\', sans-serif;">Compliance Policy Limits</div>', unsafe_allow_html=True)
                    limit_r2 = st.slider("Min R² Score", 0.10, 0.95, saved_limits.get("min_r2", 0.50), step=0.05, key="custom_sim_l_r2")
                    limit_gap = st.slider("Max Demographic Bias Gap", 0.01, 0.30, saved_limits.get("max_gap", 0.10), step=0.01, key="custom_sim_l_gap_reg")
                    
                    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
                    if st.button("🔒 Lock & Apply Policy Globally", use_container_width=True, key="lock_policy_regression"):
                        limits = {
                            "min_r2": limit_r2,
                            "max_gap": limit_gap
                        }
                        st.session_state["custom_limits"] = limits
                        pipeline.run_compliance_gate(limits)
                        st.success("Compliance policy updated globally!")
                        st.rerun()
                        
                with p2:
                    model_metrics_reg = pipeline.metrics.get("linear_regression" if sim_model == "Ridge Regression" else "random_forest", {})
                    r2_val = model_metrics_reg.get("r2_score", 0.0)
                    mae_val = model_metrics_reg.get("mae", 0.0)
                    rmse_val = model_metrics_reg.get("rmse", 0.0)
                    
                    gap_val = pipeline.fairness_report.get("demographic_gap", 0.0)
                    
                    dec_r2 = "PASS" if r2_val >= limit_r2 else "FAIL"
                    dec_gap = "PASS" if gap_val <= limit_gap else "FAIL"
                    passed_gates = [dec_r2, dec_gap].count("PASS")
                    
                    if passed_gates == 2:
                        sim_status = "APPROVED"
                    elif dec_r2 == "PASS":
                        sim_status = "CONDITIONALLY APPROVED"
                    else:
                        sim_status = "REJECTED"
                        
                    fc_sim = status_color(sim_status)
                    pulse_class_sim = "pulse-green" if sim_status == "APPROVED" else ("pulse-amber" if sim_status == "CONDITIONALLY APPROVED" else "pulse-red")
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(145deg, rgba(10,14,24,0.8), rgba(6,9,16,0.9));
                        border: 1px solid rgba(255,255,255,0.04);
                        border-radius: 20px; padding: 24px;
                        margin-bottom: 24px;
                        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
                        display: flex; align-items: center; justify-content: space-between;
                    ">
                        <div>
                            <div style="font-size: 11px; color: #64748b !important; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">Simulated Status</div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <div class="{pulse_class_sim}" style="width: 10px; height: 10px; border-radius: 50%; background: {fc_sim}; box-shadow: 0 0 14px {fc_sim};"></div>
                                <span style="font-size: 18px; font-weight: 900; color: {fc_sim} !important; font-family: 'Outfit', sans-serif;">{sim_status}</span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 11px; color: #64748b !important; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">Compliance Score</div>
                            <div style="font-size: 24px; font-weight: 900; color: #ffffff !important; font-family: 'Outfit', sans-serif;">{passed_gates}/2 <span style="font-size: 14px; color: #475569; font-weight: 600;">Gates Passed</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    sm1, sm2, sm3 = st.columns(3)
                    with sm1:
                        st.markdown(render_circle_gauge("Simulated R² Score", r2_val, f"Limit: ≥ {limit_r2:.2f}", "#10b981" if dec_r2 == "PASS" else "#ef4444", "📈"), unsafe_allow_html=True)
                    with sm2:
                        st.markdown(render_circle_gauge("MAE", mae_val, "Mean absolute error", "#38bdf8", "🔍"), unsafe_allow_html=True)
                    with sm3:
                        st.markdown(render_circle_gauge("RMSE", rmse_val, "Root mean squared error", "#f472b6", "🌲"), unsafe_allow_html=True)
                        
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    
                    fc_left, fc_right = st.columns(2)
                    with fc_left:
                        st.markdown("""
                        <div class="glass-card">
                            <div class="card-title">Simulated Bias Gap</div>
                            <div class="card-subtitle">Demographic average prediction disparity.</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(render_gauge_card(f"{pipeline.demographic_col.title()} Gap", gap_val, limit_gap, "👥", "#10b981" if dec_gap == "PASS" else "#ef4444"), unsafe_allow_html=True)
                        
                    with fc_right:
                        st.markdown("""
                        <div class="glass-card">
                            <div class="card-title">Compliance Gate Checklist</div>
                            <div class="card-subtitle">Real-time simulated check rules.</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(render_compliance_item("Model R² Score", dec_r2, f"Threshold ≥ {limit_r2:.2f} · Actual: {r2_val:.3f}"), unsafe_allow_html=True)
                        st.markdown(render_compliance_item(f"{pipeline.demographic_col.title()} Fairness", dec_gap, f"Gap ≤ {limit_gap:.2f} · Actual: {gap_val:.3f}"), unsafe_allow_html=True)
    else:
        tab1, tab2 = st.tabs(["🔮 Customer Risk Playground", "🎛️ Threshold & Policy Simulator"])
        
        with tab1:
            st.write("### 🔮 Individual Customer Risk Predictor")
            st.write("Adjust the attributes below to estimate churn probability for a single customer.")
            
            c_dem, c_srv, c_bill = st.columns(3)
            
            with c_dem:
                st.markdown('<div style="font-size: 13px; font-weight: 700; color: #818cf8; margin-bottom: 8px; font-family: \'Outfit\', sans-serif;">Demographics</div>', unsafe_allow_html=True)
                gender = st.selectbox("Gender", ["Male", "Female"], index=1)
                senior = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
                partner = st.selectbox("Partner", ["No", "Yes"], index=1)
                dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
                paperless = st.selectbox("Paperless Billing", ["No", "Yes"], index=1)
                
            with c_srv:
                st.markdown('<div style="font-size: 13px; font-weight: 700; color: #818cf8; margin-bottom: 8px; font-family: \'Outfit\', sans-serif;">Services &amp; Usage</div>', unsafe_allow_html=True)
                tenure = st.slider("Tenure (Months)", 0, 72, 12)
                phone = st.selectbox("Phone Service", ["No", "Yes"], index=1)
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0)
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1)
                security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0)
                backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=0)
                protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0)
                support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0)
                tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=0)
                movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=0)
                
            with c_bill:
                st.markdown('<div style="font-size: 13px; font-weight: 700; color: #818cf8; margin-bottom: 8px; font-family: \'Outfit\', sans-serif;">Billing &amp; Model</div>', unsafe_allow_html=True)
                monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
                payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
                
                total_charges = tenure * monthly_charges if tenure > 0 else monthly_charges
                st.text_input("Estimated Total Charges ($)", value=f"{total_charges:.2f}", disabled=True)
                
                selected_model = st.selectbox("Select Model for Prediction", ["Logistic Regression", "Random Forest"], index=0)
                submit = st.button("Calculate Churn Risk", use_container_width=True)

            if submit:
                inputs = {
                    'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
                    'tenure': tenure, 'PhoneService': phone, 'PaperlessBilling': paperless,
                    'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
                    'MultipleLines': multiple_lines, 'InternetService': internet,
                    'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection,
                    'TechSupport': support, 'StreamingTV': tv, 'StreamingMovies': movies,
                    'Contract': contract, 'PaymentMethod': payment
                }
                
                proba, factors = predict_single_customer(inputs, selected_model)
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                r_left, r_right = st.columns([1, 1.8])
                
                with r_left:
                    st.markdown("""
                    <div class="glass-card" style="text-align: center;">
                        <div class="card-title">Prediction Result</div>
                        <div class="card-subtitle">Estimated probability of customer churn.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if proba >= 0.7:
                        risk_color_hex = "#ef4444"
                    elif proba >= 0.4:
                        risk_color_hex = "#f59e0b"
                    else:
                        risk_color_hex = "#10b981"
                        
                    st.markdown(render_circle_gauge("Churn Risk", proba, "Model prediction probability", risk_color_hex, "🔮"), unsafe_allow_html=True)
                    
                with r_right:
                    st.markdown("""
                    <div class="glass-card">
                        <div class="card-title">Key Risk Drivers</div>
                        <div class="card-subtitle">Local explanations for this customer's score.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if factors:
                        st.markdown(render_risk_factors(factors), unsafe_allow_html=True)
                    else:
                        st.info("Feature importance contribution maps are only calculated for Logistic Regression. For Random Forest, global feature importances apply.")

        with tab2:
            st.write("### 🎛️ Interactive Threshold & Policy Simulator")
            st.write("Adjust the decision threshold and safety gate constraints to see live performance and policy compliance updates on the test dataset.")
            
            X_test, y_test, gender_test, senior_test = load_sandbox_data()
            
            if X_test is not None:
                p1, p2 = st.columns([1, 2])
                
                with p1:
                    st.markdown('<div style="font-size: 13px; font-weight: 700; color: #818cf8; margin-bottom: 8px; font-family: \'Outfit\', sans-serif;">Simulation Parameters</div>', unsafe_allow_html=True)
                    sim_model = st.selectbox("Simulation Model", ["Logistic Regression", "Random Forest"], index=0)
                    sim_threshold = st.slider("Classification Threshold", 0.05, 0.95, 0.50, step=0.05)
                    
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<div style="font-size: 13px; font-weight: 700; color: #818cf8; margin-bottom: 8px; font-family: \'Outfit\', sans-serif;">Compliance Policy Limits</div>', unsafe_allow_html=True)
                    limit_acc = st.slider("Min Accuracy", 0.50, 0.95, 0.75, step=0.05)
                    limit_rec = st.slider("Min Recall", 0.40, 0.90, 0.50, step=0.05)
                    limit_gender = st.slider("Max Gender Bias Gap", 0.01, 0.20, 0.05, step=0.01)
                    limit_senior = st.slider("Max Senior Bias Gap", 0.01, 0.30, 0.15, step=0.01)
                    
                with p2:
                    import joblib
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    try:
                        serialized_path = Path(__file__).resolve().parent.parent / "models" / "serialized_pipeline.joblib"
                        if serialized_path.exists():
                            saved_data = joblib.load(serialized_path)
                            if sim_model == "Logistic Regression":
                                model_obj = saved_data["linear"].named_steps['model']
                            else:
                                model_obj = saved_data["ensemble"].named_steps['model']
                        else:
                            raise FileNotFoundError("models/serialized_pipeline.joblib not found. Please run the pipeline.")
                            
                        y_proba = model_obj.predict_proba(X_test)[:, 1]
                        y_pred = (y_proba >= sim_threshold).astype(int)
                        
                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, zero_division=0)
                        rec = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        
                        df_fair = pd.DataFrame({
                            "predicted": y_pred,
                            "gender": gender_test,
                            "SeniorCitizen": senior_test
                        })
                        
                        gender_rates = df_fair.groupby("gender")["predicted"].mean().to_dict()
                        gender_gap = abs(gender_rates.get("Female", 0) - gender_rates.get("Male", 0))
                        
                        senior_rates = df_fair.groupby("SeniorCitizen")["predicted"].mean().to_dict()
                        senior_gap = abs(senior_rates.get(0, 0) - senior_rates.get(1, 0))
                        
                        dec_acc = "PASS" if acc >= limit_acc else "FAIL"
                        dec_rec = "PASS" if rec >= limit_rec else "FAIL"
                        dec_gen = "PASS" if gender_gap <= limit_gender else "FAIL"
                        dec_sen = "PASS" if senior_gap <= limit_senior else "FAIL"
                        
                        passed_gates = [dec_acc, dec_rec, dec_gen, dec_sen].count("PASS")
                        
                        if passed_gates == 4:
                            sim_status = "APPROVED"
                        elif dec_acc == "PASS" and dec_rec == "PASS":
                            sim_status = "CONDITIONALLY APPROVED"
                        else:
                            sim_status = "REJECTED"
                            
                        fc_sim = status_color(sim_status)
                        pulse_class_sim = "pulse-green" if sim_status == "APPROVED" else ("pulse-amber" if sim_status == "CONDITIONALLY APPROVED" else "pulse-red")
                        
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(145deg, rgba(10,14,24,0.8), rgba(6,9,16,0.9));
                            border: 1px solid rgba(255,255,255,0.04);
                            border-radius: 20px; padding: 24px;
                            margin-bottom: 24px;
                            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
                            display: flex; align-items: center; justify-content: space-between;
                        ">
                            <div>
                                <div style="font-size: 11px; color: #64748b !important; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">Simulated Status</div>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <div class="{pulse_class_sim}" style="width: 10px; height: 10px; border-radius: 50%; background: {fc_sim}; box-shadow: 0 0 14px {fc_sim};"></div>
                                    <span style="font-size: 18px; font-weight: 900; color: {fc_sim} !important; font-family: 'Outfit', sans-serif;">{sim_status}</span>
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 11px; color: #64748b !important; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">Compliance Score</div>
                                <div style="font-size: 24px; font-weight: 900; color: #ffffff !important; font-family: 'Outfit', sans-serif;">{passed_gates}/4 <span style="font-size: 14px; color: #475569; font-weight: 600;">Gates Passed</span></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        sm1, sm2, sm3, sm4 = st.columns(4)
                        with sm1:
                            st.markdown(render_circle_gauge("Simulated Accuracy", acc, f"Limit: ≥ {limit_acc:.2f}", "#10b981" if dec_acc == "PASS" else "#ef4444", "📈"), unsafe_allow_html=True)
                        with sm2:
                            st.markdown(render_circle_gauge("Simulated Recall", rec, f"Limit: ≥ {limit_rec:.2f}", "#10b981" if dec_rec == "PASS" else "#ef4444", "🎯"), unsafe_allow_html=True)
                        with sm3:
                            st.markdown(render_circle_gauge("Simulated Precision", prec, "Quality rate", "#f472b6", "🔍"), unsafe_allow_html=True)
                        with sm4:
                            st.markdown(render_circle_gauge("Simulated F1 Score", f1, "Balanced metric", "#38bdf8", "⚡"), unsafe_allow_html=True)
                            
                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        
                        fc_left, fc_right = st.columns(2)
                        
                        with fc_left:
                            st.markdown("""
                            <div class="glass-card">
                                <div class="card-title">Simulated Bias Gaps</div>
                                <div class="card-subtitle">Demographic prediction disparity.</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(render_gauge_card("Gender Gap", gender_gap, limit_gender, "👥", "#10b981" if dec_gen == "PASS" else "#ef4444"), unsafe_allow_html=True)
                            st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
                            st.markdown(render_gauge_card("Senior Gap", senior_gap, limit_senior, "👴", "#10b981" if dec_sen == "PASS" else "#ef4444"), unsafe_allow_html=True)
                            
                        with fc_right:
                            st.markdown("""
                            <div class="glass-card">
                                <div class="card-title">Compliance Gate Checklist</div>
                                <div class="card-subtitle">Real-time simulated check rules.</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(render_compliance_item("Model Accuracy", dec_acc, f"Threshold ≥ {limit_acc:.2f} · Actual: {acc:.3f}"), unsafe_allow_html=True)
                            st.markdown(render_compliance_item("Model Recall", dec_rec, f"Threshold ≥ {limit_rec:.2f} · Actual: {rec:.3f}"), unsafe_allow_html=True)
                            st.markdown(render_compliance_item("Gender Fairness", dec_gen, f"Gap ≤ {limit_gender:.2f} · Actual: {gender_gap:.3f}"), unsafe_allow_html=True)
                            st.markdown(render_compliance_item("Senior Fairness", dec_sen, f"Gap ≤ {limit_senior:.2f} · Actual: {senior_gap:.3f}"), unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Error running threshold simulation: {e}")


# ── RISK & FAIRNESS ──
elif page == "Risk & Fairness":
    st.markdown("""
    <div class="glass-card animate-in">
        <div class="card-title">Fairness Risk Profile</div>
        <div class="card-subtitle">Prediction disparities across demographic groups. Gaps may indicate model bias.</div>
    </div>
    """, unsafe_allow_html=True)

    if fairness_report:
        if is_custom:
            pipeline = st.session_state["automl_pipeline"]
            gap = fairness_report.get("demographic_gap", 0)
            st.write(f"### Demographic Audited Column: **{pipeline.demographic_col}**")
            
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.markdown(render_gauge_card(f"{pipeline.demographic_col.title()} Gap", gap, 0.10, "👥", "#10b981" if gap <= 0.10 else "#ef4444"), unsafe_allow_html=True)
            with c2:
                rates = {}
                group_reports = fairness_report.get("group_reports", {})
                for g, r_dict in group_reports.items():
                    if "selection_rate" in r_dict:
                        rates[g] = r_dict["selection_rate"]
                    elif "mean_prediction" in r_dict:
                        rates[g] = r_dict["mean_prediction"]
                    elif "accuracy" in r_dict:
                        rates[g] = r_dict["accuracy"]
                if rates:
                    title_chart = "Predicted Average Value" if pipeline.task_type == "regression" else "Predicted Positive Rate"
                    fig = dark_bar_chart(list(rates.keys()), list(rates.values()), "#818cf8", f"{title_chart} by {pipeline.demographic_col.title()}", value_fmt=".4f")
                    st.pyplot(fig); plt.close(fig)
                    
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            if gap > 0.10:
                st.markdown(render_signal_banner(f"⚠️ <strong>Demographic fairness gap on '{pipeline.demographic_col}' exceeds threshold of 0.10.</strong> Review bias and gate metrics.", "warn"), unsafe_allow_html=True)
            else:
                st.markdown(render_signal_banner("✅ <strong>All fairness thresholds within acceptable range.</strong>", "good"), unsafe_allow_html=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                gg = fairness_report.get("gender_gap", 0)
                st.markdown(render_gauge_card("Gender Gap", gg, 0.05, "👥", "#10b981" if gg <= 0.05 else "#ef4444"), unsafe_allow_html=True)
                st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
                gd = fairness_report.get("gender_churn_rate", {})
                if gd:
                    fig = dark_bar_chart(list(gd.keys()), list(gd.values()), "#34d399", "Predicted Churn Rate by Gender", value_fmt=".4f")
                    st.pyplot(fig); plt.close(fig)

            with col2:
                sg = fairness_report.get("senior_gap", 0)
                st.markdown(render_gauge_card("Senior Gap", sg, 0.15, "👴", "#10b981" if sg <= 0.15 else "#ef4444"), unsafe_allow_html=True)
                st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
                sd = fairness_report.get("senior_churn_rate", {})
                if sd:
                    labels = ["Non-Senior" if str(k) == "0" else "Senior" for k in sd.keys()]
                    fig = dark_bar_chart(labels, list(sd.values()), "#fb7185", "Predicted Churn Rate by Senior Status", value_fmt=".4f")
                    st.pyplot(fig); plt.close(fig)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            if fairness_report.get("senior_gap", 0) > 0.15:
                st.markdown(render_signal_banner("⚠️ <strong>Fairness risk concentrated in senior citizen segment.</strong> Primary blocker to full approval. Consider reweighting, feature removal, or threshold adjustment.", "bad"), unsafe_allow_html=True)
            else:
                st.markdown(render_signal_banner("✅ <strong>All fairness thresholds within acceptable range.</strong>", "good"), unsafe_allow_html=True)


# ── MODEL PERFORMANCE ──
elif page == "Model Performance":
    st.markdown("""
    <div class="glass-card animate-in">
        <div class="card-title">Model Performance Review</div>
        <div class="card-subtitle">Candidate model evaluation and selected production model.</div>
    </div>
    """, unsafe_allow_html=True)

    if is_custom:
        pipeline = st.session_state["automl_pipeline"]
        if pipeline.task_type == "classification":
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(render_circle_gauge("LR Accuracy", log_model.get('accuracy', 0), "Selected model", "#818cf8", "🎯"), unsafe_allow_html=True)
            with m2:
                st.markdown(render_circle_gauge("LR Recall", log_model.get('recall', 0), "Churn detection", "#a78bfa", "📡"), unsafe_allow_html=True)
            with m3:
                st.markdown(render_circle_gauge("RF Accuracy", rf_model.get('accuracy', 0), "Alternative", "#475569", "🌲"), unsafe_allow_html=True)
            with m4:
                st.markdown(render_circle_gauge("RF Recall", rf_model.get('recall', 0), "Alternative", "#475569", "🌲"), unsafe_allow_html=True)

            st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)
            metrics_list = ["Accuracy", "Precision", "Recall", "F1 Score"]
            lv = [log_model.get("accuracy", 0), log_model.get("precision", 0), log_model.get("recall", 0), log_model.get("f1_score", 0)]
            rv = [rf_model.get("accuracy", 0), rf_model.get("precision", 0), rf_model.get("recall", 0), rf_model.get("f1_score", 0)]
            fig = dark_grouped_bar_chart(metrics_list, lv, rv, "Logistic Regression", "Random Forest", title="Comparative Model Metrics")
            st.pyplot(fig); plt.close(fig)
            st.markdown(render_badge("✓ Selected Model: Logistic Regression", "#4f46e5"), unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">Confusion Matrix</div>
                <div class="card-subtitle">Classification performance on holdout test set.</div>
            </div>
            """, unsafe_allow_html=True)

            y_pred = pipeline.linear_model.predict(pipeline.X_test)
            cl, cr = st.columns([1, 1.5])
            with cl:
                fig = dark_confusion_matrix(pipeline.y_test, y_pred)
                st.pyplot(fig); plt.close(fig)
            with cr:
                tp = int(((pipeline.y_test == 1) & (y_pred == 1)).sum())
                fn = int(((pipeline.y_test == 1) & (y_pred == 0)).sum())
                tn = int(((pipeline.y_test == 0) & (y_pred == 0)).sum())
                fp = int(((pipeline.y_test == 0) & (y_pred == 1)).sum())
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown(render_metric_card("True Positives", str(tp), "Correctly predicted positive", "#10b981", "✅"), unsafe_allow_html=True)
                with r2:
                    st.markdown(render_metric_card("True Negatives", str(tn), "Correctly predicted negative", "#38bdf8", "✅"), unsafe_allow_html=True)
                st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
                r3, r4 = st.columns(2)
                with r3:
                    st.markdown(render_metric_card("False Positives", str(fp), "False alarms", "#f59e0b", "⚠️"), unsafe_allow_html=True)
                with r4:
                    st.markdown(render_metric_card("False Negatives", str(fn), "Missed predictions", "#ef4444", "❌"), unsafe_allow_html=True)
        else:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(render_circle_gauge("Ridge R² Score", log_model.get('r2_score', 0), "Selected model", "#818cf8", "🎯"), unsafe_allow_html=True)
            with m2:
                st.markdown(render_metric_card("Ridge MAE", f"{log_model.get('mae', 0):.3f}", "Mean Absolute Error", "#a78bfa", "📉"), unsafe_allow_html=True)
            with m3:
                st.markdown(render_circle_gauge("RF R² Score", rf_model.get('r2_score', 0), "Alternative", "#38bdf8", "🌲"), unsafe_allow_html=True)

            st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)
            metrics_list = ["R² Score"]
            lv = [log_model.get("r2_score", 0)]
            rv = [rf_model.get("r2_score", 0)]
            fig = dark_grouped_bar_chart(metrics_list, lv, rv, "Ridge Regression", "Random Forest Regressor", title="Comparative R² Score")
            st.pyplot(fig); plt.close(fig)
            st.markdown(render_badge("✓ Selected Model: Ridge Regression", "#4f46e5"), unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">Residual Analysis Plot</div>
                <div class="card-subtitle">Predicted vs Actual values on holdout test set.</div>
            </div>
            """, unsafe_allow_html=True)

            y_pred = pipeline.linear_model.predict(pipeline.X_test)
            cl, cr = st.columns([1.2, 1])
            with cl:
                fig = dark_residuals_plot(pipeline.y_test, y_pred)
                st.pyplot(fig); plt.close(fig)
            with cr:
                st.markdown(render_metric_card("Ridge MAE", f"{log_model.get('mae', 0):.4f}", "Mean Absolute Error", "#818cf8", "📐"), unsafe_allow_html=True)
                st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
                st.markdown(render_metric_card("Ridge RMSE", f"{log_model.get('rmse', 0):.4f}", "Root Mean Squared Error", "#10b981", "⚡"), unsafe_allow_html=True)
    else:
        if model_metrics:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(render_circle_gauge("LR Accuracy", log_model.get('accuracy', 0), "Selected model", "#818cf8", "🎯"), unsafe_allow_html=True)
            with m2:
                st.markdown(render_circle_gauge("LR Recall", log_model.get('recall', 0), "Churn detection", "#a78bfa", "📡"), unsafe_allow_html=True)
            with m3:
                st.markdown(render_circle_gauge("RF Accuracy", rf_model.get('accuracy', 0), "Alternative", "#475569", "🌲"), unsafe_allow_html=True)
            with m4:
                st.markdown(render_circle_gauge("RF Recall", rf_model.get('recall', 0), "Alternative", "#475569", "🌲"), unsafe_allow_html=True)

            st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)
            metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
            lv = [log_model.get("accuracy", 0), log_model.get("precision", 0), log_model.get("recall", 0), log_model.get("f1_score", 0)]
            rv = [rf_model.get("accuracy", 0), rf_model.get("precision", 0), rf_model.get("recall", 0), rf_model.get("f1_score", 0)]
            fig = dark_grouped_bar_chart(metrics, lv, rv, "Logistic Regression", "Random Forest", title="Comparative Model Metrics")
            st.pyplot(fig); plt.close(fig)
            st.markdown(render_badge("✓ Selected Model: Logistic Regression", "#4f46e5"), unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">Confusion Matrix</div>
            <div class="card-subtitle">Classification performance on holdout test set.</div>
        </div>
        """, unsafe_allow_html=True)

        try:
            serialized_path = MODELS_DIR / "serialized_pipeline.joblib"
            if serialized_path.exists():
                saved_data = joblib.load(serialized_path)
                X_test = saved_data["X_test"]
                y_test = saved_data["y_test"]
                model_obj = saved_data["linear"].named_steps['model']
                y_pred = model_obj.predict(X_test)
            else:
                pd_dir = BASE_DIR / "data" / "processed"
                X_test = pd.read_csv(pd_dir / "X_test.csv", index_col=0)
                y_test = pd.read_csv(pd_dir / "y_test.csv", index_col=0).iloc[:, 0]
                raise FileNotFoundError("models/serialized_pipeline.joblib not found.")

            cl, cr = st.columns([1, 1.5])
            with cl:
                fig = dark_confusion_matrix(y_test, y_pred)
                st.pyplot(fig); plt.close(fig)
            with cr:
                tp = int(((y_test == 1) & (y_pred == 1)).sum())
                fn = int(((y_test == 1) & (y_pred == 0)).sum())
                tn = int(((y_test == 0) & (y_pred == 0)).sum())
                fp = int(((y_test == 0) & (y_pred == 1)).sum())
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown(render_metric_card("True Positives", str(tp), "Correctly predicted churn", "#10b981", "✅"), unsafe_allow_html=True)
                with r2:
                    st.markdown(render_metric_card("True Negatives", str(tn), "Correctly predicted no churn", "#38bdf8", "✅"), unsafe_allow_html=True)
                st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
                r3, r4 = st.columns(2)
                with r3:
                    st.markdown(render_metric_card("False Positives", str(fp), "False alarms", "#f59e0b", "⚠️"), unsafe_allow_html=True)
                with r4:
                    st.markdown(render_metric_card("False Negatives", str(fn), "Missed churns", "#ef4444", "❌"), unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not render confusion matrix: {e}")


# ── EXPLAINABILITY ──
elif page == "Explainability":
    st.markdown("""
    <div class="glass-card animate-in">
        <div class="card-title">Explainability (SHAP)</div>
        <div class="card-subtitle">Feature contributions ranked by mean absolute SHAP value. What drives predictions.</div>
    </div>
    """, unsafe_allow_html=True)

    if is_custom:
        pipeline = st.session_state["automl_pipeline"]
        fd = explainability_report.get("shap_importances", [])
        if fd:
            cl, cr = st.columns([1.3, 1])
            with cl:
                fig = pipeline.get_shap_summary_plot()
                if fig:
                    st.pyplot(fig); plt.close(fig)
            with cr:
                st.markdown("""
                <div class="glass-card">
                    <div class="card-title">Top Drivers Table</div>
                    <div class="card-subtitle">Ranked feature contributions.</div>
                </div>
                """, unsafe_allow_html=True)
                sdf = pd.DataFrame(fd)
                sdf.columns = ["Feature", "Importance"]
                sdf.index = range(1, len(sdf) + 1)
                sdf.index.name = "Rank"
                st.markdown(render_html_table(sdf), unsafe_allow_html=True)
    else:
        if explainability_report:
            fd = explainability_report.get("top_features_by_mean_abs_shap", [])
            if fd:
                cl, cr = st.columns([1.3, 1])
                with cl:
                    fig = dark_bar_chart([f["feature"] for f in fd], [f["importance"] for f in fd], "#a78bfa", "Top SHAP Feature Importances", horizontal=True, value_fmt=".4f")
                    st.pyplot(fig); plt.close(fig)
                with cr:
                    st.markdown("""
                    <div class="glass-card">
                        <div class="card-title">Top Drivers Table</div>
                        <div class="card-subtitle">Ranked feature contributions.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    sdf = pd.DataFrame(fd)
                    sdf.columns = ["Feature", "Importance"]
                    sdf.index = range(1, len(sdf) + 1)
                    sdf.index.name = "Rank"
                    st.markdown(render_html_table(sdf), unsafe_allow_html=True)

        if shap_img.exists():
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">SHAP Summary Plot</div>
                <div class="card-subtitle">Global feature impact. Red = high feature value, blue = low.</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(str(shap_img), use_container_width=True)


# ── DATA HEALTH ──
elif page == "Data Health":
    st.markdown("""
    <div class="glass-card animate-in">
        <div class="card-title">Data Health Report</div>
        <div class="card-subtitle">Quality validation for the source dataset before model training.</div>
    </div>
    """, unsafe_allow_html=True)

    if quality_report:
        q1, q2, q3, q4, q5 = st.columns(5)
        with q1:
            st.markdown(render_metric_card("Rows", str(quality_report.get("num_rows", "N/A")), "Total records", "#6366f1", "📋"), unsafe_allow_html=True)
        with q2:
            st.markdown(render_metric_card("Columns", str(quality_report.get("num_columns", "N/A")), "Feature count", "#8b5cf6", "📐"), unsafe_allow_html=True)
        with q3:
            mv = quality_report.get("total_missing_values", 0)
            st.markdown(render_metric_card("Missing", str(mv), "Null cells", "#10b981" if mv == 0 else "#f59e0b", "🔍"), unsafe_allow_html=True)
        with q4:
            dp = quality_report.get("duplicate_rows", 0)
            st.markdown(render_metric_card("Duplicates", str(dp), "Duplicate rows", "#10b981" if dp == 0 else "#f59e0b", "📑"), unsafe_allow_html=True)
        with q5:
            qs = quality_report.get("quality_score", 0)
            color = "#10b981" if qs >= 80 else "#ef4444"
            st.markdown(render_metric_card("Score", f"{qs} / 100", "Overall health score", color, "⭐"), unsafe_allow_html=True)
            st.markdown(render_progress_bar(qs, color), unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        status = quality_report.get("status", "N/A")
        if status == "PASS":
            st.markdown(render_signal_banner("✅ <strong>Dataset passed quality validation.</strong> No critical issues. Ready for training.", "good"), unsafe_allow_html=True)
        else:
            st.markdown(render_signal_banner("❌ <strong>Dataset failed quality validation.</strong> Review missing values and duplicates.", "bad"), unsafe_allow_html=True)

    if metadata_report:
        st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">Dataset Metadata</div>
            <div class="card-subtitle">Identity, ownership, and catalog information.</div>
        """, unsafe_allow_html=True)
        st.markdown(render_key_value_list(metadata_report), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Render feature drift PSI metrics in the "Data Health" tab.
    if is_custom and hasattr(pipeline, "drift_report") and pipeline.drift_report:
        st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">Population Stability Index (PSI) Feature Drift</div>
            <div class="card-subtitle">Comparison of feature distributions between baseline train split and holdout test set.</div>
        </div>
        """, unsafe_allow_html=True)
        
        drift_rows = []
        for col_name, drift_info in pipeline.drift_report.items():
            psi = drift_info["psi"]
            status = drift_info["status"]
            
            if status == "DRIFT":
                status_str = "🔴 Significant Distribution Shift (PSI > 0.20)"
            elif status == "WARN":
                status_str = "🟡 Moderate Shift (PSI 0.10 - 0.20)"
            else:
                status_str = "🟢 Stable (PSI < 0.10)"
                
            drift_rows.append({
                "Feature Name": col_name,
                "PSI Metric Value": f"{psi:.4f}",
                "Stability Status": status_str
            })
            
        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)


# ── LINEAGE ──
elif page == "Lineage":
    st.markdown("""
    <div class="glass-card animate-in">
        <div class="card-title">Pipeline Lineage</div>
        <div class="card-subtitle">Step-by-step execution history of the latest pipeline run. Full audit trail.</div>
    </div>
    """, unsafe_allow_html=True)

    cl, cr = st.columns([1.2, 1])

    with cl:
        if latest_lineage:
            st.markdown(render_lineage_timeline(latest_lineage), unsafe_allow_html=True)
        else:
            st.info("No lineage data available. Run `python main.py` to generate.")

    with cr:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">Run Summary</div>
            <div class="card-subtitle">Statistics from the latest pipeline execution.</div>
        </div>
        """, unsafe_allow_html=True)

        if latest_lineage:
            st.markdown(render_metric_card("Total Steps", str(len(latest_lineage)), "Pipeline stages executed", "#6366f1", "📊"), unsafe_allow_html=True)
            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            st.markdown(render_metric_card("Started At", latest_lineage[0].get("timestamp", "N/A"), "First step timestamp", "#818cf8", "🕐"), unsafe_allow_html=True)
            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            st.markdown(render_metric_card("Completed At", latest_lineage[-1].get("timestamp", "N/A"), "Final step timestamp", "#10b981", "✅"), unsafe_allow_html=True)

        total_runs = len(lineage_log) // max(len(PIPELINE_STEPS), 1)
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        st.markdown(render_metric_card("Total Runs", str(total_runs), "Historical pipeline executions", "#f59e0b", "🔄"), unsafe_allow_html=True)


# ── ARTIFACTS ──
elif page == "Artifacts":
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("""
        <div class="glass-card animate-in">
            <div class="card-title">Dataset Catalog</div>
            <div class="card-subtitle">Metadata snapshot from last pipeline run.</div>
        """, unsafe_allow_html=True)

        if metadata_report:
            st.markdown(render_key_value_list({
                "dataset_name": metadata_report.get("dataset_name"),
                "num_rows": metadata_report.get("num_rows"),
                "num_columns": metadata_report.get("num_columns"),
                "target_column": metadata_report.get("target_column"),
                "owner": metadata_report.get("owner"),
                "created_at": metadata_report.get("created_at"),
            }), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card animate-in">
            <div class="card-title">Download Artifacts</div>
            <div class="card-subtitle">Export reports and governance outputs.</div>
        """, unsafe_allow_html=True)

        if is_custom:
            import json
            files_data = {
                "📊 Model Metrics": json.dumps(model_metrics, indent=2),
                "📋 Quality Report": json.dumps(quality_report, indent=2),
                "⚖️ Fairness Report": json.dumps(fairness_report, indent=2),
                "💡 Explainability": json.dumps(explainability_report, indent=2),
                "🛡️ Compliance": json.dumps(compliance_report, indent=2),
                "📁 Metadata": json.dumps(metadata_report, indent=2),
                "📜 Lineage Log": json.dumps(lineage_log, indent=2),
            }
            dl_cols = st.columns(2)
            idx = 0
            for label, content in files_data.items():
                with dl_cols[idx % 2]:
                    st.download_button(
                        label=label,
                        data=content,
                        file_name=f"{label.split()[-1].lower()}_report.json",
                        key=f"dl_btn_custom_{idx}",
                        mime="application/json"
                    )
                idx += 1
                
            st.markdown('<div style="height: 12px; border-top: 1px solid rgba(255,255,255,0.06); margin-top: 15px; padding-top: 15px;"></div>', unsafe_allow_html=True)
            try:
                pdf_data = generate_pdf_report(pipeline, filename=metadata_report.get("dataset_name", "Custom Dataset"))
                st.download_button(
                    label="📄 Download PDF Compliance Certificate",
                    data=pdf_data,
                    file_name="anchor_ai_compliance_certificate.pdf",
                    key="dl_btn_custom_pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF Certificate: {e}")
        else:
            files = {
                "📊 Model Metrics": REPORTS_DIR / "model_metrics.json",
                "📋 Quality Report": REPORTS_DIR / "quality_report.json",
                "⚖️ Fairness Report": REPORTS_DIR / "fairness_report.json",
                "💡 Explainability": REPORTS_DIR / "explainability_report.json",
                "🛡️ Compliance": REPORTS_DIR / "compliance_results.json",
                "📁 Metadata": BASE_DIR / "metadata" / "dataset_catalog.json",
                "📜 Lineage Log": BASE_DIR / "metadata" / "lineage_log.json",
            }

            dl_cols = st.columns(2)
            idx = 0
            for label, path in files.items():
                if path.exists():
                    with dl_cols[idx % 2]:
                        with open(path, "rb") as f:
                            st.download_button(label=label, data=f, file_name=path.name, key=f"dl_btn_{idx}", mime="application/json")
                    idx += 1

            if shap_img.exists():
                st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
                with open(shap_img, "rb") as f:
                    st.download_button(label="🖼️ Download SHAP Plot", data=f, file_name="shap_summary.png", key="dl_btn_shap", mime="image/png")
        st.markdown("</div>", unsafe_allow_html=True)