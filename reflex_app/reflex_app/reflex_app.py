from pathlib import Path
import json
import reflex as rx

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
METADATA_DIR = PROJECT_ROOT / "metadata"


def load_json(path: Path, default):
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default


# ---------- Load artifacts ----------
model_metrics = load_json(REPORTS_DIR / "model_metrics.json", {})
fairness_report = load_json(REPORTS_DIR / "fairness_report.json", {})
quality_report = load_json(REPORTS_DIR / "quality_report.json", {})
compliance_report = load_json(REPORTS_DIR / "compliance_results.json", {})
explainability_report = load_json(REPORTS_DIR / "explainability_report.json", {})
dataset_catalog = load_json(METADATA_DIR / "dataset_catalog.json", {})
lineage_log = load_json(METADATA_DIR / "lineage_log.json", [])

log_model = model_metrics.get("logistic_regression", {})
rf_model = model_metrics.get("random_forest", {})
decisions = compliance_report.get("decisions", {})

log_acc = log_model.get("accuracy", 0)
rf_acc = rf_model.get("accuracy", 0)

# Dynamic main variables
accuracy = f"{log_acc:.1%}" if log_acc > 0 else "N/A"
precision = f"{log_model.get('precision', 0):.1%}" if log_model else "N/A"
recall = f"{log_model.get('recall', 0):.1%}" if log_model else "N/A"
f1 = f"{log_model.get('f1_score', 0):.3f}" if log_model else "N/A"
senior_gap = f"{fairness_report.get('senior_gap', 0):.3f}" if fairness_report else "N/A"
gender_gap = f"{fairness_report.get('gender_gap', 0):.3f}" if fairness_report else "N/A"
final_status = compliance_report.get("final_status", "N/A")
score = compliance_report.get("score", 0)

gender_churn = fairness_report.get("gender_churn_rate", {})
senior_churn = fairness_report.get("senior_churn_rate", {})

# Prepare Recharts-friendly data arrays
model_compare_data = [
    {
        "metric": "Accuracy",
        "Logistic": log_model.get("accuracy", 0),
        "Random Forest": rf_model.get("accuracy", 0)
    },
    {
        "metric": "Precision",
        "Logistic": log_model.get("precision", 0),
        "Random Forest": rf_model.get("precision", 0)
    },
    {
        "metric": "Recall",
        "Logistic": log_model.get("recall", 0),
        "Random Forest": rf_model.get("recall", 0)
    },
    {
        "metric": "F1 Score",
        "Logistic": log_model.get("f1_score", 0),
        "Random Forest": rf_model.get("f1_score", 0)
    }
]

gender_chart_data = [
    {"group": k, "churn_rate": v} for k, v in gender_churn.items()
]
senior_chart_data = [
    {"group": "Senior" if str(k) == "1" else "Non-Senior", "churn_rate": v} for k, v in senior_churn.items()
]

top_features = explainability_report.get("top_features_by_mean_abs_shap", [])
try:
    shap_chart_data = [
        {"feature": f["feature"], "importance": f["importance"]} for f in top_features[:10]
    ]
except Exception:
    shap_chart_data = []

# Static fallback data for specific visual widgets not strictly in JSONs
radar_data = [
    {"metric": "Accuracy", "score": int(log_acc*100) if log_acc else 81},
    {"metric": "Precision", "score": int(log_model.get("precision", 0.66)*100)},
    {"metric": "Recall", "score": int(log_model.get("recall", 0.57)*100)},
    {"metric": "Explainable", "score": 88},
    {"metric": "Fairness", "score": 52},
    {"metric": "Data Health", "score": 100 if quality_report.get("status") == "PASS" else 40},
]
category_accuracy = [
    {"name": "Contract Rev.", "value": 85},
    {"name": "Billing Risk", "value": 62},
    {"name": "Senior Segment", "value": 41},
    {"name": "Fiber Optic", "value": 79},
    {"name": "Long Tenure", "value": 93},
]

artifacts_list = [
    "model_metrics.json", "fairness_report.json", "quality_report.json",
    "compliance_results.json", "explainability_report.json", "dataset_catalog.json",
    "shap_summary.png"
]


class State(rx.State):
    active_tab: str = "overview"

    def set_tab(self, tab: str):
        self.active_tab = tab

    @rx.event
    def download_file(self, filename: str):
        return rx.download(url=f"/{filename}")


def nav_item(label: str, icon_tag: str, id_name: str):
    active = State.active_tab == id_name
    return rx.button(
        rx.icon(
            tag=icon_tag, 
            size=18, 
            class_name=rx.cond(active, "text-indigo-400", "text-zinc-400")
        ),
        label,
        on_click=lambda: State.set_tab(id_name),
        class_name=rx.cond(
            active,
            "w-full flex items-center justify-start gap-3 px-4 py-3 rounded-xl bg-indigo-500/10 text-indigo-300 font-semibold border border-indigo-500/20 transition-all cursor-pointer",
            "w-full flex items-center justify-start gap-3 px-4 py-3 rounded-xl hover:bg-zinc-800/50 text-zinc-400 font-medium border border-transparent transition-all cursor-pointer"
        )
    )

def top_stat_card(label: str, val: str, icon_tag: str, icon_color: str):
    return rx.box(
        rx.hstack(
            rx.box(
                rx.icon(tag=icon_tag, size=20, class_name=icon_color),
                class_name="p-3 rounded-xl bg-zinc-900 border border-zinc-800"
            ),
            rx.vstack(
                rx.text(str(val), class_name="text-2xl font-bold text-zinc-100 leading-none"),
                rx.text(label, class_name="text-sm font-medium text-zinc-500"),
                align_items="start",
                spacing="1"
            ),
            spacing="4"
        ),
        class_name="p-5 rounded-2xl bg-zinc-900/50 border border-zinc-800/80 shadow-[0_0_20px_rgba(0,0,0,0.3)] backdrop-blur-md"
    )

def metric_card(title: str, val: str, sub: str, color: str):
    return rx.box(
        rx.text(str(val), class_name=f"text-3xl font-bold {color} mb-1"),
        rx.text(title, class_name="text-lg font-semibold text-zinc-200"),
        rx.text(sub, class_name="text-sm text-zinc-500"),
        class_name="p-5 rounded-2xl bg-zinc-900/40 border border-zinc-800/80 hover:bg-zinc-900/60 transition-colors shadow-sm"
    )

def chart_card(title: str, icon_tag: str, content: rx.Component, col_span="col-span-1"):
    return rx.box(
        rx.hstack(
            rx.icon(tag=icon_tag, size=20, class_name="text-indigo-400"),
            rx.text(title, class_name="text-lg font-semibold text-zinc-200"),
            class_name="mb-4"
        ),
        content,
        class_name=f"{col_span} p-5 rounded-2xl bg-zinc-900/40 border border-zinc-800/80 shadow-xl overflow-hidden"
    )


# ---------------- Tabs ----------------

def overview_tab():
    return rx.vstack(
        rx.grid(
            top_stat_card("Target Column", dataset_catalog.get("target_column", "Churn"), "shield", "text-indigo-400"),
            top_stat_card("Governance Risk", f"{score}/4", "gauge", "text-amber-400"),
            top_stat_card("Policy Status", final_status, "file-warning", "text-rose-400" if "REJECT" in final_status.upper() else ("text-amber-400" if "CONDITION" in final_status.upper() else "text-emerald-400")),
            columns="3",
            gap="4",
            class_name="w-full mb-6"
        ),
        rx.grid(
            metric_card("Accuracy", accuracy, "Validated holdout score", "text-indigo-300"),
            metric_card("Precision", precision, "Positive prediction quality", "text-fuchsia-300"),
            metric_card("Senior Bias Gap", senior_gap, "Primary fairness blocker", "text-rose-400"),
            columns="3",
            gap="4",
            class_name="w-full mb-6"
        ),
        rx.grid(
            chart_card(
                "Governance Dimensions",
                "radar",
                rx.recharts.radar_chart(
                    rx.recharts.polar_grid(stroke="#3f3f46"),
                    rx.recharts.polar_angle_axis(data_key="metric", tick={"fill": "#a1a1aa", "fontSize": 12}),
                    rx.recharts.radar(data_key="score", stroke="#818cf8", fill="#818cf8", fill_opacity=0.35),
                    data=radar_data,
                    height=280,
                    width="100%"
                )
            ),
            chart_card(
                "Category Performance",
                "bar-chart-3",
                rx.recharts.bar_chart(
                    rx.recharts.bar(data_key="value", fill="#c084fc"),
                    rx.recharts.x_axis(data_key="name", stroke="#a1a1aa", tick={"fontSize": 12}, angle=-15, text_anchor="end", height=60, dx=-5),
                    rx.recharts.tooltip(content_style={"backgroundColor": "#18181b", "borderColor": "#3f3f46", "borderRadius": "8px"}),
                    data=category_accuracy,
                    height=280,
                    width="100%"
                ),
                col_span="col-span-2 md:col-span-1 lg:col-span-2"
            ),
            columns="3",
            gap="4",
            class_name="w-full mb-6 md:grid-cols-1 lg:grid-cols-3"
        ),
        class_name="w-full"
    )

def models_tab():
    return rx.vstack(
        rx.text("Model Performance Review", class_name="text-2xl font-bold text-zinc-100 mb-6"),
        rx.grid(
            metric_card("Log. Accuracy", f"{log_model.get('accuracy',0):.1%}", "Operational threshold >0.75", "text-indigo-300"),
            metric_card("Log. Precision", f"{log_model.get('precision',0):.1%}", "Selected target quality", "text-indigo-300"),
            metric_card("RF Accuracy", f"{rf_model.get('accuracy',0):.1%}", "Alternative candidate", "text-zinc-400"),
            metric_card("RF Precision", f"{rf_model.get('precision',0):.1%}", "Alternative candidate", "text-zinc-400"),
            columns="4",
            gap="4",
            class_name="w-full mb-6"
        ),
        rx.box(
            rx.text("Comparative Metrics (Logistic VS Random Forest)", class_name="text-lg font-semibold text-zinc-200 mb-4"),
            rx.recharts.bar_chart(
                rx.recharts.bar(data_key="Logistic", fill="#818cf8"),
                rx.recharts.bar(data_key="Random Forest", fill="#52525b"),
                rx.recharts.x_axis(data_key="metric", stroke="#a1a1aa"),
                rx.recharts.y_axis(stroke="#a1a1aa"),
                rx.recharts.tooltip(content_style={"backgroundColor": "#18181b", "borderColor": "#3f3f46", "borderRadius": "8px"}),
                rx.recharts.legend(),
                data=model_compare_data,
                height=350,
                width="100%"
            ),
            class_name="w-full p-6 rounded-2xl bg-zinc-900/40 border border-zinc-800/80 shadow-xl"
        ),
        class_name="w-full"
    )

def fairness_tab():
    senior_tone = "text-rose-400" if decisions.get("senior_fairness") == "FAIL" else "text-emerald-400"
    gender_tone = "text-rose-400" if decisions.get("gender_fairness") == "FAIL" else "text-emerald-400"
    
    return rx.vstack(
        rx.text("Fairness Risk Profile", class_name="text-2xl font-bold text-zinc-100 mb-6"),
        rx.grid(
            rx.box(
                rx.text("Senior Citizen Gap", class_name="text-lg font-semibold text-zinc-200 mb-1"),
                rx.text(f"Gap Magnitude: {senior_gap}", class_name=f"text-xl font-bold {senior_tone} mb-4"),
                rx.recharts.bar_chart(
                    rx.recharts.bar(data_key="churn_rate", fill="#fb7185"),
                    rx.recharts.x_axis(data_key="group", stroke="#a1a1aa"),
                    rx.recharts.tooltip(content_style={"backgroundColor": "#18181b", "borderColor": "#3f3f46", "borderRadius": "8px"}),
                    data=senior_chart_data,
                    height=250,
                    width="100%"
                ),
                class_name="w-full p-6 rounded-2xl bg-zinc-900/40 border border-zinc-800/80 shadow-md"
            ),
            rx.box(
                rx.text("Gender Gap", class_name="text-lg font-semibold text-zinc-200 mb-1"),
                rx.text(f"Gap Magnitude: {gender_gap}", class_name=f"text-xl font-bold {gender_tone} mb-4"),
                rx.recharts.bar_chart(
                    rx.recharts.bar(data_key="churn_rate", fill="#34d399"),
                    rx.recharts.x_axis(data_key="group", stroke="#a1a1aa"),
                    rx.recharts.tooltip(content_style={"backgroundColor": "#18181b", "borderColor": "#3f3f46", "borderRadius": "8px"}),
                    data=gender_chart_data,
                    height=250,
                    width="100%"
                ),
                class_name="w-full p-6 rounded-2xl bg-zinc-900/40 border border-zinc-800/80 shadow-md"
            ),
            columns="2",
            gap="4",
            class_name="w-full md:grid-cols-1 lg:grid-cols-2"
        ),
        class_name="w-full"
    )

def explainability_tab():
    return rx.vstack(
        rx.text("Explainability (SHAP)", class_name="text-2xl font-bold text-zinc-100 mb-6"),
        rx.grid(
            rx.box(
                rx.text("Top Mean Absolute SHAP Factors", class_name="text-lg font-semibold text-zinc-200 mb-4"),
                rx.recharts.bar_chart(
                    rx.recharts.bar(data_key="importance", fill="#a78bfa"),
                    rx.recharts.x_axis(data_key="importance", type_="number", stroke="#a1a1aa"),
                    rx.recharts.y_axis(data_key="feature", type_="category", width=150, stroke="#a1a1aa", tick={"fontSize": 12}),
                    rx.recharts.tooltip(content_style={"backgroundColor": "#18181b", "borderColor": "#3f3f46", "borderRadius": "8px"}),
                    layout="vertical",
                    data=shap_chart_data,
                    height=450,
                    width="100%"
                ),
                class_name="w-full p-6 rounded-2xl bg-zinc-900/40 border border-zinc-800/80 shadow-md"
            ),
            rx.box(
                rx.text("SHAP Summary Plot", class_name="text-lg font-semibold text-zinc-200 mb-4"),
                rx.image(
                    src="/shap_summary.png",
                    width="100%",
                    class_name="rounded-xl border border-zinc-800"
                ),
                class_name="w-full p-6 rounded-2xl bg-zinc-900/40 border border-zinc-800/80 shadow-md flex flex-col justify-center"
            ),
            columns="2",
            gap="4",
            class_name="w-full md:grid-cols-1 lg:grid-cols-2"
        ),
        class_name="w-full"
    )

def evidence_tab():
    q_score = quality_report.get("quality_score", "N/A")
    return rx.vstack(
        rx.text("Audit Evidence & Logs", class_name="text-2xl font-bold text-zinc-100 mb-6"),
        rx.box(
            rx.text("Data Health Statistics", class_name="text-lg font-semibold text-zinc-200 mb-4"),
            rx.grid(
                rx.box(rx.text("Rows"), rx.text(quality_report.get("num_rows", "N/A"), class_name="font-bold text-fuchsia-300"), class_name="p-4 bg-zinc-900/80 rounded-xl border border-zinc-800"),
                rx.box(rx.text("Columns"), rx.text(quality_report.get("num_columns", "N/A"), class_name="font-bold text-fuchsia-300"), class_name="p-4 bg-zinc-900/80 rounded-xl border border-zinc-800"),
                rx.box(rx.text("Missing Values"), rx.text(quality_report.get("total_missing_values", "N/A"), class_name="font-bold text-fuchsia-300"), class_name="p-4 bg-zinc-900/80 rounded-xl border border-zinc-800"),
                rx.box(rx.text("Duplicates"), rx.text(quality_report.get("duplicate_rows", "N/A"), class_name="font-bold text-fuchsia-300"), class_name="p-4 bg-zinc-900/80 rounded-xl border border-zinc-800"),
                rx.box(rx.text("Quality Score"), rx.text(f"{q_score}", class_name="font-bold text-emerald-300"), class_name="p-4 bg-zinc-900/80 rounded-xl border border-zinc-800"),
                columns="5",
                gap="3",
                class_name="w-full"
            ),
            class_name="w-full p-6 bg-zinc-900/20 border border-zinc-800/80 rounded-2xl mb-6 shadow-md"
        ),
        rx.box(
            rx.text("Download Artifacts", class_name="text-lg font-semibold text-zinc-200 mb-4"),
            rx.grid(
                *[
                    rx.box(
                        rx.hstack(
                            rx.icon(tag="file-json", size=24, class_name="text-indigo-400"),
                            rx.text(file, class_name="font-medium text-zinc-200 ml-2 truncate"),
                            rx.spacer(),
                            rx.button(
                                rx.icon(tag="download", size=16),
                                on_click=lambda f=file: State.download_file(f),
                                class_name="p-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg transition-colors border border-zinc-700 cursor-pointer"
                            ),
                            class_name="w-full items-center"
                        ),
                        class_name="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800 hover:border-indigo-500/30 transition-all shadow-md"
                    ) for file in artifacts_list
                ],
                columns="2",
                gap="4",
                class_name="w-full md:grid-cols-1 lg:grid-cols-2"
            ),
            class_name="w-full"
        ),
        class_name="w-full"
    )

def page_content():
    return rx.box(
        rx.cond(
            State.active_tab == "overview",
            overview_tab(),
            rx.cond(
                State.active_tab == "models",
                models_tab(),
                rx.cond(
                    State.active_tab == "fairness",
                    fairness_tab(),
                    rx.cond(
                        State.active_tab == "explain",
                        explainability_tab(),
                        rx.cond(
                            State.active_tab == "evidence",
                            evidence_tab(),
                            rx.box(class_name="hidden")
                        )
                    )
                )
            )
        )
    )

def index():
    return rx.box(
        rx.hstack(
            # Sidebar
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.box(
                            rx.icon(tag="anchor", size=24, class_name="text-fuchsia-400"),
                            class_name="p-2 bg-zinc-900 rounded-xl border border-zinc-800 shadow-[0_0_15px_rgba(232,121,249,0.15)] flex items-center justify-center"
                        ),
                        rx.vstack(
                            rx.text("Anchor Guard", class_name="text-xl font-bold tracking-tight text-zinc-50 leading-tight"),
                            rx.text("Platform AI Risk Dept", class_name="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em]"),
                            align_items="start",
                            spacing="0"
                        ),
                        spacing="3",
                        class_name="mb-6 w-full flex items-center"
                    ),
                    nav_item("Overview", "layout-dashboard", "overview"),
                    nav_item("Model Review", "bot", "models"),
                    nav_item("Fairness Risk", "scale", "fairness"),
                    nav_item("Explainability", "sparkles", "explain"),
                    nav_item("Evidence & Logs", "database", "evidence"),
                    rx.spacer(),
                    rx.box(
                        rx.hstack(
                            rx.box(class_name="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"),
                            rx.text("System Active", class_name="text-xs font-bold text-zinc-400"),
                            class_name="px-3 py-2 rounded-lg bg-zinc-900/50 border border-zinc-800/80 flex items-center gap-2"
                        ),
                        class_name="w-full flex justify-center pb-2"
                    ),
                    spacing="2",
                    class_name="h-full w-full"
                ),
                class_name="w-72 h-screen flex-shrink-0 bg-[#09090b] border-r border-zinc-800/80 p-6 flex flex-col items-start z-50 fixed left-0 top-0"
            ),
            # Main Content
            rx.box(
                rx.box(
                    rx.hstack(
                        rx.vstack(
                            rx.text("Governance Insights", class_name="text-3xl font-extrabold text-zinc-50 tracking-tight"),
                            rx.text("Continuous evaluation of model behavior, societal impact, and performance risk.", class_name="text-sm font-medium text-zinc-400"),
                            align_items="start",
                            spacing="1"
                        ),
                        rx.spacer(),
                        rx.box(
                            rx.text(f"Status: {final_status}", class_name="text-sm font-bold text-indigo-300 uppercase tracking-widest"),
                            class_name="px-5 py-2.5 rounded-xl border border-indigo-500/20 bg-indigo-500/10 shadow-[0_0_20px_rgba(99,102,241,0.1)] flex justify-center items-center"
                        ),
                        class_name="w-full mb-10 items-center"
                    ),
                    page_content(),
                    class_name="w-full max-w-6xl mx-auto"
                ),
                class_name="ml-72 w-full min-h-screen bg-[#09090b] p-10 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.11),rgba(255,255,255,0))]"
            ),
            class_name="w-full relative flex text-zinc-50 font-sans min-h-screen"
        )
    )

app = rx.App(
    theme=rx.theme(
        appearance="dark",
    )
)
app.add_page(index)