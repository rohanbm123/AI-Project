from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from src.utils import save_json


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def explain_model_with_shap(model, X_train, X_test):
    background = X_train.sample(min(100, len(X_train)), random_state=42)

    explainer = shap.Explainer(model, background)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(
        abs(shap_values.values).mean(axis=0),
        index=X_test.columns
    ).sort_values(ascending=False)

    explanation = {
        "top_features_by_mean_abs_shap": [
            {"feature": feature, "importance": float(value)}
            for feature, value in mean_abs_shap.head(10).items()
        ]
    }

    save_json(explanation, REPORTS_DIR / "explainability_report.json")

    return explanation