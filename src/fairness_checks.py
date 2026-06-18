import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import save_json

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def fairness_check(X_test, y_test, model, original_df, target_col=None, demographic_col=None, task_type=None):
    # Auto-detect target_col if not specified
    if target_col is None:
        target_col = "Churn" if "Churn" in original_df.columns else original_df.columns[-1]

    # Auto-detect demographic_col if not specified
    if demographic_col is None:
        demographic_keywords = ["gender", "sex", "race", "age", "citizen", "income", "demographic", "origin", "nationality", "senior"]
        for col in original_df.columns:
            if col == target_col:
                continue
            if any(kw in col.lower() for kw in demographic_keywords):
                demographic_col = col
                break
        if demographic_col is None:
            for col in original_df.columns:
                if col == target_col:
                    continue
                if not pd.api.types.is_numeric_dtype(original_df[col]):
                    demographic_col = col
                    break
        if demographic_col is None:
            cols = [c for c in original_df.columns if c != target_col]
            demographic_col = cols[0] if cols else None

    # Auto-detect task_type
    if task_type is None:
        if y_test.nunique() > 10 or not pd.api.types.is_integer_dtype(y_test):
            task_type = "regression"
        else:
            task_type = "classification"

    if not demographic_col or demographic_col not in original_df.columns:
        results = {
            "demographic_gap": 0.0,
            "group_reports": {},
            "demographic_column": None,
            "disparate_impact_ratio": 1.0,
            "status": "NO_AUDIT"
        }
        save_json(results, REPORTS_DIR / "fairness_report.json")
        return results

    # Align demographic series
    demo_series = original_df.loc[X_test.index, demographic_col].copy()
    preds = model.predict(X_test)

    df_fair = pd.DataFrame({
        "demographic": demo_series,
        "actual": y_test,
        "prediction": preds
    })

    # Bin numeric demographic if continuous
    if pd.api.types.is_numeric_dtype(df_fair["demographic"]):
        bins = 3
        try:
            df_fair["demographic"] = pd.qcut(df_fair["demographic"], q=bins, labels=["Low Range", "Mid Range", "High Range"])
        except ValueError:
            df_fair["demographic"] = pd.cut(df_fair["demographic"], bins=bins, labels=["Low Range", "Mid Range", "High Range"])

    groups = df_fair["demographic"].unique()
    group_reports = {}
    disparate_impact_ratio = 1.0

    if task_type == "classification":
        unique_classes = df_fair["actual"].nunique()
        if unique_classes == 2:
            # Binary classification
            for g in groups:
                sub = df_fair[df_fair["demographic"] == g]
                g_count = len(sub)
                if g_count == 0:
                    continue
                act_pos = float((sub["actual"] == 1).mean())
                pred_pos = float((sub["prediction"] == 1).mean())

                tp = ((sub["actual"] == 1) & (sub["prediction"] == 1)).sum()
                fp = ((sub["actual"] == 0) & (sub["prediction"] == 1)).sum()
                tn = ((sub["actual"] == 0) & (sub["prediction"] == 0)).sum()
                fn = ((sub["actual"] == 1) & (sub["prediction"] == 0)).sum()

                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

                group_reports[str(g)] = {
                    "count": g_count,
                    "actual_positive_rate": act_pos,
                    "selection_rate": pred_pos,
                    "precision": prec,
                    "recall": rec,
                    "fpr": fpr,
                    "fnr": fnr
                }

            sel_rates = [v["selection_rate"] for v in group_reports.values()]
            rec_rates = [v["recall"] for v in group_reports.values()]
            fpr_rates = [v["fpr"] for v in group_reports.values()]
            fnr_rates = [v["fnr"] for v in group_reports.values()]

            demographic_gap = float(max(sel_rates) - min(sel_rates)) if sel_rates else 0.0
            equal_opportunity_gap = float(max(rec_rates) - min(rec_rates)) if rec_rates else 0.0
            fpr_gap = float(max(fpr_rates) - min(fpr_rates)) if fpr_rates else 0.0
            fnr_gap = float(max(fnr_rates) - min(fnr_rates)) if fnr_rates else 0.0

            # Disparate Impact Ratio: min(selection_rate) / max(selection_rate)
            if sel_rates and max(sel_rates) > 0:
                disparate_impact_ratio = float(min(sel_rates) / max(sel_rates))

            results = {
                "demographic_column": demographic_col,
                "group_reports": group_reports,
                "demographic_gap": demographic_gap,
                "equal_opportunity_gap": equal_opportunity_gap,
                "fpr_gap": fpr_gap,
                "fnr_gap": fnr_gap,
                "disparate_impact_ratio": disparate_impact_ratio,
                "task_subtype": "binary"
            }
        else:
            # Multiclass
            from sklearn.metrics import f1_score, accuracy_score
            for g in groups:
                sub = df_fair[df_fair["demographic"] == g]
                g_count = len(sub)
                if g_count == 0:
                    continue
                acc = float(accuracy_score(sub["actual"], sub["prediction"]))
                f1 = float(f1_score(sub["actual"], sub["prediction"], average="macro", zero_division=0))
                dist = sub["prediction"].value_counts(normalize=True).to_dict()

                group_reports[str(g)] = {
                    "count": g_count,
                    "accuracy": acc,
                    "macro_f1": f1,
                    "prediction_distribution": {str(k): float(v) for k, v in dist.items()}
                }

            acc_rates = [v["accuracy"] for v in group_reports.values()]
            demographic_gap = float(max(acc_rates) - min(acc_rates)) if acc_rates else 0.0

            results = {
                "demographic_column": demographic_col,
                "group_reports": group_reports,
                "demographic_gap": demographic_gap,
                "disparate_impact_ratio": 1.0,
                "task_subtype": "multiclass"
            }
    else:
        # Regression
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error
        for g in groups:
            sub = df_fair[df_fair["demographic"] == g]
            g_count = len(sub)
            if g_count == 0:
                continue
            mae = float(mean_absolute_error(sub["actual"], sub["prediction"]))
            rmse = float(root_mean_squared_error(sub["actual"], sub["prediction"]))
            mean_pred = float(sub["prediction"].mean())
            mean_err = float((sub["prediction"] - sub["actual"]).mean())

            group_reports[str(g)] = {
                "count": g_count,
                "mae": mae,
                "rmse": rmse,
                "mean_prediction": mean_pred,
                "mean_error": mean_err
            }

        mae_rates = [v["mae"] for v in group_reports.values()]
        mean_preds = [v["mean_prediction"] for v in group_reports.values()]

        demographic_gap = float(max(mae_rates) - min(mae_rates)) if mae_rates else 0.0
        pred_gap = float(max(mean_preds) - min(mean_preds)) if mean_preds else 0.0

        results = {
            "demographic_column": demographic_col,
            "group_reports": group_reports,
            "demographic_gap": demographic_gap,
            "prediction_gap": pred_gap,
            "disparate_impact_ratio": 1.0,
            "task_subtype": "regression"
        }

    # Backward compatibility mappings
    if "gender" in original_df.columns:
        df_fair["gender"] = original_df.loc[X_test.index, "gender"]
        gender_rates = df_fair.groupby("gender")["prediction"].mean().to_dict()
        results["gender_churn_rate"] = gender_rates
        results["gender_gap"] = float(abs(gender_rates.get("Female", 0.0) - gender_rates.get("Male", 0.0)))
    if "SeniorCitizen" in original_df.columns:
        df_fair["SeniorCitizen"] = original_df.loc[X_test.index, "SeniorCitizen"]
        senior_rates = df_fair.groupby("SeniorCitizen")["prediction"].mean().to_dict()
        results["senior_churn_rate"] = senior_rates
        results["senior_gap"] = float(abs(senior_rates.get(0, 0.0) - senior_rates.get(1, 0.0)))

    save_json(results, REPORTS_DIR / "fairness_report.json")
    return results