import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import save_json

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def check_data_quality(df):
    report = {}

    # Missing values
    missing = df.isnull().sum().sum()
    report['total_missing_values'] = int(missing)

    # Duplicate rows
    duplicates = df.duplicated().sum()
    report['duplicate_rows'] = int(duplicates)

    # Row & column count
    report['num_rows'] = df.shape[0]
    report['num_columns'] = df.shape[1]

    # Simple quality scoring
    score = 100
    if missing > 0:
        score -= 20
    if duplicates > 0:
        score -= 10

    report['quality_score'] = score
    report['status'] = "PASS" if score >= 80 else "FAIL"

    # Save quality report
    save_json(report, REPORTS_DIR / "quality_report.json")
    return report

def calculate_psi(baseline, comparison, bins=10):
    """
    Calculates Population Stability Index (PSI) between two numeric or categorical series.
    """
    baseline = pd.Series(baseline).dropna()
    comparison = pd.Series(comparison).dropna()

    if len(baseline) == 0 or len(comparison) == 0:
        return 0.0

    # Determine if both are numeric
    if pd.api.types.is_numeric_dtype(baseline) and pd.api.types.is_numeric_dtype(comparison):
        try:
            # Bin edges from baseline deciles
            _, bin_edges = pd.qcut(baseline, q=bins, retbins=True, duplicates='drop')
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            base_counts = pd.cut(baseline, bins=bin_edges).value_counts(normalize=True)
            comp_counts = pd.cut(comparison, bins=bin_edges).value_counts(normalize=True)
        except Exception:
            # Fallback to value counts
            base_counts = baseline.value_counts(normalize=True)
            comp_counts = comparison.value_counts(normalize=True)
    else:
        # Categorical
        base_counts = baseline.value_counts(normalize=True)
        comp_counts = comparison.value_counts(normalize=True)

    # Align classes/bins
    all_classes = set(base_counts.index).union(set(comp_counts.index))

    psi_val = 0.0
    epsilon = 1e-4

    for c in all_classes:
        actual_pct = comp_counts.get(c, 0.0)
        expected_pct = base_counts.get(c, 0.0)

        # Apply epsilon for stability
        actual_pct = max(actual_pct, epsilon)
        expected_pct = max(expected_pct, epsilon)

        psi_val += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)

    return float(psi_val)

def check_data_drift(baseline_df, comparison_df, feature_cols):
    """
    Calculates feature-level data drift using PSI.
    """
    drift_report = {}
    for col in feature_cols:
        if col in baseline_df.columns and col in comparison_df.columns:
            psi = calculate_psi(baseline_df[col], comparison_df[col])
            status = "NO_DRIFT" if psi < 0.1 else ("WARN" if psi < 0.2 else "DRIFT")
            drift_report[col] = {
                "psi": psi,
                "status": status
            }
    
    # Save drift report
    save_json(drift_report, REPORTS_DIR / "drift_report.json")
    return drift_report