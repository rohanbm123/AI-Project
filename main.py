import argparse
import pandas as pd
from pathlib import Path

from src.auto_pipeline import AutoMLPipeline
from src.pdf_generator import generate_pdf_report

BASE_DIR = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Anchor AI AutoML and Governance Pipeline CLI")
    parser.add_argument("--file", type=str, default="data/raw/Customer-Churn.csv", help="Filename or absolute path to CSV dataset")
    parser.add_argument("--target", type=str, default=None, help="Target column name")
    parser.add_argument("--demographic", type=str, default=None, help="Demographic column name")
    parser.add_argument("--task_type", type=str, default=None, choices=["classification", "regression", "auto"], help="Override task type")
    parser.add_argument("--positive_class", type=str, default=None, help="Positive class label (for classification)")
    parser.add_argument("--eu_risk_tier", type=str, default="Low", choices=["High", "Medium", "Low"], help="EU AI Act risk tier mapping")
    parser.add_argument("--data_only", action="store_true", help="Run only data quality check profiling")
    args = parser.parse_args()

    # Load raw data
    file_path = Path(args.file)
    if not file_path.exists():
        file_path = BASE_DIR / args.file
    if not file_path.exists():
        print(f"Error: Dataset file not found at '{args.file}'")
        return

    print(f"Loading dataset: {file_path.name}...")
    df = pd.read_csv(file_path)

    # Initialize AutoMLPipeline
    pipeline = AutoMLPipeline(
        target_col=args.target,
        demographic_col=args.demographic,
        task_type_override=args.task_type if args.task_type != "auto" else None,
        positive_class=args.positive_class
    )

    print("Executing Pipeline...")
    pipeline.run_pipeline(df, data_only=args.data_only, eu_risk_tier=args.eu_risk_tier, filename=file_path.name)

    if args.data_only:
        print("Data Quality Profiling completed successfully.")
        print(f"Total Rows: {pipeline.data_quality['total_rows']}")
        print(f"Total Columns: {pipeline.data_quality['total_columns']}")
        print(f"Missing Values: {pipeline.data_quality['total_missing_values']}")
        print(f"Quality Score: {pipeline.data_quality['quality_score']}")
    else:
        print("\n=== AutoML Pipeline Training & Validation Success ===")
        print(f"Detected Task: {pipeline.task_type.upper()}")
        print(f"Target Column: {pipeline.target_col}")
        print(f"Protected Demographic Attribute: {pipeline.demographic_col}")
        
        print("\n=== Model Metrics ===")
        if pipeline.task_type == "classification":
            lin_m = pipeline.metrics.get("logistic_regression", {})
            ens_m = pipeline.metrics.get("random_forest", {})
            print("Linear Model (Logistic Regression):")
            print(f"  Accuracy:  {lin_m.get('accuracy', 0.0) * 100:.1f}%")
            print(f"  Recall:    {lin_m.get('recall', 0.0) * 100:.1f}%")
            print(f"  F1 Score:  {lin_m.get('f1_score', 0.0) * 100:.1f}%")
            print("Ensemble Model (HistGradientBoosting):")
            print(f"  Accuracy:  {ens_m.get('accuracy', 0.0) * 100:.1f}%")
            print(f"  Recall:    {ens_m.get('recall', 0.0) * 100:.1f}%")
            print(f"  F1 Score:  {ens_m.get('f1_score', 0.0) * 100:.1f}%")
        else:
            lin_m = pipeline.metrics.get("linear_regression", {})
            ens_m = pipeline.metrics.get("random_forest", {})
            print("Linear Model (RidgeCV):")
            print(f"  R2 Score:  {lin_m.get('r2_score', 0.0):.4f}")
            print(f"  MAE:       {lin_m.get('mae', 0.0):.4f}")
            print("Ensemble Model (HistGradientBoosting):")
            print(f"  R2 Score:  {ens_m.get('r2_score', 0.0):.4f}")
            print(f"  MAE:       {ens_m.get('mae', 0.0):.4f}")

        # Fairness
        if pipeline.demographic_col:
            print("\n=== Algorithmic Fairness Audit ===")
            print(f"Audited demographic: {pipeline.demographic_col}")
            print(f"Selection Rate Gap / MAE Gap: {pipeline.fairness_report.get('demographic_gap', 0.0):.4f}")
            if pipeline.task_type == "classification":
                print(f"Disparate Impact Ratio:       {pipeline.fairness_report.get('disparate_impact_ratio', 1.0):.4f}")

        # Governance Status
        compliance = pipeline.compliance_report
        print("\n=== Policy Engine Results ===")
        print(f"EU Risk Tier: {compliance.get('eu_risk_tier')}")
        print(f"Compliance Score: {compliance.get('score')} / {compliance.get('max_score')} rules passed")
        print(f"Final Governance Verdict: {compliance.get('final_status')}")

        # Save Report
        pdf_path = BASE_DIR / "reports" / "compliance_report.pdf"
        print(f"\nCompiling compliance PDF Report...")
        pdf_data = generate_pdf_report(pipeline, filename=file_path.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_data)
        print(f"Compliance PDF Report saved to: {pdf_path}")

if __name__ == "__main__":
    main()