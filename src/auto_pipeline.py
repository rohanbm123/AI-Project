import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, root_mean_squared_error
)
import shap

from src.data_quality import check_data_quality, check_data_drift
from src.fairness_checks import fairness_check
from src.policy_engine import policy_check

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

class AutoMLPipeline:
    def __init__(self, target_col=None, demographic_col=None, task_type_override=None, positive_class=None, id_columns_override=None):
        self.target_col = target_col
        self.demographic_col = demographic_col
        self.task_type_override = task_type_override
        self.positive_class = positive_class
        self.id_columns_override = id_columns_override if id_columns_override is not None else []
        self.task_type = None  # 'classification' or 'regression'

        # Preprocessing columns
        self.numeric_cols = []
        self.categorical_cols = []
        self.ignored_cols = []
        self.feature_columns = []
        
        # Scikit-learn Pipeline objects
        self.linear_pipeline = None
        self.ensemble_pipeline = None

        # Result dictionaries
        self.metrics = {}
        self.fairness_report = {}
        self.explainability_report = {}
        self.compliance_report = {}
        self.data_quality = {}
        self.drift_report = {}
        self.metadata = {}
        self.features_schema = {}

    @property
    def linear_model(self):
        if self.linear_pipeline is not None:
            return self.linear_pipeline.named_steps['model']
        return None

    @property
    def forest_model(self):
        if self.ensemble_pipeline is not None:
            return self.ensemble_pipeline.named_steps['model']
        return None

    def detect_task_type(self, df):
        target = self.target_col if self.target_col else df.columns[-1]
        self.target_col = target

        if self.task_type_override and self.task_type_override.lower() in ["classification", "regression"]:
            self.task_type = self.task_type_override.lower()
            return self.task_type

        col_dtype = df[target].dtype
        unique_count = df[target].nunique()

        if unique_count == 2:
            self.task_type = "classification"
        elif pd.api.types.is_numeric_dtype(col_dtype) and unique_count > 10:
            self.task_type = "regression"
        else:
            self.task_type = "classification"

        return self.task_type

    def auto_detect_demographic(self, df):
        if self.demographic_col and self.demographic_col in df.columns:
            return self.demographic_col

        demographic_keywords = ["gender", "sex", "race", "age", "citizen", "income", "demographic", "origin", "nationality", "senior"]
        for col in df.columns:
            if col == self.target_col:
                continue
            lower_col = col.lower()
            if any(kw in lower_col for kw in demographic_keywords):
                self.demographic_col = col
                return col

        # Fallback to the first categorical column
        for col in df.columns:
            if col == self.target_col:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.demographic_col = col
                return col

        cols = [c for c in df.columns if c != self.target_col]
        self.demographic_col = cols[0] if cols else None
        return self.demographic_col

    def inspect_data_quality(self, df):
        # Missing and duplicate checks
        missing_counts = df.isnull().sum().to_dict()
        total_missing = int(df.isnull().sum().sum())
        duplicate_count = int(df.duplicated().sum())

        # Exclude constant and ID columns
        cols_to_drop = list(self.id_columns_override)
        id_keywords = ["id", "key", "uuid", "index", "code", "identifier", "record_id", "customer_id", "user_id"]
        for col in df.columns:
            if col == self.target_col or col == self.demographic_col or col in cols_to_drop:
                continue
            lower_col = col.lower()
            if lower_col in id_keywords or any(lower_col.endswith(kw) for kw in id_keywords) or any(lower_col.startswith(kw) for kw in id_keywords):
                if df[col].nunique() == len(df):
                    cols_to_drop.append(col)

        self.ignored_cols = list(set(cols_to_drop))
        
        self.data_quality = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values_by_column": missing_counts,
            "total_missing_values": total_missing,
            "duplicate_rows": duplicate_count,
            "dropped_columns": self.ignored_cols
        }
        return self.data_quality

    def _calculate_sample_weights(self, df_train, y_train):
        """
        Kamiran & Calders re-weighting algorithm for fairness mitigation.
        """
        N = len(df_train)
        weights = np.ones(N)
        if self.demographic_col and self.demographic_col in df_train.columns:
            g_vals = df_train[self.demographic_col].astype(str).values
            y_vals = y_train.values
            unique_g = np.unique(g_vals)
            unique_y = np.unique(y_vals)
            
            for g in unique_g:
                for y in unique_y:
                    n_g = np.sum(g_vals == g)
                    n_y = np.sum(y_vals == y)
                    n_gy = np.sum((g_vals == g) & (y_vals == y))
                    if n_gy > 0:
                        val = (n_y * n_g) / (N * n_gy)
                        weights[(g_vals == g) & (y_vals == y)] = val
        return weights

    def run_pipeline(self, df, custom_limits=None, data_only=False, eu_risk_tier="Low", filename="dataset.csv"):
        # Clean blanks & whitespaces to NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)
        self.original_df = df.copy()

        # Task and demographics setup
        self.detect_task_type(df)
        self.auto_detect_demographic(df)
        self.inspect_data_quality(df)

        self.metadata["data_only"] = data_only
        self.metadata["filename"] = filename
        if data_only:
            check_data_quality(df)
            return

        # Target formatting
        y = df[self.target_col].copy()
        if self.task_type == "classification":
            unique_vals = sorted(list(y.dropna().astype(str).unique()))
            if len(unique_vals) == 2:
                pos = str(self.positive_class) if self.positive_class is not None else unique_vals[-1]
                if pos not in unique_vals:
                    pos = unique_vals[-1]
                neg = [v for v in unique_vals if v != pos][0]
                mapping = {neg: 0, pos: 1}
                y = y.astype(str).map(mapping)
                self.metadata["target_mapping"] = {str(k): int(v) for k, v in mapping.items()}
                self.metadata["positive_class"] = pos
            else:
                y = pd.Categorical(y).codes
            y = pd.Series(y).fillna(0).astype(int)
        else:
            y = pd.to_numeric(y, errors='coerce')
            y = y.fillna(y.median())

        # Split features and target
        X = df.drop(columns=[self.target_col] + self.ignored_cols, errors='ignore')

        # Identify numerical and categorical columns
        self.numeric_cols = []
        self.categorical_cols = []
        for col in X.columns:
            if col == self.demographic_col:
                # Keep demographic column as categorical for simple binning/encoding compatibility
                self.categorical_cols.append(col)
                continue
            if pd.api.types.is_numeric_dtype(X[col]):
                # Constant check
                if X[col].nunique() <= 1:
                    self.ignored_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            else:
                if X[col].nunique() <= 1:
                    self.ignored_cols.append(col)
                else:
                    self.categorical_cols.append(col)

        # Refilter X features list
        X = X[self.numeric_cols + self.categorical_cols]

        # Extract features schema for HIML verification
        self.features_schema = {}
        for col in self.numeric_cols:
            self.features_schema[col] = {
                "type": "numeric",
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "mean": float(X[col].mean()),
                "missing_pct": float(X[col].isnull().mean())
            }
        for col in self.categorical_cols:
            self.features_schema[col] = {
                "type": "categorical",
                "categories": [str(c) for c in X[col].dropna().unique()],
                "mode": str(X[col].mode().iloc[0]) if not X[col].mode().empty else "Missing",
                "missing_pct": float(X[col].isnull().mean())
            }

        # Train/Test Split
        if self.task_type == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Calculate Data Drift (PSI)
        self.drift_report = check_data_drift(X_train, X_test, self.numeric_cols + self.categorical_cols)

        # Build Preprocessing Pipeline
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, self.numeric_cols),
            ('cat', cat_transformer, self.categorical_cols)
        ])

        # Define Estimators
        if self.task_type == "classification":
            linear_est = LogisticRegression(max_iter=5000, solver="liblinear", random_state=42)
            ensemble_est = HistGradientBoostingClassifier(random_state=42)
        else:
            linear_est = RidgeCV()
            ensemble_est = HistGradientBoostingRegressor(random_state=42)

        # Build final Pipelines
        self.linear_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', linear_est)
        ])

        self.ensemble_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', ensemble_est)
        ])

        # Step A: Fit baseline pipelines
        self.linear_pipeline.fit(X_train, y_train)
        self.ensemble_pipeline.fit(X_train, y_train)

        # Step B: Calculate initial fairness for mitigation checks
        initial_fairness = fairness_check(
            X_test, y_test, self.linear_pipeline, self.original_df,
            target_col=self.target_col, demographic_col=self.demographic_col, task_type=self.task_type
        )
        
        # Step C: Apply mitigation if disparate impact ratio breaks the four-fifths rule
        di_ratio = initial_fairness.get("disparate_impact_ratio", 1.0)
        self.metadata["mitigation_applied"] = False
        
        if self.task_type == "classification" and self.demographic_col and di_ratio < 0.80:
            # Re-weight training samples
            sample_weights = self._calculate_sample_weights(X_train, y_train)
            self.metadata["mitigation_applied"] = True
            
            # Retrain with weights
            self.linear_pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
            self.ensemble_pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)

        # Create processed DataFrames for compatibility with metrics & explanations
        X_train_transformed = self.linear_pipeline.named_steps['preprocessor'].transform(X_train)
        self.feature_columns = list(self.linear_pipeline.named_steps['preprocessor'].get_feature_names_out())
        
        self.X_train = pd.DataFrame(X_train_transformed, columns=self.feature_columns, index=X_train.index)
        self.y_train = y_train

        X_test_transformed = self.linear_pipeline.named_steps['preprocessor'].transform(X_test)
        self.X_test = pd.DataFrame(X_test_transformed, columns=self.feature_columns, index=X_test.index)
        self.y_test = y_test

        # Compute Metrics
        self.metrics = {}
        if self.task_type == "classification":
            y_pred_lin = self.linear_pipeline.predict(X_test)
            self.metrics["logistic_regression"] = {
                "accuracy": float(accuracy_score(y_test, y_pred_lin)),
                "precision": float(precision_score(y_test, y_pred_lin, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred_lin, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred_lin, zero_division=0))
            }
            
            y_pred_ens = self.ensemble_pipeline.predict(X_test)
            self.metrics["random_forest"] = {
                "accuracy": float(accuracy_score(y_test, y_pred_ens)),
                "precision": float(precision_score(y_test, y_pred_ens, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred_ens, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred_ens, zero_division=0))
            }
        else:
            y_pred_lin = self.linear_pipeline.predict(X_test)
            self.metrics["linear_regression"] = {
                "r2_score": float(r2_score(y_test, y_pred_lin)),
                "mae": float(mean_absolute_error(y_test, y_pred_lin)),
                "rmse": float(root_mean_squared_error(y_test, y_pred_lin))
            }
            
            y_pred_ens = self.ensemble_pipeline.predict(X_test)
            self.metrics["random_forest"] = {
                "r2_score": float(r2_score(y_test, y_pred_ens)),
                "mae": float(mean_absolute_error(y_test, y_pred_ens)),
                "rmse": float(root_mean_squared_error(y_test, y_pred_ens))
            }

        # Calculate final audited reports
        self.fairness_report = fairness_check(
            X_test, y_test, self.linear_pipeline, self.original_df,
            target_col=self.target_col, demographic_col=self.demographic_col, task_type=self.task_type
        )
        self.fairness_report["target_col"] = self.target_col

        self.compute_explainability()
        self.compliance_report = policy_check(
            self.metrics, self.fairness_report,
            eu_risk_tier=eu_risk_tier, drift_results=self.drift_report
        )

        # Serialize Pipelines into a single binary
        joblib.dump({
            "task_type": self.task_type,
            "target_col": self.target_col,
            "demographic_col": self.demographic_col,
            "features_schema": self.features_schema,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "feature_columns": self.feature_columns,
            "linear": self.linear_pipeline,
            "ensemble": self.ensemble_pipeline,
            "metrics": self.metrics,
            "fairness_report": self.fairness_report,
            "drift_report": self.drift_report,
            "explainability_report": self.explainability_report,
            "compliance_report": self.compliance_report,
            "data_quality": self.data_quality,
            "metadata": self.metadata,
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test
        }, MODELS_DIR / "serialized_pipeline.joblib")

    def compute_explainability(self):
        # Local model coefficient mapping
        coefs = self.linear_model.coef_
        if self.task_type == "classification":
            coefs = coefs[0]

        feature_importances = []
        for feat, coef in zip(self.feature_columns, coefs):
            feature_importances.append({
                "feature": feat,
                "importance": float(abs(coef)),
                "coefficient": float(coef)
            })
        
        feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)

        # Calculate SHAP values on preprocessed splits
        try:
            background = self.X_train.sample(min(50, len(self.X_train)), random_state=42)
            explainer = shap.Explainer(self.linear_model, background)
            shap_values = explainer(self.X_test)
            mean_abs_shap = pd.Series(
                np.abs(shap_values.values).mean(axis=0),
                index=self.X_test.columns
            ).sort_values(ascending=False).to_dict()

            shap_importances = [
                {"feature": k, "importance": float(v)}
                for k, v in mean_abs_shap.items()
            ]
        except Exception:
            # Fallback
            shap_importances = [
                {"feature": x["feature"], "importance": x["importance"]}
                for x in feature_importances
            ]

        self.explainability_report = {
            "global_importance": shap_importances[:10],
            "coefficients": feature_importances
        }

    def predict_single(self, inputs, model_name="Logistic Regression"):
        # Make a single-row DataFrame out of user inputs
        row = {}
        for col in self.numeric_cols:
            row[col] = inputs.get(col, self.features_schema[col]["mean"])
        for col in self.categorical_cols:
            row[col] = inputs.get(col, self.features_schema[col]["mode"])

        df_single = pd.DataFrame([row])

        # Pick Pipeline
        pipeline = self.linear_pipeline if "logistic" in model_name.lower() or "linear" in model_name.lower() else self.ensemble_pipeline

        if self.task_type == "classification":
            proba = float(pipeline.predict_proba(df_single)[0][1])
            predicted_class_idx = int(pipeline.predict(df_single)[0])

            # Class mapping
            target_map = self.metadata.get("target_mapping", {})
            class_name = str(predicted_class_idx)
            if target_map:
                inv_map = {v: k for k, v in target_map.items()}
                class_name = inv_map.get(predicted_class_idx, class_name)

            # Local reasonings
            factors = []
            if "logistic" in model_name.lower() or "linear" in model_name.lower():
                preprocessor = pipeline.named_steps['preprocessor']
                row_preprocessed = preprocessor.transform(df_single)[0]
                coefs = pipeline.named_steps['model'].coef_[0]
                contrib_map = dict(zip(self.feature_columns, coefs * row_preprocessed))

                # Numerical features
                for col in self.numeric_cols:
                    name = f"num__{col}"
                    if name in contrib_map:
                        contrib = contrib_map[name]
                        val = inputs.get(col, 0.0)
                        factors.append({
                            "name": col,
                            "label": f"{col.replace('_', ' ').title()} ({val})",
                            "contribution": float(contrib)
                        })

                # Categorical features (grouping dummy back to actual value)
                for col in self.categorical_cols:
                    val = inputs.get(col, "Missing")
                    dummy_name = f"cat__{col}_{val}"
                    if dummy_name in contrib_map:
                        contrib = contrib_map[dummy_name]
                        factors.append({
                            "name": col,
                            "label": f"{col.replace('_', ' ').title()} ({val})",
                            "contribution": float(contrib)
                        })
            
            return proba, factors, class_name
        else:
            pred_val = float(pipeline.predict(df_single)[0])
            
            # Post-processing clip if targets were non-negative
            target_min = float(self.original_df[self.target_col].min())
            if target_min >= 0:
                pred_val = max(target_min, pred_val)

            factors = []
            if "logistic" in model_name.lower() or "linear" in model_name.lower():
                preprocessor = pipeline.named_steps['preprocessor']
                row_preprocessed = preprocessor.transform(df_single)[0]
                coefs = pipeline.named_steps['model'].coef_
                contrib_map = dict(zip(self.feature_columns, coefs * row_preprocessed))

                # Numerical features
                for col in self.numeric_cols:
                    name = f"num__{col}"
                    if name in contrib_map:
                        contrib = contrib_map[name]
                        val = inputs.get(col, 0.0)
                        factors.append({
                            "name": col,
                            "label": f"{col.replace('_', ' ').title()} ({val})",
                            "contribution": float(contrib)
                        })

                # Categorical features
                for col in self.categorical_cols:
                    val = inputs.get(col, "Missing")
                    dummy_name = f"cat__{col}_{val}"
                    if dummy_name in contrib_map:
                        contrib = contrib_map[dummy_name]
                        factors.append({
                            "name": col,
                            "label": f"{col.replace('_', ' ').title()} ({val})",
                            "contribution": float(contrib)
                        })

            return pred_val, factors, None
