import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils import save_json

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def train_models(X_train, X_test, y_train, y_test, raw_data_path="data/raw/Customer-Churn.csv", task_type="classification"):
    results = {}
    
    if task_type == "classification":
        imbalance = False
        class_ratio = float((y_train == 1).mean())
        if class_ratio < 0.35 or class_ratio > 0.65:
            imbalance = True
        
        w_mode = "balanced" if imbalance else None
        
        log_model = LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            class_weight=w_mode,
            random_state=42
        )
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)
        
        results["logistic_regression"] = {
            "accuracy": float(accuracy_score(y_test, y_pred_log)),
            "precision": float(precision_score(y_test, y_pred_log, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_log, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred_log, zero_division=0))
        }
        
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight=w_mode)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        results["random_forest"] = {
            "accuracy": float(accuracy_score(y_test, y_pred_rf)),
            "precision": float(precision_score(y_test, y_pred_rf, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_rf, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred_rf, zero_division=0))
        }
    else:
        from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
        
        log_model = RidgeCV(alphas=np.logspace(-3, 3, 10))
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)
        
        mape_lin = float(np.mean(np.abs((y_test - y_pred_log) / np.clip(np.abs(y_test), 1e-5, None))))
        
        results["linear_regression"] = {
            "r2_score": float(r2_score(y_test, y_pred_log)),
            "mae": float(mean_absolute_error(y_test, y_pred_log)),
            "rmse": float(root_mean_squared_error(y_test, y_pred_log)),
            "mape": mape_lin
        }
        
        rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        mape_rf = float(np.mean(np.abs((y_test - y_pred_rf) / np.clip(np.abs(y_test), 1e-5, None))))
        
        results["random_forest"] = {
            "r2_score": float(r2_score(y_test, y_pred_rf)),
            "mae": float(mean_absolute_error(y_test, y_pred_rf)),
            "rmse": float(root_mean_squared_error(y_test, y_pred_rf)),
            "mape": mape_rf
        }
        
    # Write dynamic model script files
    model_path_str = str(raw_data_path).replace("\\", "\\\\")
    
    if task_type == "classification":
        with open(MODELS_DIR / "logistic_model.py", "w") as f:
            f.write(f'''import pandas as pd
from sklearn.linear_model import LogisticRegression
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def get_model():
    from src.preprocess import preprocess_pipeline
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        "{model_path_str}", task_type_override="classification"
    )
    imbalance = False
    class_ratio = float((y_train == 1).mean())
    if class_ratio < 0.35 or class_ratio > 0.65:
        imbalance = True
    w_mode = "balanced" if imbalance else None
    model = LogisticRegression(max_iter=5000, solver="liblinear", class_weight=w_mode, random_state=42)
    model.fit(X_train, y_train)
    return model
''')
        
        with open(MODELS_DIR / "random_forest.py", "w") as f:
            f.write(f'''import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def get_model():
    from src.preprocess import preprocess_pipeline
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        "{model_path_str}", task_type_override="classification"
    )
    imbalance = False
    class_ratio = float((y_train == 1).mean())
    if class_ratio < 0.35 or class_ratio > 0.65:
        imbalance = True
    w_mode = "balanced" if imbalance else None
    model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight=w_mode)
    model.fit(X_train, y_train)
    return model
''')
    else:
        with open(MODELS_DIR / "logistic_model.py", "w") as f:
            f.write(f'''import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def get_model():
    from src.preprocess import preprocess_pipeline
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        "{model_path_str}", task_type_override="regression"
    )
    model = RidgeCV(alphas=np.logspace(-3, 3, 10))
    model.fit(X_train, y_train)
    return model
''')
        
        with open(MODELS_DIR / "random_forest.py", "w") as f:
            f.write(f'''import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def get_model():
    from src.preprocess import preprocess_pipeline
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        "{model_path_str}", task_type_override="regression"
    )
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model
''')
            
    save_json(results, REPORTS_DIR / "model_metrics.json")
    return results