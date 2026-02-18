import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

def evaluate_all(fitted_models, X_test, y_test):
    records = []
    probas = {}

    for name, model in fitted_models.items():
        p = model.predict_proba(X_test)[:, 1]
        probas[name] = p

        auc   = roc_auc_score(y_test, p)
        ap    = average_precision_score(y_test, p)

        records.append({"model": name, "roc_auc": auc, "avg_precision": ap})

    results = pd.DataFrame(records).sort_values("roc_auc", ascending=False)
    return results, probas

def roc_data(y_test, probas):
    from sklearn.metrics import roc_curve
    curves = {}
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        curves[name] = (fpr, tpr)
    return curves

def pr_data(y_test, probas):
    curves = {}
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        curves[name] = (prec, rec)
    return curves

def model_calibration_data(y_test, probas, n_bins=10):
    cal = {}
    for name, p in probas.items():
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=n_bins, strategy="quantile")
        cal[name] = (mean_pred, frac_pos)
    return cal

def feature_importance(fitted_models, feat_cols):
    rows = []

    rf  = fitted_models.get("rf")
    xgb = fitted_models.get("xgb")

    if rf:
        for feat, imp in zip(feat_cols, rf.feature_importances_):
            rows.append({"model": "rf", "feature": feat, "importance": imp})

    if xgb:
        for feat, imp in zip(feat_cols, xgb.feature_importances_):
            rows.append({"model": "xgb", "feature": feat, "importance": imp})

    return pd.DataFrame(rows)