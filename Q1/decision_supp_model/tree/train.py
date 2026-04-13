from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import KFold, StratifiedKFold


# =========================
# Paths
# =========================
INPUT_CSV = Path("/data/shared/fsibilla/clean_code/Q1/decision_supp_model/full.csv")
OUT_DIR = Path("/data/shared/fsibilla/clean_code/Q1/decision_supp_model/tree")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Config
# =========================
FEATURES = [#"frac", 
    "variability_intrinsic_mean","n_train_mean"]
REG_TARGETS = ["improvement_mean", "improvement_std"]
CLS_TARGET = "target_improvement_positive"
GROUP_COL = "experiment"

RANDOM_STATE = 42
N_INNER_SPLITS = 5

REG_PARAM_GRID = [
    {"max_depth": 2, "min_samples_leaf": 5,  "min_samples_split": 10},
    {"max_depth": 3, "min_samples_leaf": 5,  "min_samples_split": 10},
    {"max_depth": 4, "min_samples_leaf": 5,  "min_samples_split": 10},
    {"max_depth": 1, "min_samples_leaf": 5,  "min_samples_split": 10},
    {"max_depth": 3, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": 4, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": 1, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": None, "min_samples_leaf": 10, "min_samples_split": 20},
]

CLS_PARAM_GRID = [
    {"max_depth": 2, "min_samples_leaf": 5,  "min_samples_split": 10},
    {"max_depth": 3, "min_samples_leaf": 5,  "min_samples_split": 10},
    {"max_depth": 4, "min_samples_leaf": 5,  "min_samples_split": 10},
    {"max_depth": 1, "min_samples_leaf": 5,  "min_samples_split": 10},
    {"max_depth": 3, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": 4, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": 1, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": None, "min_samples_leaf": 10, "min_samples_split": 20},
]


# =========================
# Helpers
# =========================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(r2_score(y_true, y_pred))


def safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def regression_metrics(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, prefix=""):
    out = {}
    for t in REG_TARGETS:
        yt = y_true_df[t].values
        yp = y_pred_df[t].values
        out[f"{prefix}{t}_mae"] = float(mean_absolute_error(yt, yp))
        out[f"{prefix}{t}_rmse"] = rmse(yt, yp)
        out[f"{prefix}{t}_r2"] = safe_r2(yt, yp)

    out[f"{prefix}mean_mae_across_targets"] = float(
        np.mean([out[f"{prefix}{t}_mae"] for t in REG_TARGETS])
    )
    out[f"{prefix}mean_rmse_across_targets"] = float(
        np.mean([out[f"{prefix}{t}_rmse"] for t in REG_TARGETS])
    )
    return out


def classification_metrics(y_true, y_pred_label, y_pred_proba, prefix=""):
    out = {}
    out[f"{prefix}roc_auc"] = safe_roc_auc(y_true, y_pred_proba)
    out[f"{prefix}accuracy"] = float(accuracy_score(y_true, y_pred_label))
    out[f"{prefix}precision"] = float(precision_score(y_true, y_pred_label, zero_division=0))
    out[f"{prefix}recall"] = float(recall_score(y_true, y_pred_label, zero_division=0))
    out[f"{prefix}f1"] = float(f1_score(y_true, y_pred_label, zero_division=0))
    out[f"{prefix}positive_rate_true"] = float(np.mean(y_true))
    out[f"{prefix}positive_rate_pred"] = float(np.mean(y_pred_label))
    return out


def fit_reg_model(X_train, y_train, params):
    model = DecisionTreeRegressor(random_state=RANDOM_STATE, **params)
    model.fit(X_train, y_train)
    return model


def fit_cls_model(X_train, y_train, params):
    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        **params
    )
    model.fit(X_train, y_train)
    return model


def predict_positive_proba(model, X):
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        idx = classes.index(1)
        return proba[:, idx]
    return np.zeros(len(X), dtype=float)


# =========================
# Inner random CV: regression
# =========================
def inner_random_cv_select_reg_params(train_df: pd.DataFrame, n_splits: int = 5):
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    all_param_rows = []

    X = train_df[FEATURES].reset_index(drop=True)
    Y = train_df[REG_TARGETS].reset_index(drop=True)
    meta = train_df[["experiment", "adm1", "variable", "psu_num"]].reset_index(drop=True)

    for params in REG_PARAM_GRID:
        params_key = json.dumps(params, sort_keys=True)
        fold_rows = []

        for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(X), start=1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            Y_tr, Y_va = Y.iloc[tr_idx], Y.iloc[va_idx]

            model = fit_reg_model(X_tr, Y_tr, params)
            pred = model.predict(X_va)
            pred_df = pd.DataFrame(pred, columns=REG_TARGETS, index=Y_va.index)

            metrics = regression_metrics(Y_va, pred_df[REG_TARGETS], prefix="")
            row = {
                "params_key": params_key,
                **params,
                "validation_fold": fold_id,
                "n_train_rows": len(tr_idx),
                "n_val_rows": len(va_idx),
            }
            row.update(metrics)
            fold_rows.append(row)

        if fold_rows:
            fold_df = pd.DataFrame(fold_rows)
            agg = {
                "params_key": params_key,
                **params,
                "n_inner_folds": len(fold_df),
                "cv_mean_mae_across_targets": float(fold_df["mean_mae_across_targets"].mean()),
                "cv_mean_rmse_across_targets": float(fold_df["mean_rmse_across_targets"].mean()),
            }
            for t in REG_TARGETS:
                agg[f"cv_{t}_mae"] = float(fold_df[f"{t}_mae"].mean())
                agg[f"cv_{t}_rmse"] = float(fold_df[f"{t}_rmse"].mean())
                agg[f"cv_{t}_r2"] = float(fold_df[f"{t}_r2"].mean(skipna=True))
            all_param_rows.append(agg)

    summary = pd.DataFrame(all_param_rows).sort_values(
        ["cv_mean_mae_across_targets", "cv_mean_rmse_across_targets"],
        ascending=[True, True]
    ).reset_index(drop=True)

    if summary.empty:
        raise RuntimeError("Inner regression CV produced no results.")

    best_row = summary.iloc[0]
    best_params = {
        "max_depth": None if pd.isna(best_row["max_depth"]) else best_row["max_depth"],
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "min_samples_split": int(best_row["min_samples_split"]),
    }
    if best_params["max_depth"] is not None:
        best_params["max_depth"] = int(best_params["max_depth"])

    oof_parts = []
    for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        Y_tr, Y_va = Y.iloc[tr_idx], Y.iloc[va_idx]
        meta_va = meta.iloc[va_idx].copy()

        model = fit_reg_model(X_tr, Y_tr, best_params)
        pred = model.predict(X_va)

        tmp = meta_va.copy()
        tmp["validation_fold"] = fold_id
        tmp["params_key"] = json.dumps(best_params, sort_keys=True)
        for i, t in enumerate(REG_TARGETS):
            tmp[f"true_{t}"] = Y_va[t].values
            tmp[f"pred_{t}"] = pred[:, i]
        oof_parts.append(tmp)

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return best_params, summary, oof


# =========================
# Inner random CV: classification
# =========================
def inner_random_cv_select_cls_params(train_df: pd.DataFrame, n_splits: int = 5):
    X = train_df[FEATURES].reset_index(drop=True)
    y = train_df[CLS_TARGET].reset_index(drop=True)
    meta = train_df[["experiment", "adm1", "variable", "psu_num"]].reset_index(drop=True)

    min_class_count = y.value_counts().min()
    n_splits_eff = min(n_splits, int(min_class_count)) if len(y.unique()) > 1 else 0

    if n_splits_eff < 2:
        default_params = {"max_depth": 3, "min_samples_leaf": 5, "min_samples_split": 10}
        model = fit_cls_model(X, y, default_params)
        pred_label = model.predict(X)
        pred_proba = predict_positive_proba(model, X)

        oof = meta.copy()
        oof["validation_fold"] = 1
        oof["params_key"] = json.dumps(default_params, sort_keys=True)
        oof[f"true_{CLS_TARGET}"] = y.values
        oof[f"pred_{CLS_TARGET}"] = pred_label
        oof[f"pred_proba_{CLS_TARGET}"] = pred_proba

        summary = pd.DataFrame([{
            "params_key": json.dumps(default_params, sort_keys=True),
            **default_params,
            "n_inner_folds": 0,
            "note": "Too few minority examples for StratifiedKFold; used default params."
        }])
        return default_params, summary, oof

    splitter = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=RANDOM_STATE)
    all_param_rows = []

    for params in CLS_PARAM_GRID:
        params_key = json.dumps(params, sort_keys=True)
        fold_rows = []

        for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(X, y), start=1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            if y_tr.nunique() < 2:
                continue

            model = fit_cls_model(X_tr, y_tr, params)
            pred_label = model.predict(X_va)
            pred_proba = predict_positive_proba(model, X_va)

            metrics = classification_metrics(y_va.values, pred_label, pred_proba, prefix="")
            row = {
                "params_key": params_key,
                **params,
                "validation_fold": fold_id,
                "n_train_rows": len(tr_idx),
                "n_val_rows": len(va_idx),
            }
            row.update(metrics)
            fold_rows.append(row)

        if fold_rows:
            fold_df = pd.DataFrame(fold_rows)
            agg = {
                "params_key": params_key,
                **params,
                "n_inner_folds": len(fold_df),
                "cv_roc_auc": float(fold_df["roc_auc"].mean(skipna=True)),
                "cv_f1": float(fold_df["f1"].mean()),
                "cv_accuracy": float(fold_df["accuracy"].mean()),
            }
            all_param_rows.append(agg)

    summary = pd.DataFrame(all_param_rows)
    if summary.empty:
        default_params = {"max_depth": 3, "min_samples_leaf": 5, "min_samples_split": 10}
        model = fit_cls_model(X, y, default_params)
        pred_label = model.predict(X)
        pred_proba = predict_positive_proba(model, X)

        oof = meta.copy()
        oof["validation_fold"] = 1
        oof["params_key"] = json.dumps(default_params, sort_keys=True)
        oof[f"true_{CLS_TARGET}"] = y.values
        oof[f"pred_{CLS_TARGET}"] = pred_label
        oof[f"pred_proba_{CLS_TARGET}"] = pred_proba

        summary = pd.DataFrame([{
            "params_key": json.dumps(default_params, sort_keys=True),
            **default_params,
            "n_inner_folds": 0,
            "note": "Inner classification CV produced no valid folds; used default params."
        }])
        return default_params, summary, oof

    summary = summary.sort_values(
        ["cv_roc_auc", "cv_f1", "cv_accuracy"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    best_row = summary.iloc[0]
    best_params = {
        "max_depth": None if pd.isna(best_row["max_depth"]) else best_row["max_depth"],
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "min_samples_split": int(best_row["min_samples_split"]),
    }
    if best_params["max_depth"] is not None:
        best_params["max_depth"] = int(best_params["max_depth"])

    oof_parts = []
    for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        meta_va = meta.iloc[va_idx].copy()

        if y_tr.nunique() < 2:
            continue

        model = fit_cls_model(X_tr, y_tr, best_params)
        pred_label = model.predict(X_va)
        pred_proba = predict_positive_proba(model, X_va)

        tmp = meta_va.copy()
        tmp["validation_fold"] = fold_id
        tmp["params_key"] = json.dumps(best_params, sort_keys=True)
        tmp[f"true_{CLS_TARGET}"] = y_va.values
        tmp[f"pred_{CLS_TARGET}"] = pred_label
        tmp[f"pred_proba_{CLS_TARGET}"] = pred_proba
        oof_parts.append(tmp)

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return best_params, summary, oof


# =========================
# Load and prepare data
# =========================
df = pd.read_csv(INPUT_CSV)

required_cols = [GROUP_COL, "adm1", "variable", "psu_num"] + FEATURES + REG_TARGETS
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

work_df = df[required_cols].copy()

for c in ["psu_num"] + FEATURES + REG_TARGETS:
    work_df[c] = pd.to_numeric(work_df[c], errors="coerce")

work_df = work_df.dropna(
    subset=[GROUP_COL, "adm1", "variable", "psu_num"] + FEATURES + REG_TARGETS
).reset_index(drop=True)

work_df[CLS_TARGET] = (work_df["improvement_mean"] > 0).astype(int)

experiments = sorted(work_df[GROUP_COL].unique().tolist())
if len(experiments) < 2:
    raise ValueError("Need at least 2 experiments for leave-one-experiment-out evaluation.")

print(f"Loaded {len(work_df)} rows from {INPUT_CSV}")
print(f"Experiments ({len(experiments)}): {experiments}")


# =========================
# Outer loop
# =========================
all_reg_test_parts = []
all_reg_val_parts = []
all_cls_test_parts = []
all_cls_val_parts = []

reg_fold_metric_rows = []
cls_fold_metric_rows = []

reg_feature_importance_rows = []
cls_feature_importance_rows = []

reg_inner_summary_parts = []
cls_inner_summary_parts = []

for test_exp in experiments:
    print(f"\n=== Outer fold: test experiment = {test_exp} ===")

    train_df = work_df.loc[work_df[GROUP_COL] != test_exp].copy().reset_index(drop=True)
    test_df = work_df.loc[work_df[GROUP_COL] == test_exp].copy().reset_index(drop=True)

    if train_df.empty or test_df.empty:
        print(f"Skipping {test_exp} because train or test is empty.")
        continue

    # -------------------------
    # Regression
    # -------------------------
    reg_best_params, reg_inner_summary, reg_val_pred = inner_random_cv_select_reg_params(
        train_df, n_splits=N_INNER_SPLITS
    )
    reg_inner_summary = reg_inner_summary.copy()
    reg_inner_summary["outer_test_experiment"] = test_exp
    reg_inner_summary_parts.append(reg_inner_summary)

    if not reg_val_pred.empty:
        reg_val_pred = reg_val_pred.copy()
        reg_val_pred["outer_test_experiment"] = test_exp
        all_reg_val_parts.append(reg_val_pred)

    reg_model = fit_reg_model(train_df[FEATURES], train_df[REG_TARGETS], reg_best_params)
    reg_model_path = MODELS_DIR / f"reg_tree_outer_test_{test_exp}.joblib"
    dump(reg_model, reg_model_path)

    reg_test_pred = reg_model.predict(test_df[FEATURES])
    reg_test_out = test_df[["experiment", "adm1", "variable", "psu_num"] + FEATURES + REG_TARGETS].copy()
    reg_test_out["outer_test_experiment"] = test_exp
    reg_test_out["model_path"] = str(reg_model_path)
    reg_test_out["best_params"] = json.dumps(reg_best_params, sort_keys=True)

    for t in REG_TARGETS:
        reg_test_out.rename(columns={t: f"true_{t}"}, inplace=True)
    for i, t in enumerate(REG_TARGETS):
        reg_test_out[f"pred_{t}"] = reg_test_pred[:, i]

    all_reg_test_parts.append(reg_test_out)

    reg_true_df = pd.DataFrame({
        "improvement_mean": reg_test_out["true_improvement_mean"].values,
        "improvement_std": reg_test_out["true_improvement_std"].values,
    })
    reg_pred_df = pd.DataFrame({
        "improvement_mean": reg_test_out["pred_improvement_mean"].values,
        "improvement_std": reg_test_out["pred_improvement_std"].values,
    })

    reg_metrics = {
        "outer_test_experiment": test_exp,
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_train_experiments": train_df[GROUP_COL].nunique(),
        "n_test_experiments": test_df[GROUP_COL].nunique(),
        "best_params": json.dumps(reg_best_params, sort_keys=True),
        "model_path": str(reg_model_path),
    }
    reg_metrics.update(regression_metrics(reg_true_df, reg_pred_df, prefix="test_"))
    reg_fold_metric_rows.append(reg_metrics)

    for feat, imp in zip(FEATURES, reg_model.feature_importances_):
        reg_feature_importance_rows.append({
            "outer_test_experiment": test_exp,
            "feature": feat,
            "importance": float(imp),
            "best_params": json.dumps(reg_best_params, sort_keys=True),
        })

    # -------------------------
    # Classification
    # -------------------------
    cls_best_params, cls_inner_summary, cls_val_pred = inner_random_cv_select_cls_params(
        train_df, n_splits=N_INNER_SPLITS
    )
    cls_inner_summary = cls_inner_summary.copy()
    cls_inner_summary["outer_test_experiment"] = test_exp
    cls_inner_summary_parts.append(cls_inner_summary)

    if not cls_val_pred.empty:
        cls_val_pred = cls_val_pred.copy()
        cls_val_pred["outer_test_experiment"] = test_exp
        all_cls_val_parts.append(cls_val_pred)

    if train_df[CLS_TARGET].nunique() < 2:
        print(f"Skipping classification for {test_exp}: outer train has only one class.")
    else:
        cls_model = fit_cls_model(train_df[FEATURES], train_df[CLS_TARGET], cls_best_params)
        cls_model_path = MODELS_DIR / f"cls_tree_outer_test_{test_exp}.joblib"
        dump(cls_model, cls_model_path)

        cls_pred_label = cls_model.predict(test_df[FEATURES])
        cls_pred_proba = predict_positive_proba(cls_model, test_df[FEATURES])

        cls_test_out = test_df[["experiment", "adm1", "variable", "psu_num"] + FEATURES].copy()
        cls_test_out["outer_test_experiment"] = test_exp
        cls_test_out["model_path"] = str(cls_model_path)
        cls_test_out["best_params"] = json.dumps(cls_best_params, sort_keys=True)
        cls_test_out[f"true_{CLS_TARGET}"] = test_df[CLS_TARGET].values
        cls_test_out[f"pred_{CLS_TARGET}"] = cls_pred_label
        cls_test_out[f"pred_proba_{CLS_TARGET}"] = cls_pred_proba
        cls_test_out["true_improvement_mean"] = test_df["improvement_mean"].values
        cls_test_out["true_improvement_std"] = test_df["improvement_std"].values

        all_cls_test_parts.append(cls_test_out)

        cls_metrics = {
            "outer_test_experiment": test_exp,
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "n_train_experiments": train_df[GROUP_COL].nunique(),
            "n_test_experiments": test_df[GROUP_COL].nunique(),
            "best_params": json.dumps(cls_best_params, sort_keys=True),
            "model_path": str(cls_model_path),
        }
        cls_metrics.update(classification_metrics(
            test_df[CLS_TARGET].values,
            cls_pred_label,
            cls_pred_proba,
            prefix="test_"
        ))
        cls_fold_metric_rows.append(cls_metrics)

        for feat, imp in zip(FEATURES, cls_model.feature_importances_):
            cls_feature_importance_rows.append({
                "outer_test_experiment": test_exp,
                "feature": feat,
                "importance": float(imp),
                "best_params": json.dumps(cls_best_params, sort_keys=True),
            })


# =========================
# Save outputs
# =========================
reg_test_predictions_df = pd.concat(all_reg_test_parts, ignore_index=True) if all_reg_test_parts else pd.DataFrame()
reg_val_predictions_df = pd.concat(all_reg_val_parts, ignore_index=True) if all_reg_val_parts else pd.DataFrame()
cls_test_predictions_df = pd.concat(all_cls_test_parts, ignore_index=True) if all_cls_test_parts else pd.DataFrame()
cls_val_predictions_df = pd.concat(all_cls_val_parts, ignore_index=True) if all_cls_val_parts else pd.DataFrame()

reg_fold_metrics_df = pd.DataFrame(reg_fold_metric_rows)
cls_fold_metrics_df = pd.DataFrame(cls_fold_metric_rows)

reg_feature_importances_df = pd.DataFrame(reg_feature_importance_rows)
cls_feature_importances_df = pd.DataFrame(cls_feature_importance_rows)

reg_inner_cv_summary_df = pd.concat(reg_inner_summary_parts, ignore_index=True) if reg_inner_summary_parts else pd.DataFrame()
cls_inner_cv_summary_df = pd.concat(cls_inner_summary_parts, ignore_index=True) if cls_inner_summary_parts else pd.DataFrame()

reg_test_predictions_df.to_csv(OUT_DIR / "reg_test_predictions.csv", index=False)
reg_val_predictions_df.to_csv(OUT_DIR / "reg_validation_predictions.csv", index=False)
reg_fold_metrics_df.to_csv(OUT_DIR / "reg_fold_test_metrics.csv", index=False)
reg_feature_importances_df.to_csv(OUT_DIR / "reg_feature_importances.csv", index=False)
reg_inner_cv_summary_df.to_csv(OUT_DIR / "reg_inner_cv_summary.csv", index=False)

cls_test_predictions_df.to_csv(OUT_DIR / "cls_test_predictions.csv", index=False)
cls_val_predictions_df.to_csv(OUT_DIR / "cls_validation_predictions.csv", index=False)
cls_fold_metrics_df.to_csv(OUT_DIR / "cls_fold_test_metrics.csv", index=False)
cls_feature_importances_df.to_csv(OUT_DIR / "cls_feature_importances.csv", index=False)
cls_inner_cv_summary_df.to_csv(OUT_DIR / "cls_inner_cv_summary.csv", index=False)

metadata = {
    "n_rows_used": int(len(work_df)),
    "n_experiments": int(len(experiments)),
    "features": FEATURES,
    "regression_targets": REG_TARGETS,
    "classification_target": f"{CLS_TARGET} = 1(improvement_mean > 0)",
    "group_col": GROUP_COL,
    "outer_cv": "leave-one-experiment-out",
    "inner_cv_regression": f"KFold(n_splits={N_INNER_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
    "inner_cv_classification": f"StratifiedKFold(n_splits<= {N_INNER_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
    "classification_balancing": "class_weight='balanced'",
    "regression_model": "DecisionTreeRegressor (multi-output)",
    "classification_model": "DecisionTreeClassifier",
    "random_state": RANDOM_STATE,
    "regression_param_grid": REG_PARAM_GRID,
    "classification_param_grid": CLS_PARAM_GRID,
}

if not reg_fold_metrics_df.empty:
    metadata["reg_mean_test_mae_across_targets"] = float(reg_fold_metrics_df["test_mean_mae_across_targets"].mean())
    metadata["reg_mean_test_rmse_across_targets"] = float(reg_fold_metrics_df["test_mean_rmse_across_targets"].mean())

if not cls_fold_metrics_df.empty:
    metadata["cls_mean_test_roc_auc"] = float(cls_fold_metrics_df["test_roc_auc"].mean(skipna=True))
    metadata["cls_mean_test_f1"] = float(cls_fold_metrics_df["test_f1"].mean())
    metadata["cls_mean_test_accuracy"] = float(cls_fold_metrics_df["test_accuracy"].mean())

with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved regression outputs:")
print(f"  {OUT_DIR / 'reg_test_predictions.csv'}")
print(f"  {OUT_DIR / 'reg_validation_predictions.csv'}")
print(f"  {OUT_DIR / 'reg_fold_test_metrics.csv'}")
print(f"  {OUT_DIR / 'reg_feature_importances.csv'}")
print(f"  {OUT_DIR / 'reg_inner_cv_summary.csv'}")

print("\nSaved classification outputs:")
print(f"  {OUT_DIR / 'cls_test_predictions.csv'}")
print(f"  {OUT_DIR / 'cls_validation_predictions.csv'}")
print(f"  {OUT_DIR / 'cls_fold_test_metrics.csv'}")
print(f"  {OUT_DIR / 'cls_feature_importances.csv'}")
print(f"  {OUT_DIR / 'cls_inner_cv_summary.csv'}")

print(f"\nSaved metadata: {OUT_DIR / 'metadata.json'}")
print(f"Saved models in: {MODELS_DIR}")
print("\nDone.")