"""
SHAP explanations for tabular models (Module 2 fishing classifier, Module 6 risk scorer).
Produces the feature-importance plot the spec asks for:
  AIS Signal Absence: 35% | Restricted Area: 25% | Movement Pattern: 20% | ...
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def explain_xgboost(
    model,                          # trained XGBoost model
    X: pd.DataFrame,               # feature matrix
    feature_names: list = None,
    max_display: int = 10,
    save_path: str = None,
) -> shap.Explanation:
    """
    Compute SHAP values for an XGBoost model.
    Returns SHAP Explanation object; optionally saves a summary bar plot.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    plt.figure(figsize=(9, 5))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names or X.columns.tolist(),
        max_display=max_display,
        plot_type="bar",
        show=False,
    )
    plt.title("Feature Importance (SHAP)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return shap_values


def shap_waterfall_single(
    shap_values: shap.Explanation,
    idx: int,
    save_path: str = None,
):
    """Waterfall plot for a single vessel prediction — shows each feature's contribution."""
    plt.figure(figsize=(9, 5))
    shap.waterfall_plot(shap_values[idx], show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def get_top_factors(shap_values: shap.Explanation, feature_names: list, top_n: int = 5) -> dict:
    """
    Returns {feature: mean_abs_shap} for the top N features.
    Used to build the SHAP % breakdown in the dashboard.
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    total = mean_abs.sum()
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    return {feature_names[i]: round(float(mean_abs[i] / total) * 100, 1) for i in top_idx}
