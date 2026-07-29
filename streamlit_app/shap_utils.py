"""
shap_utils.py

Utility functions for generating SHAP explanations for the
COVID Decision Support System Streamlit dashboard.
"""

import pandas as pd
import shap
import streamlit as st


# ---------------------------------------------------------------------
# SHAP Explainer
# ---------------------------------------------------------------------

@st.cache_resource
def create_shap_explainer(_model):
    """
    Create and cache a SHAP TreeExplainer for a trained tree-based model.

    Parameters
    ----------
    _model : sklearn estimator
        Trained Random Forest model.

    Returns
    -------
    shap.TreeExplainer
        Cached SHAP explainer.
    """
    return shap.TreeExplainer(_model)


# ---------------------------------------------------------------------
# Global SHAP
# ---------------------------------------------------------------------

@st.cache_data
def compute_global_shap_values(
    _explainer,
    X_scaled: pd.DataFrame,
):
    """
    Compute SHAP values for an entire dataset.

    Parameters
    ----------
    _explainer : shap.TreeExplainer
        Cached SHAP explainer.
    X_scaled : pandas.DataFrame
        Feature matrix after all preprocessing used for prediction.

    Returns
    -------
    shap.Explanation
        SHAP explanation object.
    """
    return _explainer(
        X_scaled,
        check_additivity=False,
    )


# ---------------------------------------------------------------------
# Local SHAP
# ---------------------------------------------------------------------

def compute_local_shap_values(
    explainer,
    X_scaled: pd.DataFrame,
):
    """
    Compute SHAP values for a single observation.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        Cached SHAP explainer.
    X_scaled : pandas.DataFrame
        Single-row model input after preprocessing.

    Returns
    -------
    shap.Explanation
        SHAP explanation object.
    """
    return explainer(
        X_scaled,
        check_additivity=False,
    )
