import pandas as pd
import numpy as np

import shap 


def shap_tree(model, data):
    """
    Returns features sorted by importance for tree-based models using SHAP
    """
    explainer = shap.TreeExplainer(model)
    return get_shap_values(explainer, data[model.feature_names_])

def get_shap_values(explainer, data):
    """
    Returns sorted features by shap
    """
    shap_values = explainer(data)
    feature_names = shap_values.feature_names
    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    vals = np.abs(shap_df.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
    return shap_importance.sort_values(by=['feature_importance_vals'], ascending=False)#.col_name.tolist()
