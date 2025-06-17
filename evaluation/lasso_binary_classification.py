"""
Module: lasso_binary_classification

This module demonstrates how to use L1-regularized Logistic Regression
(aka Lasso-style) with feature selection and stratified cross-validation
for a binary classification target.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def lasso_logistic_feature_selection_stratified(
    df: pd.DataFrame, label_col: str = "Label", cv_splits: int = 5, random_state: int = 42, threshold: str = "mean"
):
    """
    Perform a pipeline with L1-regularized Logistic Regression for binary classification,
    including feature selection and stratified cross-validation.

    Pipeline steps:
        1) SelectFromModel(LogisticRegressionCV) to filter out low-importance features.
        2) LogisticRegressionCV on the reduced set of features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and a binary 'Label' column.
    label_col : str, optional
        Name of the label (target) column, by default "Label".
    cv_splits : int, optional
        Number of folds for stratified cross-validation, by default 5.
    random_state : int, optional
        Random seed for reproducibility, by default 42.
    threshold : str or float, optional
        Threshold for SelectFromModel; defaults to "mean". Could be "median", a float, etc.

    Returns
    -------
    Pipeline
        Fitted scikit-learn Pipeline with:
          - feature_selection: SelectFromModel(LogisticRegressionCV)
          - logistic_cv: LogisticRegressionCV
    """
    # Separate features (X) and the binary target (y)
    y = df[label_col]
    X = df.drop(columns=[label_col])

    # Define a StratifiedKFold object for cross-validation
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # First step: Feature selection using L1-regularized logistic regression
    feature_selector = SelectFromModel(
        estimator=LogisticRegressionCV(
            cv=skf,
            penalty="l1",  # L1 penalty -> "LASSO-like" for classification
            solver="saga",  # 'saga' solver supports L1 for LogisticRegressionCV
            random_state=random_state,
            max_iter=10000,  # Increase iterations for convergence
            scoring="accuracy",  # Or your preferred classification metric
        ),
        threshold=threshold,  # "mean", "median", or a float
    )

    # Second step: Final logistic regression with CV
    final_log_reg = LogisticRegressionCV(
        cv=skf, penalty="l1", solver="saga", random_state=random_state, max_iter=10000, scoring="accuracy"
    )

    # Combine steps into a pipeline
    pipeline = Pipeline([("feature_selection", feature_selector), ("logistic_cv", final_log_reg)])

    # Fit the pipeline
    pipeline.fit(X, y)

    return pipeline


# Example usage (if running as a script):
if __name__ == "__main__":
    # Example: create a dummy dataset
    import numpy as np

    # Suppose we have 100 samples, 10 features, and a binary label
    np.random.seed(42)
    dummy_X = np.random.randn(100, 10)
    dummy_y = np.random.choice([0, 1], size=100, p=[0.4, 0.6])
    columns = [f"Feature_{i}" for i in range(10)]
    df_dummy = pd.DataFrame(dummy_X, columns=columns)
    df_dummy["Label"] = dummy_y

    # Use the pipeline
    pipeline_fitted = lasso_logistic_feature_selection_stratified(df_dummy)

    # Get feature selection mask
    selector_mask = pipeline_fitted.named_steps["feature_selection"].get_support()
    selected_features = df_dummy.drop(columns=["Label"]).columns[selector_mask]
    print("Selected Features:", list(selected_features))

    # Check final coefficients
    final_coef = pipeline_fitted.named_steps["logistic_cv"].coef_
    print("Final coefficients shape:", final_coef.shape)
    print("Final coefficients:", final_coef)

    # Predict on the same data
    preds = pipeline_fitted.predict(df_dummy.drop(columns=["Label"]))
    print("Predictions:", preds)
