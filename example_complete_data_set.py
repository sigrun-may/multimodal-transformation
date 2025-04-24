import pandas as pd
import os

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.2"
from rpy2.robjects import pandas2ri, globalenv, FloatVector

from multimodal_transformation import multimodal_transformation as mt
from evaluation.lasso_binary_classification import lasso_logistic_feature_selection_stratified
import numpy as np


# read csv
def read_csv(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and return a DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def generate_synthetic_data(number_of_classes) -> pd.DataFrame:
    """
    Generate synthetic data for testing.
    """
    # Generate synthetic data
    # simulate a small bimodal dataset with outliers
    np.random.seed(42)
    rng = np.random.default_rng(42)

    if number_of_classes == 3:
        data1 = np.random.normal(loc=-2, scale=1, size=20)
        data2 = np.random.normal(loc=3, scale=5, size=30)
        data3 = np.random.lognormal(size=20)
        original_data = np.hstack([data1, data2, data3])
        #original_data = np.random.normal(loc=5, scale=1, size=70)

        data1 = np.random.normal(loc=-1, scale=2, size=20)
        data2 = np.random.normal(loc=4, scale=2, size=30)
        data3 = np.random.lognormal(size=20)
        original_data2 = np.hstack([data1, data2, data3])
        #original_data2 = np.random.normal(loc=5, scale=1, size=70)

        # Add labels to the original data
        labels = ["Data1"] * len(data1) + ["Data2"] * len(data2) + ["Data3"] * len(data3)
        original_data_df = pd.DataFrame({"Label": labels, "Feature_i1": original_data, "Feature_i2": original_data2})

    elif number_of_classes == 2:
        data1 = np.random.normal(loc=-2, scale=1, size=20)
        data2 = np.random.normal(loc=3, scale=2, size=30)
        original_data = np.hstack([data1, data2])
        #original_data = np.random.normal(loc=5, scale=1, size=70)
        data1 = np.random.normal(loc=-1, scale=2, size=20)
        data2 = np.random.normal(loc=4, scale=2, size=30)
        original_data2 = np.hstack([data1, data2])
        #original_data2 = np.random.normal(loc=5, scale=1, size=70)
        # Add labels to the original data
        labels = ["Data1"] * len(data1) + ["Data2"] * len(data2)
        original_data_df = pd.DataFrame({"Label": labels, "Feature_i1": original_data, "Feature_i2": original_data2})

    # generate random features
    random_features_df = pd.DataFrame(np.random.normal(loc=5, scale=1, size=[original_data_df.shape[0], 30]))

    # generate feature names
    random_features_df.columns = [f"feature_{i}" for i in range(random_features_df.shape[1])]

    return pd.concat([original_data_df, random_features_df], axis=1)


data_df = generate_synthetic_data(2)
data_df_cp = data_df.copy()

# data_df_cp = mt.transform_data_set(data_df, interpolation_method="CubicSpline", cohens_d_value=5, inplace=True)
#
# # Use the pipeline
# pipeline_fitted = lasso_logistic_feature_selection_stratified(data_df_cp)
#
# # Get feature selection mask
# selector_mask = pipeline_fitted.named_steps['feature_selection'].get_support()
# selected_features = data_df_cp.drop(columns=["Label"]).columns[selector_mask]
# print("Selected Features:", list(selected_features))
#
# # Check final coefficients
# final_coef = pipeline_fitted.named_steps['logistic_cv'].coef_
# print("Final coefficients shape:", final_coef.shape)
# print("Final coefficients:", final_coef)
#
# # Predict on the same data
# preds = pipeline_fitted.predict(data_df_cp.drop(columns=["Label"]))
# print("Predictions:", preds)


# transform data
transformed_data_df = mt.transform_data_set(
    data_df, target_overlap=0.0001, interpolation_method="CubicSpline", inplace=True, std_list=[1,1]
)

# check whether the transformation was successful and which features were transformed
# compare the original data_df and transformed data transformed_data_df
for feature in data_df.columns[1:]:

    if "i" not in feature:
        continue
    # check whether the transformation was successful
    original_data = data_df_cp[feature].values
    transformed_data = transformed_data_df[feature].values

    # sort values
    original_data = np.sort(original_data)
    transformed_data = np.sort(transformed_data)

    # check whether the transformation was successful
    if np.array_equal(original_data, transformed_data):
        print(f"{feature} was not transformed")
    else:
        print(f"{feature} was transformed")


unique_labels = data_df["Label"].unique()
for feature in data_df.columns[1:]:
    if not "i" in feature:
        continue

    # calculate number of peaks
    # peaks = mt.count_peaks(data_df_cp[feature].values)

    # calculate the effect size before and after the transformation
    original_cohens_d = mt.cohens_d(
        data_df_cp[data_df_cp["Label"] == unique_labels[0]][feature].values,
        data_df_cp[data_df_cp["Label"] == unique_labels[1]][feature].values,
    )
    transformed_cohens_d = mt.cohens_d(
        transformed_data_df[transformed_data_df["Label"] == unique_labels[0]][feature].values,
        transformed_data_df[transformed_data_df["Label"] == unique_labels[1]][feature].values,
    )
    robust_original_cohens_d = mt.robust_cohens_d(
        data_df_cp[data_df_cp["Label"] == unique_labels[0]][feature].values,
        data_df_cp[data_df_cp["Label"] == unique_labels[1]][feature].values,
    )
    robust_transformed_cohens_d = mt.robust_cohens_d(
        transformed_data_df[transformed_data_df["Label"] == unique_labels[0]][feature].values,
        transformed_data_df[transformed_data_df["Label"] == unique_labels[1]][feature].values,
    )
    print(feature)
    # print the effect size
    print(f"Effect size before transformation: {original_cohens_d:.4f}")
    print(f"Effect size after transformation: {transformed_cohens_d:.4f}")
    print(f"Robust effect size before transformation: {robust_original_cohens_d:.4f}")
    print(f"Robust effect size after transformation: {robust_transformed_cohens_d:.4f}")
    # print(mt.is_bimodal(data_df_cp[feature].values, verbose=True))
    print(" ")
