import pandas as pd
import numpy as np
from multimodal_transformation.multimodal_transformation import (
    transform_data,
    transform_data_set,
)
from multimodal_transformation.ideal_distribution_generator import simulate_multimodal_distribution
import pytest


def generate_synthetic_data():
    """Generate synthetic data for testing."""
    # simulate a small bimodal dataset with outliers
    np.random.seed(42)
    data1 = np.random.normal(loc=-1, scale=1, size=10)
    data3 = np.random.lognormal(size=20)
    original_data = np.hstack([data1, data3])
    # original_data = np.random.normal(loc=5, scale=1, size=60)

    data1 = np.random.normal(loc=-1, scale=1, size=10)
    data3 = np.random.lognormal(size=20)
    original_data2 = np.hstack([data1, data3])
    # original_data = np.random.normal(loc=5, scale=1, size=60)

    # Add labels to the original data
    labels = ["Data1"] * len(data1) + ["Data3"] * len(data3)
    original_data_df = pd.DataFrame({"Label": labels, "Feature_i1": original_data, "Feature_i2": original_data2})

    # insert 30 more features into the original data
    for i in range(30):
        original_data_df[f"Feature{i}"] = np.random.normal(loc=5, scale=1, size=30)

    return original_data_df


def test_transform_data():
    # Example data
    source_data = {"Label": ["A", "B", "A", "B"], "Feature": [1.0, 2.0, 3.0, 4.0]}
    target_data = {"Label": ["A", "B", "A", "B"], "Feature": [10.0, 20.0, 30.0, 40.0]}

    source_df = pd.DataFrame(source_data)
    target_df = pd.DataFrame(target_data)

    transformed_df = transform_data(source_df.copy(), target_df, "Feature", interpolation_method="Linear")

    assert "original_index" not in transformed_df.columns, (
        "The original_index column should not be present in the " "transformed DataFrame."
    )

    # Test 1: Check order of samples
    assert (transformed_df.index == source_df.index).all(), "The order of samples has changed."

    # Test 2: Check if labels are the same
    assert (transformed_df["Label"] == source_df["Label"]).all(), "The labels do not match."

    # check if the Label column from transformed_data_df equals the Label column form data_df
    assert transformed_df["Label"].equals(source_df["Label"]), (
        "The transformed data must have the same labels as " "the original data."
    )

    transformed_df2 = transform_data(source_df.copy(), target_df, "Feature", interpolation_method="CubicSpline")

    # Test 1: Check order of samples
    assert (transformed_df2.index == source_df.index).all(), "The order of samples has changed."

    # Test 2: Check if labels are the same
    assert (transformed_df2["Label"] == source_df["Label"]).all(), "The labels do not match."


def test_transform_data_set():
    # Example data
    source_df = generate_synthetic_data()

    # Call the transformation function
    transformed_df = transform_data_set(source_df.copy(), interpolation_method="Linear")

    # Test 1: Check if the transformed DataFrame has the same number of samples as the original data
    assert transformed_df.shape[0] == source_df.shape[0], (
        f"The number of rows of the DataFrame has changed from " f"{source_df.shape[0]} to {transformed_df.shape[0]}."
    )

    # Test 2: Check if the order of samples remains the same
    assert (transformed_df.index == source_df.index).all(), "The order of samples has changed."

    # Test 3: Check if the labels remain the same before and after transformation
    assert (transformed_df["Label"] == source_df["Label"]).all(), "The labels have changed."

    # Test 4: Check if the column names remain the same
    for original_column, transformed_column in zip(source_df.columns, transformed_df.columns):
        assert original_column == transformed_column, (
            f"The column names have changed from {original_column} to " f"{transformed_column}."
        )

    # Test 5: Check if the data types remain the same
    for original_dtype, transformed_dtype in zip(source_df.dtypes, transformed_df.dtypes):
        assert original_dtype == transformed_dtype, (
            f"The data types have changed from {original_dtype} to " f"{transformed_dtype}."
        )

    # Test 6: Check if the transformed DataFrame has the same number of columns
    assert transformed_df.shape[1] == source_df.shape[1], (
        f"The number of columns of the DataFrame has changed from "
        f"{source_df.shape[1]} to {transformed_df.shape[1]}."
    )


def test_create_ideal_artificial_multimodal_distribution():
    # Example data
    source_data = {"Label": ["A", "B", "A", "B"], "Feature": [1.0, 2.0, 3.0, 4.0]}
    source_df = pd.DataFrame(source_data)

    # Call the function to create an ideal artificial multimodal distribution
    ideal_df = simulate_multimodal_distribution(
        source_df.copy(),
        "Feature",
        scale_factor=10,
    )

    # check if the ideal target distribution has the same labels as the original data
    assert set(ideal_df["Label"].unique()) == set(
        source_df["Label"].unique()
    ), "The ideal target distribution must have the same labels as the original data."

    # Test 1: Check if the DataFrame has the correct number of rows
    expected_rows = len(source_df) * 10
    assert ideal_df.shape[0] == expected_rows, (
        f"The number of rows should be {expected_rows}, but got " f"{ideal_df.shape[0]}."
    )

    # Test 2: Check if the DataFrame has the correct columns
    expected_columns = ["Label", "Feature"]
    assert list(ideal_df.columns) == expected_columns, (
        f"The columns should be {expected_columns}, but got " f"{list(ideal_df.columns)}."
    )

    # Test 3: Check if the labels are correct
    expected_labels = ["A", "B"]
    assert set(ideal_df["Label"].unique()) == set(expected_labels), (
        f"The labels should be {expected_labels}, but got " f"{ideal_df['Label'].unique()}."
    )

    # Test 4: Check if the means of the generated data are approximately correct
    generated_means = ideal_df.groupby("Label")["Feature"].mean().values
    # Check if the means are approximately shifted by the mean_shift_factor
    shift = max(generated_means) - min(generated_means)
    assert shift > 0, (
        f"The means should be shifted, but got {generated_means}."
    )
