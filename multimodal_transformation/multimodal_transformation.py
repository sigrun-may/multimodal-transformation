# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal
from itertools import combinations
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from statsmodels.distributions import ECDF

from multimodal_transformation.ideal_distribution_generator import simulate_multimodal_distribution


def transform_data(
    original_source_data_df: pd.DataFrame,
    ideal_target_data_df: pd.DataFrame,
    feature_name: str,
    interpolation_method: Literal["CubicSpline", "Linear"] = "CubicSpline",
    plot=False,
) -> pd.DataFrame:
    """Transform the source data to match the ideal target data distribution using the specified interpolation method.
    The function quantile transforms the source data to match the distribution of the target data.
    This is done by sorting the source and target data by the feature name, calculating the uniform quantiles,
    and then mapping the quantiles of the source data to the target data.
    The mapped values are then returned as a new DataFrame.
    Args:
        original_source_data_df: The source data to transform.
        ideal_target_data_df: The target data with the desired distribution.
        feature_name: The name of the feature to transform.
        interpolation_method: The interpolation method to use. Can be "Linear" or "CubicSpline". Default is CubicSpline
            using the PchipInterpolator.  PCHIP stands for Piecewise Cubic Hermite Interpolating Polynomial.
            The interpolant uses monotonic cubic splines to find the value of new points.
        plot: Whether to plot the mapped values. Default is False.
    Returns:
        The transformed data with the same distribution as the target data.
    """
    # save initial index
    original_source_data_df["original_index"] = original_source_data_df.index

    # sort the source and target data by the feature name
    original_source_sorted = original_source_data_df.sort_values(by=feature_name, ascending=True, inplace=False)
    target_sorted = ideal_target_data_df.sort_values(by=feature_name, ascending=True, inplace=False)

    # Compute the Empirical Cumulative Distribution Function (ECDF) for the source data
    # The ECDF represents the proportion of data points less than or equal to a given value.
    original_source_ecdf = ECDF(original_source_sorted[feature_name].values)

    if plot:
        plt.plot(original_source_ecdf.x, original_source_ecdf.y)  # Plot the ECDF curve
        plt.title("ECDF of Source Data")
        plt.show()

    # Map the source data values to their corresponding percentiles using the ECDF
    # Percentiles represent the relative position of each value in the distribution.
    original_source_percentiles = original_source_ecdf(original_source_sorted[feature_name].values)

    # Compute the ECDF for the target data
    # This will be used to map the source percentiles to the target distribution.
    ideal_target_ecdf = ECDF(target_sorted[feature_name].values)

    # If plotting is enabled, visualize the ECDF of the target data
    if plot:
        plt.plot(ideal_target_ecdf.x, ideal_target_ecdf.y)
        plt.title("ECDF of Target Data")
        plt.show()

    # Map the target data values to their corresponding percentiles using the ECDF
    ideal_target_percentiles = ideal_target_ecdf(target_sorted[feature_name].values)

    if interpolation_method == "CubicSpline":
        # Create a cubic spline interpolation function for the target data
        # The spline maps target percentiles to the corresponding feature values in the target distribution.
        # The `extrapolate=True` parameter ensures that the spline can handle values outside the given range.

        # PCHIP ist ein spezielles Verfahren zur stückweisen kubischen Interpolation, das in der Regel
        # monotonieerhaltend (bzw. „shape-preserving“) wirkt. „Monotonieerhaltend“ bedeutet hier, dass
        # wenn die gegebene Datenreihe zwischen zwei Punkten steigt oder fällt, die interpolierte Kurve
        # ebenfalls in diesem Bereich keine künstlichen Oszillationen oder Überschwingungen hervorruft.
        ideal_target_spline = PchipInterpolator(
            ideal_target_percentiles, target_sorted[feature_name].values, extrapolate=True
        )

        # plot ideal_target_spline
        if plot:
            plt.plot(ideal_target_percentiles, target_sorted[feature_name].values)
            plt.title("Ideal Target Spline")
            plt.show()

        # Transform the source data by mapping its percentiles to the target distribution
        # This is done by passing the source percentiles through the target spline.
        transformed_values = ideal_target_spline(original_source_percentiles)

        # If plotting is enabled, visualize the spline function
        if plot:
            plt.plot(ideal_target_percentiles, target_sorted[feature_name].values)
            plt.title("Ideal Target Spline")
            plt.show()

    elif interpolation_method == "Linear":
        # Create a linear transformation function for the target data
        # This function maps the target percentiles to the corresponding feature values in the target distribution.
        # The `extrapolate=True` parameter ensures that the linear function can handle values outside the given range.
        ideal_target_linear = interp1d(
            ideal_target_percentiles, target_sorted[feature_name].values, kind="linear", fill_value="extrapolate"
        )
        # Transform the source data by mapping its percentiles to the target distribution
        # This is done by passing the source percentiles through the target linear function.
        transformed_values = ideal_target_linear(original_source_percentiles)

        # If plotting is enabled, visualize the linear function
        if plot:
            plt.plot(ideal_target_percentiles, target_sorted[feature_name].values)
            plt.title("Ideal Target Linear")
            plt.show()
    else:
        raise ValueError("Invalid interpolation method. Choose 'CubicSpline' or 'Linear'.")

    # map the transformed_values to the initial orginal index from source_data_df
    transformed_series = pd.Series(transformed_values, index=original_source_sorted["original_index"]).sort_index(
        ascending=True
    )
    # drop the original index column
    original_source_data_df.drop(columns=["original_index"], inplace=True)
    original_source_data_df[feature_name] = transformed_series

    # seaborn histogram plot of feature_name of source_data_df
    if plot:
        sns.histplot(
            original_source_data_df,
            x=feature_name,
            hue="Label",
            multiple="layer",
            bins=30,
            kde=True,
            alpha=0.6,
        )
        plt.title(f"Transformed {feature_name} Values")
        plt.show()

    return original_source_data_df


# def quantile_transform(
#     source_data_df: pd.DataFrame,
#     target_data_df: pd.DataFrame,
#     feature_name: str,
#     plot=False,
# ) -> pd.DataFrame:
#     """Quantile transform the source data to match an ideal bimodal data distribution.
#
#     The function quantile transforms the source data to match the distribution of the target data.
#     This is done by sorting the source and target data by the feature name, calculating the uniform quantiles,
#     and then mapping the quantiles of the source data to the target data.
#     The mapped values are then returned as a new DataFrame.
#
#     Args:
#         source_data_df: The source data to transform.
#         target_data_df: The target data with the desired distribution.
#         feature_name: The name of the feature to transform.
#         plot: Whether to plot the mapped values. Default is False.
#
#     Returns:
#         The transformed data with the same distribution as the target data.
#     """
#     source_sorted = source_data_df.sort_values(by=feature_name, inplace=False).reset_index(inplace=False)
#     target_sorted = target_data_df.sort_values(by=feature_name, inplace=False).reset_index(drop=True, inplace=False)
#
#     # define uniform quantiles
#     quantiles = np.linspace(0, 1, len(source_sorted[feature_name]))
#
#     # Perform the quantile transformation by mapping the quantiles of the source data source_sorted to the target data
#     # target_sorted. Interpolate the values of the target data at the quantiles of the source data.
#     # This effectively transforms the source data to have the same distribution as the target data.
#     mapped_values = np.interp(
#         quantiles,
#         np.linspace(0, 1, len(target_sorted[feature_name])),
#         target_sorted[feature_name],
#     )
#
#     if plot:
#         # plot mapped_values
#         plt.plot(quantiles, mapped_values)
#         plt.title("Mapped Values")
#         plt.show()
#
#     # return pd.DataFrame({feature_name: mapped_values, "Label": source_sorted["Label"]})
#
#     transformed_df = pd.DataFrame({feature_name: mapped_values, "Label": source_sorted["Label"]})
#     transformed_df.set_index(source_sorted["index"], inplace=True)
#     transformed_df.sort_index(inplace=True)
#
#     return transformed_df


# def quantile_transform_ecdf(
#     source_data_df: pd.DataFrame,
#     target_data_df: pd.DataFrame,
#     feature_name="Feature",
#     plot=False,
# ) -> pd.DataFrame:
#     """Quantile transform the source data to match the target data distribution using the ECDF.
#
#     The function quantile transforms the source data to match the distribution of the target data using the empirical
#     cumulative distribution function (ECDF).
#
#     Args:
#         source_data_df: The source data to transform.
#         target_data_df: The target data with the desired distribution.
#         feature_name: The name of the feature to transform.
#         plot: Whether to plot the ECDF of the source data and the inverse CDF of the target data. Default is False.
#
#     Returns:
#         pd.DataFrame: The transformed data with the same distribution as the target data.
#     """
#     original_data = source_data_df.copy()
#     original_data["original_index"] = original_data.index
#
#     source_values = original_data[feature_name].values
#     target_values = target_data_df[feature_name].values
#
#     # Create ECDF from source and inverse CDF from target
#     ecdf_source = ECDF(source_values)
#     percentiles = ecdf_source(source_values)
#
#     target_sorted = np.sort(target_values)
#     n_target = len(target_sorted)
#
#     # generate uniformly distributed probabilities between 1/n_target and 1
#     target_probabilities = np.linspace(1 / n_target, 1, n_target)
#
#     # Use CubicSpline to create a smooth interpolation of the target values based on the probabilities
#     inverse_target_cdf = CubicSpline(target_probabilities, target_sorted)
#     transformed_values = inverse_target_cdf(percentiles)
#
#     result = original_data.copy()
#     result[feature_name] = transformed_values
#     result = result.sort_values(by="original_index").drop(columns="original_index")
#
#     if plot:
#         # plot ecdf
#         plt.plot(ecdf_source.x, ecdf_source.y)
#         plt.title("ECDF of Source Data")
#         plt.show()
#
#         # plot inverse_target_cdf
#         plt.plot(target_probabilities, target_sorted)
#         plt.title("Inverse Target CDF")
#         plt.show()
#
#         # plot histogram of result
#         sns.histplot(
#             result,
#             x=feature_name,
#             hue="Label",
#             multiple="layer",
#             bins=30,
#             kde=True,
#             alpha=0.6,
#         )
#         plt.title("Transformed Values")
#         plt.show()
#
#     return result[[feature_name, "Label"]]


# def calculate_second_mean_by_cohens_d(cohens_d_value, given_mean1, std_dev1, std_dev2):
#     """Calculate the first mean given Cohen's d, the second mean, and the standard deviation.
#
#     Args:
#         cohens_d_value (float): Cohen's d effect size.
#         given_mean1 (float): The mean of the first class.
#         std_dev1 (float): The standard deviation of the first class.
#         std_dev2 (float): The standard deviation of the second class.
#
#     Returns:
#         float: The mean of the first group.
#     """
#     pooled_std = np.sqrt((std_dev1**2 + std_dev2**2) / 2)
#     mean_difference = cohens_d_value * pooled_std
#
#     calculated_mean2 = mean_difference + given_mean1
#
#     return calculated_mean2
#
#     # calculate the effect size using Cohen's d


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size for two groups.

    Args:
        group1 (array): The data for group 1.
        group2 (array): The data for group 2.

    Returns:
        float: The effect size as Cohen's d.
    """
    # calculate the difference of the means of the samples
    mean_diff = np.mean(group1) - np.mean(group2)
    # calculate the pooled standard deviation
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)

    effect_size = mean_diff / pooled_std

    # calculate the second mean given Cohen's d, the first mean, and the standard deviation

    # calculate the absolute effect size
    return effect_size


def robust_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate the robust Cohen's d effect size for two groups.

    The robust Cohen's d effect size is calculated as the absolute difference of the medians of the two groups divided
    by the median absolute deviation (MAD) of the combined data.

    Args:
        group1 (array): The data for group 1.
        group2 (array): The data for group 2.

    Returns:
        float: The robust effect size as Cohen's d.
    """
    # calculate the difference of the medians of the samples
    median_diff = np.median(group1) - np.median(group2)

    if len(group1) == len(group2):
        # calculate the median absolute deviation (MAD) for balanced groups
        mad = np.median(np.abs(group1 - np.median(group1)) + np.abs(group2 - np.median(group2)))
    else:
        # calculate the median absolute deviation (MAD) for unbalanced groups
        mad = np.median(np.abs(np.concatenate([group1, group2]) - np.median(np.concatenate([group1, group2]))))
    # calculate the absolute effect size
    return median_diff / mad


def cliffs_delta(group1: np.ndarray, group2: np.ndarray)-> float:
    """Calculate Cliff's Delta effect size between two 1D arrays.
    Cliff's Delta is a non-parametric effect size measure that quantifies the degree of overlap between two distributions.
    It is defined as the difference between the proportion of pairs (x, y) such that x > y and the proportion of pairs
    (x, y) such that x < y, divided by the total number of pairs.
    Args:
        group1: The first data set.
        group2: The second data set.
    Returns:
        The effect size as Cliff's Delta.
    """
    n_group1 = len(group1)
    n_group2 = len(group2)
    bigger = 0
    smaller = 0

    for x in group1:
        for y in group2:
            if x > y:
                bigger += 1
            elif x < y:
                smaller += 1
    return (bigger - smaller) / (n_group1 * n_group2)


# def cliffs_delta(x, y):
#     """
#     Compute Cliff's Delta effect size between two 1D arrays.
#
#     Parameters:
#     - x, y: Lists or numpy arrays of numeric data
#
#     Returns:
#     - delta: float, Cliff's Delta value between -1 and 1
#     - interpretation: str, qualitative interpretation of the effect size
#     """
#     x = np.asarray(x)
#     y = np.asarray(y)
#     n_x = len(x)
#     n_y = len(y)
#
#     more = sum(xi > yj for xi in x for yj in y)
#     less = sum(xi < yj for xi in x for yj in y)
#
#     delta = (more - less) / (n_x * n_y)

    # # Interpretation guidelines (Romano et al., 2006)
    # abs_delta = abs(delta)
    # if abs_delta < 0.147:
    #     interpretation = "negligible"
    # elif abs_delta < 0.33:
    #     interpretation = "small"
    # elif abs_delta < 0.474:
    #     interpretation = "medium"
    # else:
    #     interpretation = "large"
    #
    # return delta, interpretation


# def cliffs_delta(values_class1, values_class2):
#     pass


def transform_data_set(
    data_df: pd.DataFrame,
    interpolation_method: Literal["CubicSpline", "Linear"] = "CubicSpline",
    target_overlap=0.1,
    max_shift=10,
    mean_shift_factor=1,
    std_list=None,
    scale_std=1,
    cliffs_delta_threshold=0.5,
    plot=False,
    inplace=True,
) -> pd.DataFrame:
    """Main function to run the multimodal transformation.

    Args:
        data_df: The original data to transform. Must contain a 'Label' column.
        interpolation_method: The interpolation method to use. Default is PchipInterpolator.
            PCHIP stands for Piecewise Cubic Hermite Interpolating Polynomial. The interpolant uses
            monotonic cubic splines to find the value of new points.
        target_overlap: The target overlap for the bimodal distribution. Default is 0.0001.
        max_shift: The maximum shift for the bimodal distribution. Default is 10.
        mean_shift_factor: The mean shift factor for the bimodal distribution. Default is 1.
        std_list: The list of standard deviations for the bimodal distribution. Default is None.
        scale_std: The scale factor for the standard deviation of the bimodal distribution. Default is 1.
        cliffs_delta_threshold: The threshold for the cliffs delta effect size. Default is 0.5.
        inplace: Whether to modify the original data_df or return a new DataFrame. Default is True.
        plot: Whether to plot the original and transformed data. Default is False.
    Returns:
        The transformed data with an optimized distribution.
    """
    # check if the data_df has a 'Label' column
    if "Label" not in data_df.columns:
        raise ValueError("The data_df must have a 'Label' column.")
    # check if the data_df has more than one class
    if data_df["Label"].nunique() < 2:
        raise ValueError("The data_df must have more than one class.")
    # check if the data_df has more than one feature
    if data_df.shape[1] < 2:
        raise ValueError("The data_df must have more than one feature.")

    data_df_cp = data_df.copy()

    scaler = MinMaxScaler()
    # data_df[data_df.columns[data_df.columns != "Label"]] = scaler.fit_transform(data_df.drop("Label", axis=1, inplace=False))
    # assert "Label" in data_df.columns, "The data_df must have a 'Label' column."

    # grouped_medians_df = data_df.groupby("Label").median(numeric_only=True)
    # print(grouped_medians_df)

    for feature in data_df.drop(columns=["Label"], inplace=False).columns:
        robust_cohens_d_list = []
        transformed_robust_cohens_d_list = []
        cliffs_delta_list = []

        # # sort grouped_medians_df by feature
        # grouped_medians_df.sort_values(by=feature, ascending=True, inplace=True)

        # current_original_feature_df = data_df[["Label", feature]].copy()

        # # reindex current_original_feature_df to match the index of grouped_medians_df
        # current_original_feature_df = current_original_feature_df.set_index("Label")

        # for label_index in len(grouped_medians_df.index-1):
        #     # calculate cliffs delta for each label_index and label_index + 1
        #     cliffs_delta_= cliffs_delta(
        #         data_df[data_df["Label"] == grouped_medians_df.index[label_index]][feature].values,
        #         data_df[data_df["Label"] == grouped_medians_df.index[label_index + 1]][feature].values,
        #     )

        # calculate the effect size for each combination of labels
        for class_tuple in combinations(data_df["Label"].unique(), 2):
            values_class1 = data_df[data_df["Label"] == class_tuple[0]][feature].values
            values_class2 = data_df[data_df["Label"] == class_tuple[1]][feature].values
            #print(f"Class {class_tuple[0]} and {class_tuple[1]}:")
            #print(f"Cliffs Delta: {cliffs_delta(values_class1, values_class2):.4f}")
            cliffs_delta_list.append(cliffs_delta(values_class1, values_class2))
            #print(cohens_d(values_class1, values_class2))
            #print(robust_cohens_d(values_class1, values_class2))
            robust_cohens_d_list.append(robust_cohens_d(values_class1, values_class2))

        if np.max(np.abs(cliffs_delta_list)) > cliffs_delta_threshold and np.max(np.abs(robust_cohens_d_list)) > cliffs_delta_threshold:
            print(f"Feature {feature} is multimodal with max cliff: {max(np.abs(cliffs_delta_list)):.4f} and max robutst cohens d: {max(np.abs(robust_cohens_d_list)):.4f}")
            # transform the data
            current_original_feature_df = data_df[["Label", feature]].copy()

            # create an ideal multimodal distribution
            ideal_target_df = simulate_multimodal_distribution(
                current_original_feature_df,
                feature_name=feature,
                defined_std=std_list,
                scale_std=scale_std,
                target_overlap=target_overlap,
                max_shift=max_shift,
                mean_shift_factor=mean_shift_factor,
                plot=plot,
            )

            transformed_data_df = transform_data(current_original_feature_df, ideal_target_df, feature_name=feature, interpolation_method=interpolation_method, plot=plot)

            # check if transformed_data_df is transformed in comparison to the original data_df
            assert not data_df_cp[feature].equals(
                transformed_data_df[feature]
            ), f"The transformed data is equal to the original data for feature {feature}."

            for class_tuple in combinations(data_df["Label"].unique(), 2):
                values_class1 = transformed_data_df[transformed_data_df["Label"] == class_tuple[0]][feature].values
                values_class2 = transformed_data_df[transformed_data_df["Label"] == class_tuple[1]][feature].values
                transformed_robust_cohens_d_list.append(robust_cohens_d(values_class1, values_class2))

            for original, transformed in zip(robust_cohens_d_list, transformed_robust_cohens_d_list):
                if abs(original) < abs(transformed):
                    print(f"Feature {feature} is transformed")
                    data_df.loc[:, feature] = transformed_data_df[feature]

                    # plot with plotly
                    # Plot the original next to the transformed feature using Plotly
                    # fig = px.histogram(
                    #     data_df_cp,
                    #     x=feature,
                    #     color="Label",
                    #     nbins=30,
                    #     title=f"Original {feature}",
                    #     opacity=0.6,
                    #     marginal="kde",
                    # )
                    # fig.show()
                    #
                    # fig = px.histogram(
                    #     data_df,
                    #     x=feature,
                    #     color="Label",
                    #     nbins=30,
                    #     title=f"Transformed {feature}",
                    #     opacity=0.6,
                    #     marginal="kde",
                    # )
                    # fig.show()

                    # plot the original next to the transformed feature
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    sns.histplot(data_df_cp, x=feature, hue="Label", bins=30, kde=True)
                    plt.title(
                        f"Original {feature}\n cliff {cliffs_delta_list}"
                    )
                    plt.subplot(1, 2, 2)
                    sns.histplot(data_df, x=feature, hue="Label", bins=30, kde=True)
                    plt.title(
                        f"Transformed {feature} \n {robust_cohens_d_list} \n{transformed_robust_cohens_d_list}"
                    )
                    plt.show()
                    break
                else:
                    data_df.loc[:, feature] = scaler.fit_transform(data_df[feature].values.reshape(-1, 1))
        else:
            data_df.loc[:, feature] = scaler.fit_transform(data_df[feature].values.reshape(-1, 1))

    #         transformed_cliffs_delta, _ = cliffs_delta(
    #             x=transformed_data_df[transformed_data_df["Label"] == transformed_data_df["Label"].unique()[0]][
    #                 feature
    #             ].values,
    #             y=transformed_data_df[transformed_data_df["Label"] == transformed_data_df["Label"].unique()[1]][
    #                 feature
    #             ].values,
    #         )
    #         # # check if the cohens_d and robust_cohens_d values increase for the transformed data in comparison
    #         # # to the original data
    #         transformed_cohens_d = cohens_d(
    #             transformed_data_df[data_df["Label"] == transformed_data_df["Label"].unique()[0]][feature].values,
    #             transformed_data_df[data_df["Label"] == transformed_data_df["Label"].unique()[1]][feature].values,
    #         )
    #         robust_transformed_cohens_d = robust_cohens_d(
    #             transformed_data_df[transformed_data_df["Label"] == transformed_data_df["Label"].unique()[0]][
    #                 feature
    #             ].values,
    #             transformed_data_df[transformed_data_df["Label"] == transformed_data_df["Label"].unique()[1]][
    #                 feature
    #             ].values,
    #         )
    #
    #         # check if the effect size increases
    #         if abs(original_cohens_d) < abs(transformed_cohens_d):
    #
    #             # min max scale feature
    #             # data_df[feature] = scaler.fit_transform(transformed_data_df.drop(columns=["Label"], inplace=False))
    #             data_df[feature] = transformed_data_df[feature]
    #
    #             # plot the original next to the transformed feature
    #             plt.figure(figsize=(10, 5))
    #             plt.subplot(1, 2, 1)
    #             sns.histplot(data_df_cp, x=feature, hue="Label", bins=30, kde=True)
    #             plt.title(
    #                 f"Original {feature} \n d {original_cohens_d:.4f} and robust d {orginal_robust_cohens_d} \n cliff {cliffs_d:.4f}"
    #             )
    #             plt.subplot(1, 2, 2)
    #             sns.histplot(data_df, x=feature, hue="Label", bins=30, kde=True)
    #             plt.title(
    #                 f"Transformed {feature} \n d {transformed_cohens_d:.4f} and robust d  {robust_transformed_cohens_d}"
    #             )
    #             plt.show()
    #     else:
    #         data_df[feature] = scaler.fit_transform(data_df.drop(columns=["Label"], inplace=False))
    #
    # # plot histogram of list of Cohen's d values
    # plt.figure(figsize=(10, 5))
    # sns.histplot(original_cohens_d_list, bins=30, kde=True, color="r")
    # sns.histplot(cliffs_delta_list, bins=30, kde=True, color="b")
    # plt.title("Histogram of Original Cohen's d Values")
    # plt.xlabel("Cohen's d")
    # plt.ylabel("Frequency")
    # plt.show()
    return data_df
