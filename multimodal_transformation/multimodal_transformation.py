# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF


def count_peaks(data: np.ndarray) -> int:
    """Count the number of peaks in a distribution using a Gaussian KDE.

    The function estimates the probability density function of the data using a Gaussian kernel density estimator (KDE).
    It then identifies the peaks in the estimated probability density function by finding the points where the second
    derivative changes sign from positive to negative. This indicates a change from increasing to decreasing density,
    which corresponds to a peak in the distribution.
    The number of peaks is calculated by counting the number of points where the second derivative changes
    sign from positive to negative.

    Args:
        data: The data to analyze.

    Returns:
        int: The number of peaks in the distribution.
    """
    # calculate if the pdf probability density function has more than one peak
    kde = stats.gaussian_kde(data)

    # generate a range of values (x) by creating an array of 1000 evenly spaced values between the minimum and maximum of
    # the original_data
    x = np.linspace(min(data), max(data), 100000)
    # compute the estimated probability density function values at each point in x using the Gaussian KDE object kde
    y = kde(x)

    # Identify peaks in the estimated probability density function:
    # The peaks are identified by finding the points where the second derivative of the estimated probability density
    # function changes sign from positive to negative. This indicates a change from increasing to decreasing density,
    # which corresponds to a peak in the distribution. The number of peaks is then calculated by counting the number of
    # points where the second derivative changes sign.
    # np.diff(y) computes the discrete difference of the y values. np.sign(np.diff(y)) returns the sign of the
    # differences, indicating whether the function is increasing or decreasing. np.diff(np.sign(np.diff(y))) computes
    # the difference of the signs, identifying changes in direction (from increasing to decreasing or vice versa).
    # < 0 checks where the change in direction is from increasing to decreasing, which indicates a peak.
    # So, peaks = np.diff(np.sign(np.diff(y))) < 0 creates a boolean array where True values correspond to the indices
    # of the peaks in the y array.
    peaks = np.diff(np.sign(np.diff(y))) < 0
    n_peaks = sum(peaks)
    print(f"Number of peaks: {n_peaks}")
    return n_peaks


# map the original data to the ideal target distribution
def quantile_transform(source_data: pd.DataFrame, target_data:pd.DataFrame, feature_name='Feature')->pd.DataFrame:
    """Quantile transform the source data to match the target data distribution.

    The function quantile transforms the source data to match the distribution of the target data.
    This is done by sorting the source and target data by the feature name, calculating the uniform quantiles,
    and then mapping the quantiles of the source data to the target data.
    The mapped values are then returned as a new DataFrame.

    Args:
        source_data (pd.DataFrame): The source data to transform.
        target_data (pd.DataFrame): The target data with the desired distribution.
        feature_name (str): The name of the feature to transform.

    Returns:
        pd.DataFrame: The transformed data with the same distribution as the target data.
    """
    source_sorted = source_data.sort_values(by=feature_name).reset_index(drop=True)
    target_sorted = target_data.sort_values(by=feature_name).reset_index(drop=True)

    # define uniform quantiles
    quantiles = np.linspace(0, 1, len(source_sorted[feature_name]))

    # Perform the quantile transformation by mapping the quantiles of the source data source_sorted to the target data
    # target_sorted. Interpolate the values of the target data at the quantiles of the source data.
    # This effectively transforms the source data to have the same distribution as the target data.
    mapped_values = np.interp(quantiles, np.linspace(0, 1, len(target_sorted[feature_name])), target_sorted[feature_name])

    return pd.DataFrame({'Feature': mapped_values, 'Label': source_sorted['Label']})


def quantile_transform_ecdf(source_data: pd.DataFrame, target_data: pd.DataFrame, feature_name='Feature')->pd.DataFrame:
    """Quantile transform the source data to match the target data distribution using the ECDF.

    The function quantile transforms the source data to match the distribution of the target data using the empirical
    cumulative distribution function (ECDF).

    Args:
        source_data (pd.DataFrame): The source data to transform.
        target_data (pd.DataFrame): The target data with the desired distribution.
        feature_name (str): The name of the feature to transform.

    Returns:
        pd.DataFrame: The transformed data with the same distribution as the target data.
    """
    source_data = source_data.copy()
    source_data['original_index'] = source_data.index

    source_values = source_data[feature_name].values
    target_values = target_data[feature_name].values

    # Create ECDF from source and inverse CDF from target
    ecdf_source = ECDF(source_values)
    percentiles = ecdf_source(source_values)

    target_sorted = np.sort(target_values)
    n_target = len(target_sorted)
    target_probs = np.linspace(1/n_target, 1, n_target)

    inverse_target_cdf = interp1d(target_probs, target_sorted, bounds_error=False,
                                  fill_value=(target_sorted[0], target_sorted[-1]))

    transformed_values = inverse_target_cdf(percentiles)

    result = source_data.copy()
    result['Feature'] = transformed_values
    result = result.sort_values(by='original_index').drop(columns='original_index')

    return result[['Feature', 'Label']]


# calculate the effect size using Cohen's d
def cohens_d(group1:np.ndarray, group2: np.ndarray)->float:
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
    # calculate the absolute effect size
    return np.abs(mean_diff / pooled_std)


def robust_cohens_d(group1:np.ndarray, group2:np.ndarray)->float:
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
    # calculate the median absolute deviation (MAD)
    mad = np.median(np.abs(group1 - np.median(group1)) + np.abs(group2 - np.median(group2)))
    # calculate the absolute effect size
    return np.abs(median_diff / mad)


