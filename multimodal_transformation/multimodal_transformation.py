# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import interp1d, CubicSpline
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns

def create_ideal_artificial_bimodal_distribution(original_data_df: pd.DataFrame, feature_name: str,
                                                 modes:list=None, sigmas: list=None, scale_factor: int=100, plot=False)->pd.DataFrame:
    """Create an ideal artificial bimodal distribution.

    The function creates an ideal artificial bimodal distribution by combining two normal distributions with the given
    parameters. The number of samples in each class is determined by the number of samples in the original data and a
    scale factor. The scale factor determines the number of samples to generate for each class within the ideal bimodal
    distribution. A higher scale will result in a smoother distribution.

    Args:
        original_data_df: The original data. The number of samples in each class is used to generate the bimodal
            distribution. The original data must have a 'Label' column with the class labels.
        feature_name: The name of the column in the original data_df that contains the source to create the ideal
            bimodal distribution.
        modes: The list of means of the normal distributions.
        sigmas: The list of standard deviations of the normal distributions.
        scale_factor: Scale factor to determine the number of samples to generate for each class within the ideal
            bimodal distribution. A higher scale will result in a smoother distribution. Default is 100.
        plot: Whether to plot the ideal target distribution. Default is False.

    Returns:
        A DataFrame containing the artificial bimodal distribution with the specified parameters.
    """
    # check if either mode_list or sigma_list are None but not both
    if modes is None and sigmas is not None:
        raise ValueError("Both list of modes and list of sigmas must be provided.")

    if modes is not None and sigmas is None:
        raise ValueError("Both list of modes and list of sigmas must be provided.")

    # calculate median and get sample size of the original classes
    original_median_sample_size_df = pd.DataFrame(index=original_data_df['Label'].unique())
    for label in original_data_df['Label'].unique():
        class_median = np.median(original_data_df[original_data_df['Label'] == label][feature_name])
        original_median_sample_size_df.loc[label, ["median"]] = class_median
        class_size = len(original_data_df[original_data_df['Label'] == label][feature_name])
        original_median_sample_size_df.loc[label, ["sample_size"]] = class_size

    # sorted_original_data_median_series = original_median_sample_size_df['median'].sort_values(ascending=True, inplace=False)
    # original_median_sample_size_df = original_median_sample_size_df.reindex(sorted_original_data_median_series.index)

    # reindex the df by the sorted original data median series
    original_median_sample_size_df.sort_values(by="median", inplace=True, ascending=True)

    # create a DataFrame to store the parameters for the artificial distribution
    artificial_parameters_df = pd.DataFrame(index=original_median_sample_size_df.index)
    artificial_parameters_df["sample_size"] = original_median_sample_size_df["sample_size"] * scale_factor

    # check if mode_list and sigma_list are not None
    if modes is not None and sigmas is not None:
        assert len(modes) == len(sigmas), "The number of modes and sigmas must be the same."
        assert len(modes) == len(set(modes)), "The means of the normal distributions must be different."
        assert original_data_df['Label'].nunique() == len(modes), "The number of modes and classes must be the same."

        # sort modes ascending and keep the corresponding sigmas in the same order and insert them into the artificial_parameters_df
        mode_sigma_df = pd.DataFrame({'mean': modes, 'sigma': sigmas})
        mode_sigma_df.sort_values(by='mean', inplace=True)
        artificial_parameters_df["mean"] = mode_sigma_df["mean"].values
        artificial_parameters_df["sigma"] = mode_sigma_df["sigma"].values

    else:
        sigmas = np.ones(len(original_median_sample_size_df))
        # calculate the mode_list

        # use the median of the original data as blueprint to generate a mode list where the distributions will not
        # overlap for the given sigma by changing each mode by 3*sigma to the next mode as minimum distance
        modes = [original_median_sample_size_df["median"].values[0]]
        # for i, mode in enumerate(original_median_sample_size_df["median"].values[1:]):
        #     if mode < modes[i] + 8 * sigmas[i]:
        #         modes.append(modes[i] + 8 * sigmas[i])
        #     else:
        #         modes.append(mode)

        for i, mode in enumerate(original_median_sample_size_df["median"].values[1:]):
            # calculate artificial mean by defining a Cohen's d value
            defined_cohens_d = 10
            std = sigmas[i]
            previous_mean = modes[i]
            artificial_mean = defined_cohens_d * std + previous_mean
            if artificial_mean < mode:
                modes.append(mode)
            else:
                modes.append(artificial_mean)

        artificial_parameters_df["mean"] = modes
        artificial_parameters_df["sigma"] = sigmas

    # create the ideal bimodal distribution
    # numpy random generator
    np.random.seed(42)
    rng = np.random.default_rng()

    ideal_target_df = pd.DataFrame()
    ideal_classes = []
    corresponding_ideal_class_labels = []
    for index, row in artificial_parameters_df.iterrows():
        ideal_class = rng.normal(loc=row['mean'], scale=row['sigma'], size=int(row['sample_size']))
        ideal_classes.append(ideal_class)
        label_class = [index] * int(row['sample_size'])
        corresponding_ideal_class_labels.append(label_class)
    assert len (np.concatenate(corresponding_ideal_class_labels)) == len(np.concatenate(ideal_classes)), "The number of labels and samples must be the same."

    # create a DataFrame with the ideal multimodal distribution with the labels in the 'Label' column as first column
    ideal_target_df = pd.DataFrame({'Label': np.concatenate(corresponding_ideal_class_labels), feature_name: np.concatenate(ideal_classes)})

    if plot:
        plt.figure(figsize=(5, 5))
        plt.title("Ideal Target Distribution")
        sns.histplot(ideal_target_df, x=feature_name, hue='Label', multiple='layer', bins=200, kde=True, alpha=0.6)
        plt.show()

    original_median_sample_size_df["artificial_mean_sigma"] = sorted(zip(modes, sigmas))

    # sort mode and sigma list
    sorted_mode_sigma_list = sorted(zip(modes, sigmas))

    # insert sorted mean list
    artificial_parameters_df["mean"] = [mean_sigma[0] for mean_sigma in sorted_mode_sigma_list]

    # insert sorted sigma list
    artificial_parameters_df["sigma"] = [mean_sigma[1] for mean_sigma in sorted_mode_sigma_list]
    return ideal_target_df


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


def quantile_transform(source_data_df: pd.DataFrame, target_data_df:pd.DataFrame, feature_name:str)->pd.DataFrame:
    """Quantile transform the source data to match an ideal bimodal data distribution.

    The function quantile transforms the source data to match the distribution of the target data.
    This is done by sorting the source and target data by the feature name, calculating the uniform quantiles,
    and then mapping the quantiles of the source data to the target data.
    The mapped values are then returned as a new DataFrame.

    Args:
        source_data_df: The source data to transform.
        target_data_df: The target data with the desired distribution.
        feature_name: The name of the feature to transform.

    Returns:
        The transformed data with the same distribution as the target data.
    """
    source_sorted = source_data_df.sort_values(by=feature_name).reset_index(drop=True)
    target_sorted = target_data_df.sort_values(by=feature_name).reset_index(drop=True)

    # define uniform quantiles
    quantiles = np.linspace(0, 1, len(source_sorted[feature_name]))

    # Perform the quantile transformation by mapping the quantiles of the source data source_sorted to the target data
    # target_sorted. Interpolate the values of the target data at the quantiles of the source data.
    # This effectively transforms the source data to have the same distribution as the target data.
    mapped_values = np.interp(quantiles, np.linspace(0, 1, len(target_sorted[feature_name])), target_sorted[feature_name])

    # plot mapped_values
    plt.plot(quantiles, mapped_values)
    plt.title('Mapped Values')
    plt.show()

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

    # plot ecdf
    plt.plot(ecdf_source.x, ecdf_source.y)

    # set title
    plt.title('ECDF of Source Data')
    plt.show()

    percentiles = ecdf_source(source_values)

    target_sorted = np.sort(target_values)
    n_target = len(target_sorted)
    target_probs = np.linspace(1/n_target, 1, n_target)

    # inverse_target_cdf = interp1d(target_probs, target_sorted, bounds_error=False,
    #                             fill_value=(target_sorted[0], target_sorted[-1]))
    inverse_target_cdf = CubicSpline(target_probs, target_sorted)

    # plot inverse_target_cdf
    plt.plot(target_probs, target_sorted)
    plt.title('Inverse Target CDF')
    plt.show()

    transformed_values = inverse_target_cdf(percentiles)

    result = source_data.copy()
    result['Feature'] = transformed_values
    result = result.sort_values(by='original_index').drop(columns='original_index')

    # plot histogramm of result
    sns.histplot(result, x='Feature', hue='Label', multiple='layer', bins=30, kde=True, alpha=0.6)
    plt.title('Transformed Values')
    plt.show()

    return result[['Feature', 'Label']]


def calculate_second_mean_by_cohens_d(cohens_d_value, given_mean, std_dev):
    """
    Calculate the first mean given Cohen's d, the second mean, and the standard deviation.

    Args:
        cohens_d_value (float): Cohen's d effect size.
        given_mean (float): The mean of the second group.
        std_dev (float): The standard deviation.

    Returns:
        float: The mean of the first group.
    """
    return cohens_d_value * std_dev + given_mean


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

    if len(group1) == len(group2):
        # calculate the median absolute deviation (MAD) for balanced groups
        mad = np.median(np.abs(group1 - np.median(group1)) + np.abs(group2 - np.median(group2)))
    else:
        # calculate the median absolute deviation (MAD) for unbalanced groups
        mad = np.median(np.abs(np.concatenate([group1, group2]) - np.median(np.concatenate([group1, group2]))))
    # calculate the absolute effect size
    return np.abs(median_diff / mad)


