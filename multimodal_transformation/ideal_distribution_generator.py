# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import seaborn as sns


def overlap_normal(mean1, std1, mean2, std2, n_points=1000, n_std=6):
    """Calculate the overlap between two normal distributions.
    Args:
        mean1 : Mean of the first normal distribution.
        std1 : Standard deviation of the first normal distribution.
        mean2 : Mean of the second normal distribution.
        std2 : Standard deviation of the second normal distribution.
        n_points : Number of points to compute the overlap.
        n_std : Number of standard deviations to cover in the range.
    Returns:
        Overlap between the two normal distributions.
    """
    # Generate a range of x values that covers both distributions
    # The range is determined by the means and standard deviations of the distributions
    x = np.linspace(
        min(mean1 - n_std * std1, mean2 - n_std * std2),  # Lower bound
        max(mean1 + n_std * std1, mean2 + n_std * std2),  # Upper bound
        n_points,  # Number of points in the range
    )

    # Compute the probability density function (PDF) for the first normal distribution
    pdf1 = norm.pdf(x, mean1, std1)

    # Compute the probability density function (PDF) for the second normal distribution
    pdf2 = norm.pdf(x, mean2, std2)

    # Calculate the overlap by integrating the minimum of the two PDFs
    # This represents the area under the curve where the two distributions overlap
    return np.trapz(np.minimum(pdf1, pdf2), x)


def find_minimum_shift_of_mean(
    std1: float,
    std2: float,
    target_overlap: float = 0.1,
    max_shift: float = 10.0,
    min_shift: float = None,
    tol: float = 1e-4,
):
    """Find the minimum shift of the mean of a normal distribution to achieve a target overlap with another normal distribution.
    Args:
        std1 : Standard deviation of the first normal distribution.
        std2 : Standard deviation of the second normal distribution.
        target_overlap : Target overlap between the two normal distributions.
        max_shift : Maximum shift of the mean of the first normal distribution.
        min_shift : Minimum shift of the mean of the first normal distribution.
        tol : Tolerance for the binary search.
    Returns:
        Minimum shift of the mean of the first normal distribution to achieve the target overlap.
    """
    # Initialize the lower and upper bounds for the binary search
    low, high = min_shift, max_shift

    # Perform binary search until the difference between high and low is within the tolerance
    while (high - low) > tol:
        # Calculate the midpoint between the current bounds
        mid = (low + high) / 2

        # Compute the overlap of two normal distributions with shifted means
        ov = overlap_normal(-mid / 2, std1, +mid / 2, std2)

        # If the overlap is less than the target, reduce the upper bound
        if ov < target_overlap:
            high = mid
        # Otherwise, increase the lower bound
        else:
            low = mid

    # Return the midpoint of the final bounds as the minimum shift
    return (low + high) / 2


def find_minimum_scale_factors_for_std(std1, std2, mean1, mean2, target_overlap=0.00000001, tol=1e-4):
    assert (mean2 - mean1) > tol
    # reduce std1 and std2 to achieve the target overlap
    scale_factor1 = scale_factor2 = 1.0
    ov = overlap_normal(mean1, std1 * scale_factor1, mean2, std2 * scale_factor2)
    while ov > target_overlap:
        # scale std1 and std2
        # decrease scale_factor1 and scale_factor2
        scale_factor1 -= 0.1
        scale_factor2 -= 0.1
        std1 *= scale_factor1
        std2 *= scale_factor2
        ov = overlap_normal(mean1, std1, mean2, std2 * scale_factor2)
    return std1, std2, scale_factor1, scale_factor2


def simulate_multimodal_distribution(
    original_df,
    feature_name,
    scale_std=1,
    defined_std=None,
    mean_shift_factor=1,
    max_shift=10,
    min_shift=0.1,
    target_overlap=1e-8,
    scale_factor=100,
    min_max_sclaling=True,
    plot=False,
):
    """
    Create an ideal artificial multimodal distribution.

    The function creates an ideal artificial multimodal distribution by combining normal distributions.
    The number of samples in each class is determined by the number of samples in the original data and a
    scale factor. The scale factor determines the number of samples to generate for each class within the ideal
    multimodal distribution. A higher scale will result in a smoother distribution.


    Args:
        original_df : Original DataFrame with features and labels.
        feature_name: Name of the feature to simulate.
        scale_std : Scale factor for standard deviation, by default 1.
        defined_std : List of standard deviations for each class, by default None.
        mean_shift_factor : Factor to shift means apart, by default 1.
        max_shift : Maximum shift for means, by default 10.
        min_shift : Minimum shift for means, by default 0.1.
        target_overlap : Target overlap for distributions, by default 1e-8.
        scale_factor : Factor to scale the sample size, by default 100.
        min_max_sclaling : Whether to apply min-max scaling, by default True.
        plot : Whether to plot the distributions, by default False. Should only be used for debugging.

    Returns:
        DataFrame with simulated ideal mulimodal distribution.
    """
    assert mean_shift_factor >= 1, "mean_shift_factor should be greater than or equal to 1"
    rng = np.random.RandomState(42)
    distribution_list = []
    labels = []
    parameters_df = pd.DataFrame(
        index=original_df["Label"].unique(),
        columns=["loc", "std", "min_value", "max_value", "original_std", "sample_size"],
    )

    if defined_std is not None:
        parameters_df["std"] = defined_std
    for class_label in parameters_df.index:
        class_values = original_df.loc[original_df["Label"] == class_label][feature_name].values
        parameters_df.loc[class_label, "loc"] = np.median(class_values)
        parameters_df.loc[class_label, "sample_size"] = len(class_values)
        parameters_df.loc[class_label, "min_value"] = np.min(class_values)
        parameters_df.loc[class_label, "max_value"] = np.max(class_values)
        parameters_df.loc[class_label, "original_std"] = np.std(class_values)
        if defined_std is None:
            parameters_df.loc[class_label, "std"] = np.std(class_values, ddof=1) * scale_std

    # order medians ascending
    parameters_df.sort_values(by=["loc"], ascending=True, inplace=True)

    # shift means
    distribution1 = rng.normal(
        loc=parameters_df["loc"].iloc[0],
        scale=parameters_df["std"].iloc[0],
        size=parameters_df["sample_size"].iloc[0] * scale_factor,
    )
    distribution_list.append(distribution1)
    for i in range(len(parameters_df.index) - 1):
        class_label = parameters_df.index[i]
        labels += [class_label] * parameters_df["sample_size"].iloc[i] * scale_factor

        # find the minimum shift that achieves the desired overlap but does not decrease the effect size
        delta = find_minimum_shift_of_mean(
            std1=distribution_list[i].std(),
            std2=parameters_df["std"].iloc[i + 1],
            # min_shift=median_diff,
            min_shift=min_shift,
            max_shift=max_shift,
            target_overlap=target_overlap,
        )
        next_ideal_mean = (distribution_list[i].mean() + delta) * mean_shift_factor
        next_distribution = rng.normal(
            loc=next_ideal_mean,
            scale=parameters_df["std"].iloc[i + 1],
            size=parameters_df["sample_size"].iloc[i + 1] * scale_factor,
        )

        # # find the scale factors for std1 and std2
        # std1, std2, scale_factor1, scale_factor2 = find_minimum_scale_factors_for_std(
        #     std1=distribution_list[i].std(),
        #     std2=parameters_df["std"].iloc[i+1],
        #     mean1=distribution_list[i].mean(),
        #     mean2=parameters_df["loc"].iloc[i + 1],
        #     target_overlap=target_overlap,
        #     tol=1e-4,
        # )
        # assert std1 > 0 and std2 > 0, f"std1{std1} and std2{std2} should be greater than 0"
        # next_distribution = rng.normal(
        #     loc=parameters_df["loc"].iloc[i + 1],
        #     scale=std2,
        #     size=parameters_df["sample_size"].iloc[i + 1] * scale_factor,
        # )
        distribution_list.append(next_distribution)

    assert len(distribution_list) == len(original_df["Label"].unique())
    labels += [parameters_df.index[-1]] * parameters_df["sample_size"].iloc[-1] * scale_factor
    for label_count, class_label in zip(parameters_df["sample_size"], parameters_df.index):
        assert label_count * scale_factor == labels.count(
            class_label
        ), f"label {label_count} should be {label_count * scale_factor} but got {labels.count(class_label)}"
    ideal_distribution_values = np.concatenate(distribution_list, axis=0)
    assert len(labels) == ideal_distribution_values.size
    ideal_distribution_df = pd.DataFrame({"Label": labels, feature_name: ideal_distribution_values})

    if min_max_sclaling:
        min_value = ideal_distribution_df[feature_name].min()
        max_value = ideal_distribution_df[feature_name].max()
        ideal_distribution_df[feature_name] = (ideal_distribution_df[feature_name] - min_value) / (
            max_value - min_value
        )

    if plot:
        plt.figure(figsize=(10, 5))
        sns.histplot(ideal_distribution_df, x=feature_name, hue="Label", bins=30, kde=True)
        plt.title(f"Simulated {feature_name}")
        plt.show()
    return ideal_distribution_df


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
#
# def iqr_to_std(iqr):
#     """Convert IQR to approximate standard deviation (for normal data)"""
#     return iqr / 1.349
#
# def pooled_std(s1, s2, n1, n2):
#     """Compute pooled standard deviation"""
#     return np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
#
# def hedges_g(x, y):
#     """Compute Hedges' g"""
#     n1, n2 = len(x), len(y)
#     s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
#     pooled = pooled_std(s1, s2, n1, n2)
#     d = (np.mean(x) - np.mean(y)) / pooled
#     correction = 1 - (3 / (4*(n1 + n2) - 9))
#     return d * correction
#
# def simulate_bimodal_from_median(x, y, scale_std=0.6, n_samples=None, plot=True):
#     """
#     Generate ideal bimodal distribution using medians, simulate data, and compute Hedges' g.
#
#     Parameters:
#     - x, y: original data arrays
#     - scale_std: factor to reduce spread (default = 0.6)
#     - n_samples: how many samples per group to simulate (default = min(len(x), len(y)))
#     - plot: whether to visualize the result
#
#     Returns:
#     - sim_x, sim_y: simulated distributions
#     - g: Hedges' g of the simulated distributions
#     """
#     if n_samples is None:
#         n_samples = min(len(x), len(y))
#
#     # Robust stats
#     median_x, median_y = np.median(x), np.median(y)
#     iqr_x = np.percentile(x, 75) - np.percentile(x, 25)
#     iqr_y = np.percentile(y, 75) - np.percentile(y, 25)
#     std_x = iqr_to_std(iqr_x)
#     std_y = iqr_to_std(iqr_y)
#
#     # Simulate ideal groups
#     sim_x = np.random.normal(loc=median_x, scale=std_x * scale_std, size=n_samples)
#     sim_y = np.random.normal(loc=median_y, scale=std_y * scale_std, size=n_samples)
#
#     # Compute Hedges' g
#     g = hedges_g(sim_x, sim_y)
#
#     if plot:
#         # Plotting
#         sim_data = np.concatenate([sim_x, sim_y])
#         x_range = np.linspace(min(sim_data) - 1, max(sim_data) + 1, 500)
#         plt.figure(figsize=(10, 6))
#         plt.hist(sim_data, bins=50, density=True, alpha=0.5, label="Simulated Bimodal")
#         plt.plot(x_range, norm.pdf(x_range, median_x, std_x * scale_std), label="Group 1 (sim)", lw=2)
#         plt.plot(x_range, norm.pdf(x_range, median_y, std_y * scale_std), label="Group 2 (sim)", lw=2)
#         plt.axvline(median_x, color='blue', linestyle='--', alpha=0.7)
#         plt.axvline(median_y, color='orange', linestyle='--', alpha=0.7)
#         plt.title(f"Ideal Bimodal Distribution (Hedges' g = {g:.2f})")
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#
#     return sim_x, sim_y, g
