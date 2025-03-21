# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import multimodal_transformation.multimodal_transformation as mt

# simulate a small bimodal dataset with outliers
np.random.seed(42)
data1 = np.random.normal(loc=-1, scale=1, size=50)
data2 = np.random.lognormal(size=50)
# original_data = np.hstack([data1, data2])
original_data = np.random.normal(loc=5, scale=1, size=100)

mt.count_peaks(original_data)

# Add labels to the original data (e.g., 'Data1' for the first half, 'Data2' for the last half)
labels = ['Data1'] * len(data1) + ['Data2'] * len(data2)
original_data_df = pd.DataFrame({'Feature': original_data, 'Label': labels})

# create ideal bimodal target distribution
ideal_target_df = pd.DataFrame({
    'Feature': np.hstack([np.random.normal(loc=-2, size=150), np.random.normal(loc=5, size=150)]),
    'Label': ['Label1'] * 150 + ['Label2'] * 150
})

methods = [mt.quantile_transform(original_data_df, ideal_target_df), mt.quantile_transform_ecdf(original_data_df, ideal_target_df) ]
for mapped_data_df in methods:
    # # map the original data to the ideal target distribution
    # mapped_data_df = mt.quantile_transform(original_data_df, ideal_target_df)
    # # mapped_data_df = mt.quantile_transform_ecdf(original_data_df, ideal_target_df)

    # visualize the original and mapped data
    plt.figure(figsize=(10, 5))

    # before: original distribution with outliers
    plt.subplot(1, 2, 1)
    # set the colors according to the labels
    sns.histplot(
        original_data_df,
        x='Feature',
        hue='Label',
        multiple='layer',
        bins=20,
        kde=True,
        alpha=0.6,
        # legend=True,
        palette={ 'Data1': 'blue', 'Data2': 'green' })
    sns.kdeplot(original_data_df['Feature'].values)
    plt.title("Original bimodal overlapping distribution with outliers")

    # after: transformed distribution with optimal mixture
    plt.subplot(1, 2, 2)
    # set the colors according to the labels
    sns.histplot(
        mapped_data_df,
        x='Feature',
        hue='Label',
        multiple='layer',
        bins=20,
        kde=True,
        alpha=0.6,
        #legend=True,
        palette={ 'Data1': 'blue', 'Data2': 'green' })
    sns.kdeplot(mapped_data_df['Feature'].values)
    plt.title("After ECDF-Mapping on Ideal Target Distribution")

    plt.tight_layout()
    plt.show()

    # Cohen’s d before and after the transformation
    orig_d = mt.cohens_d(data1, data2)
    # mapped data where label is 'Data1' and 'Data2'
    mapped_data_label1 = mapped_data_df[mapped_data_df['Label'] == 'Data1']['Feature'].values
    mapped_data_label2 = mapped_data_df[mapped_data_df['Label'] == 'Data2']['Feature'].values
    mapped_d = mt.cohens_d(mapped_data_label1, mapped_data_label2)

    print(f"Cohen’s d vorher: {orig_d:.3f}")
    print(f"Cohen’s d nachher: {mapped_d:.3f}")

    # calculate the effect size using robust Cohen's d
    orig_robust_d = mt.robust_cohens_d(data1, data2)
    mapped_robust_d = mt.robust_cohens_d(mapped_data_label1, mapped_data_label2)

    print(f"Robust Cohen’s d vorher: {orig_robust_d:.3f}")
    print(f"Robust Cohen’s d nachher: {mapped_robust_d:.3f}")


