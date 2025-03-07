import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# simulate a small bimodal dataset with outliers
np.random.seed(42)
data1 = np.random.normal(loc=-2, scale=2, size=15)  # Modus 1
data2 = np.random.lognormal(size=15) # Modus 2
original_data = np.hstack([data1, data2])  # Gesamtmenge: 30 Werte

# original_data = np.random.normal(loc=5, scale=1, size=30)

# Add labels to the original data (e.g., 'Data1' for the first 15, 'Data2' for the last 15)
labels = ['Data1'] * len(data1) + ['Data2'] * len(data2)
original_data_df = pd.DataFrame({'Feature': original_data, 'Label': labels})

# calculate if the pdf probability density function has more than one peak
kde = stats.gaussian_kde(original_data)
x = np.linspace(min(original_data), max(original_data), 1000)
y = kde(x)
peaks = np.diff(np.sign(np.diff(y))) < 0
n_peaks = sum(peaks)
print(f"Number of peaks: {n_peaks}")

# create ideal bimodal target distribution
ideal_target_df = pd.DataFrame({
    'Feature': np.hstack([np.random.normal(loc=-2, size=150), np.random.normal(loc=5, size=150)]),
    'Label': ['Label1'] * 150 + ['Label2'] * 150
})

# map the original data to the ideal target distribution
def quantile_transform(source_data: pd.DataFrame, target_data:pd.DataFrame, feature_name='Feature'):
    source_sorted = source_data.sort_values(by=feature_name).reset_index(drop=True)
    target_sorted = target_data.sort_values(by=feature_name).reset_index(drop=True)

    # define uniform quantiles
    quantiles = np.linspace(0, 1, len(source_sorted[feature_name]))

    # Perform the quantile transformation by mapping the quantiles of the source data source_sorted to the target data
    # target_sorted. Interpolate the values of the target data at the quantiles of the source data.
    # This effectively transforms the source data to have the same distribution as the target data.
    mapped_values = np.interp(quantiles, np.linspace(0, 1, len(target_sorted[feature_name])), target_sorted[feature_name])

    return pd.DataFrame({'Feature': mapped_values, 'Label': source_sorted['Label']})

# map the original data to the ideal target distribution
mapped_data = quantile_transform(original_data_df, ideal_target_df)

# visualize the original and mapped data
plt.figure(figsize=(10, 5))

# before: original distribution with outliers
plt.subplot(1, 2, 1)
# set the colors according to the labels
sns.histplot(
    original_data_df,
    x='Feature',
    hue='Label',
    multiple='stack',
    bins=20,
    kde=False,
    # alpha=0.6,
    # legend=True,
    palette={ 'Data1': 'blue', 'Data2': 'green' })
sns.kdeplot(original_data_df.values)
plt.title("Original bimodal overlapping distribution with outliers")

# after: transformed distribution with optimal mixture
plt.subplot(1, 2, 2)
# set the colors according to the labels
sns.histplot(
    mapped_data,
    x='Feature',
    hue='Label',
    multiple='stack',
    bins=20,
    kde=False,
    # alpha=0.6,
    #legend=True,
    palette={ 'Data1': 'blue', 'Data2': 'green' })
sns.kdeplot(mapped_data.values)
plt.title("After ECDF-Mapping on Ideal Target Distribution")

plt.tight_layout()
plt.show()


# calculate the effect size using Cohen's d
def cohens_d(group1, group2):
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)
    return mean_diff / pooled_std

def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return np.abs((u1 - u2)) / s


# Cohen’s d before and after the transformation
orig_d = cohens_d(data1, data2)  # Ohne Ausreißer
# mapped data where label is 'Data1' and 'Data2'
data_label1 = mapped_data[mapped_data['Label'] == 'Data1']['Feature'].values
data_label2 = mapped_data[mapped_data['Label'] == 'Data2']['Feature'].values
mapped_d = cohens_d(data_label1, data_label2)

orig_d2 = cohend(data1, data2)
mapped_d2 = cohend(data_label1, data_label2)

print(f"Cohen’s d vorher: {orig_d:.3f}")
print(f"Cohen’s d nachher: {mapped_d:.3f}")
print(f"Cohen’s d vorher2: {orig_d2:.3f}")
print(f"Cohen’s d nachher2: {mapped_d2:.3f}")

