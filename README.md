# Quantile Mapping for Bimodal Data Transformation

## Overview
This project demonstrates a method for mapping a bimodal dataset with outliers onto an ideal target distribution using quantile transformation. The approach aims to reshape the original data into a well-separated bimodal distribution while maintaining essential statistical properties to increase the effect size.

## Features
- Simulates a small bimodal dataset with outliers as an example.
- Detects the number of peaks in the probability density function to prevent creating an artificial effect size.
- Creates an ideal bimodal target distribution.
- Maps the original dataset to the target distribution using quantile transformation.
- Visualizes the original and transformed distributions.
- Computes Cohen's d effect size before and after transformation.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas scipy matplotlib seaborn
```

## Usage
### 1. Clone the Repository
```bash
git clone https://github.com/sigrun-may/multimodal-transformation.git
cd multimodal-transformation
```
### 2. Run the Script
Execute the script in a Python environment:
```bash
python multimodal_transformation.py
```
This will generate visualizations and print Cohen's d effect sizes.

## Code Explanation
### Data Simulation
- Generates a bimodal dataset with outliers.
- Assigns labels (`Data1`, `Data2`) for distinction.

### Peak Detection
- Uses Gaussian KDE to estimate the probability density function.
- Counts the number of peaks to check bimodality.

### Quantile Transformation
- Maps the original dataset to a predefined target distribution with two well-separated modes.
- Uses empirical cumulative distribution function (ECDF) for transformation.

### Visualization
- Plots the original dataset with outliers.
- Plots the transformed dataset after ECDF mapping.

### Effect Size Computation
- Computes **Cohen’s d** before and after transformation to measure the effect size.

## Results
- The quantile transformation reshapes the distribution into a well-separated bimodal structure.
- Cohen’s d effect size increases, indicating improved separability.

## Example Output
```
Number of peaks: 2
Cohen’s d before: 0.854
Cohen’s d after: 2.356
```

## License
This project is licensed under the MIT License.

## Author
Sigrun May - [GitHub Profile](https://github.com/sigrun-may)

---
Feel free to contribute by submitting issues or pull requests!

