# 00 Initialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm

desired_width = 500
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', desired_width)
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width, suppress=True, formatter={'float_kind': '{:3}'.format})
plt.rc('font', size=12)

Data = pd.DataFrame()
Data['Variable'] = np.random.randn(100)

# 01 Get data attributes
X = Data['Variable']
SortedValues = np.sort(X.values)
N = len(X)
X_Bar = np.mean(X)
S_X = np.std(X, ddof=1)

# 05 Kernel density estimation (Gaussian kernel)
KernelEstimator = np.zeros(N)
NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25, 0.75]), 0, 1)))
DataIQR = np.abs(X.quantile(0.75)) - np.abs(X.quantile(0.25))
KernelHalfWidth = 0.9 * N ** (-1 / 5) * min(np.abs([S_X, DataIQR / NormalIQR]))
for Value in SortedValues:
    KernelEstimator += norm.pdf(SortedValues - Value, 0, KernelHalfWidth * 2)
KernelEstimator = KernelEstimator / N

## Histogram and density distribution
TheoreticalDistribution = norm.pdf(SortedValues, X_Bar, S_X)
Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.hist(X, density=True, bins=20, edgecolor=(0, 0, 1), color=(1, 1, 1), label='Histogram')
Axes.plot(SortedValues, KernelEstimator, color=(1, 0, 0), label='Kernel Density')
Axes.plot(SortedValues, TheoreticalDistribution, linestyle='--', color=(0, 0, 0), label='Normal Distribution')
plt.xlabel('Variable')
plt.ylabel('Density (-)')
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15), prop={'size': 10})
plt.show()
