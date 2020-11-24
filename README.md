# Model Monitor
A library containing an object useful for detecting feature shift over time.  
# Installation
This library is not yet hosted on PyPI. To install it yourself, simply clone and install as follows:

```
git clone https://github.com/timbook/modelmonitor
cd modelmonitor
pip install .
```

# Usage
Given some metric, the `ModelMonitor` class computes that metric across all input arrays sequentially. If two-dimensional arrays (Numpy arrays or Pandas DataFrames) are given, the metric is computed column-wise.

## Example 1: Between two or more matrices
```python
import numpy as np
from scipy.stats import wasserstein_distance

from modelmonitor import ModelMonitor

np.random.seed(42)
X = np.random.randn(100, 5)
Y = np.random.randn(100, 5)
Z = np.random.randn(100, 5)
Z[:, 4] = 1.1*Z[:, 4]

mm = ModelMonitor(metric=wasserstein_distance)

mm.evaluate(X, Y)
# 0    0.127148
# 1    0.206939
# 2    0.179723
# 3    0.135772
# 4    0.241676
# dtype: float64

mm.evaluate(X, Z)
# 0    0.161786
# 1    0.171553
# 2    0.232102
# 3    0.101372
# 4    0.356038
# dtype: float64

mm.set_labels(list('XYZ'))
mm.evaluate(X, Y, Z)
#         X_Y       Y_Z
# 0  0.127148  0.117008
# 1  0.206939  0.245297
# 2  0.179723  0.119153
# 3  0.135772  0.121778
# 4  0.241676  0.126118
```

## Example 2: Between groups in a DataFrame
```python
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from modelmonitor import ModelMonitor

np.random.seed(42)
df = pd.DataFrame(np.random.randn(1000, 5), columns=['V1', 'V2', 'V3', 'V4', 'V5'])
df['group'] = np.random.choice(list("ABCDE") , size=1000, replace=True)
mm.evaluate(df, groupby='group')
#          A_B       B_C       C_D       D_E
# V1  0.136266  0.128704  0.080303  0.105263
# V2  0.124382  0.210017  0.076414  0.138592
# V3  0.077364  0.178851  0.148629  0.125337
# V4  0.097102  0.075041  0.139569  0.124033
# V5  0.069333  0.099172  0.114340  0.132053
```
Note that the labels are sorted in alphabetical order. Allowing for custom orderings via a pd.Categorical is a TODO.
