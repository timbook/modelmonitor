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

mm.evaluate(X, Y, Z, labels=list('XYZ'))
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
df = pd.DataFrame(np.random.randn(1000, 5), columns=list('ABCDE'))
df['group'] = np.random.choice(['A', 'B', 'C', 'D'], size=1000, replace=True)
mm.evaluate(df, groupby='group')
#         A_B       B_C       C_D
# A  0.066709  0.080742  0.219649
# B  0.090407  0.155144  0.137199
# C  0.135637  0.099173  0.153390
# D  0.127743  0.095118  0.084503
# E  0.091089  0.176952  0.052983
```

