import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf
from modelmonitor import ModelMonitor

sym = 'YUM'
df = yf.download(sym, start='2017-01-01', end='2020-01-01', interval='1d', progress=False)
df.index = pd.to_datetime(df.index)
df.columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume']

df1 = df[df.index.year == 2017]
df2 = df[df.index.year == 2018]
df3 = df[df.index.year == 2019]

ks = lambda v, w: stats.ks_2samp(v, w).statistic

mm = ModelMonitor(ks, subset=['open', 'close'], labels=[2017, 2018, 2019], sep='::')
#  print(mm.evaluate(df1.close, df2.close))
#  print(mm.evaluate(df1.close, df2.close, df3.close))
#  print(mm.evaluate(df1, df2))
print(mm.evaluate(df1, df2, df3))
