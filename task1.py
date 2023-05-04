import pandas as pd
import numpy as np

data = pd.read_csv('winequality-white.csv')

rows = np.random.choice(data.index, size=int(len(data)*0.05), replace=False)
data.iloc[rows,:] = np.nan

data.to_csv('reducedWine.csv', index=False, na_rep='NaN')
