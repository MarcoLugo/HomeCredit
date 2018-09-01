# Marco Lugo

import numpy as np
import pandas as pd
import glob
from functools import reduce

input_files = glob.glob('*.csv')

df = reduce(lambda left,right: pd.merge(left, right, on='SK_ID_CURR',
			 how='left'), map(pd.read_csv, input_files))

avg_cols = [col for col in df.columns if col != 'SK_ID_CURR']
df['BLEND'] = df[avg_cols].mean(axis=1)
df.drop(avg_cols, axis=1, inplace=True)
df.rename(columns={'BLEND': 'TARGET'}, inplace=True)
df.to_csv('blend.csv', index=False)
