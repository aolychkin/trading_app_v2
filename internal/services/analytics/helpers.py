from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm

import sqlite3
import pandas as pd
import numpy as np


def merge_2m_1h(m, h):
  # print(len(m.columns))
  # print(tabulate(m.iloc[:15], headers='keys', tablefmt='psql'))
  # print(tabulate(h.iloc[:15], headers='keys', tablefmt='psql'))
  print("Starting...")
  df = pd.merge(m, h, on='id', how="outer")
  m_col = len(m.columns)+1
  h_col = len(h.columns) + len(m.columns)-1
  for i in tqdm(range(1, len(df))):
    if np.isnan(df.iloc[i, m_col]):
      df.iloc[i, m_col:h_col] = df.iloc[i-1, m_col:h_col]

  return df
