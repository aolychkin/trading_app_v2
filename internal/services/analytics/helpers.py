from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm

import sqlite3
import pandas as pd
import numpy as np


def merge_2m_1h(m, h):
  df = pd.merge(m, h, on='id', how="outer")
  df.fillna(method='ffill', inplace=True)

  return df
