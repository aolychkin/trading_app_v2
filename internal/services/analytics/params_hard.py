from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm

import sqlite3
import pandas as pd
import numpy as np

from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

import plotly.graph_objects as go


def cross(s_down, f_down, s_up, f_up):  # down должен пересекать снизу up
  y1 = 1
  y2 = 2
  y3 = 1
  y4 = 2
  n = 0
  down_up = -1
  if (y2 - y1 != 0):
    q = (f_down - s_down) / (y1 - y2)
    sn = (s_up - f_up) + (y3 - y4) * q
    if (not sn):
      return 0
    fn = (s_up - s_down) + (y3 - y1) * q
    n = fn / sn
  else:
    n = (y3 - y1) / (y3 - y4)

  dot2 = y3 + (y4 - y3) * n
  if (1 <= dot2 <= 2):
    if (dot2 == 2):
      down_up = 1 if (s_down/s_up-1 < 0) else 0
    elif (dot2 == 1):
      down_up = 1 if (f_down/f_up-1 > 0) else 0
    else:
      down_up = 1 if (s_down/s_up-1 < 0) else 0
  else:
    down_up = -1

  if down_up == 1:
    return 1  # покупать
  elif down_up == 0:
    return -1  # продавать
  else:
    return 0


def macd(df, type="min"):
  df.insert(loc=0, column='MACD_signal_pred', value=df["MACD_signal"].shift(1))
  df.insert(loc=2, column='MACD_pred', value=df["MACD"].shift(1))
  df["cross_type"] = df.apply(lambda row: cross(row["MACD_pred"], row["MACD"], row["MACD_signal_pred"], row["MACD_signal"]), axis=1)
  for i in tqdm(range(1, len(df))):
    power = np.abs(df.loc[i, "MACD"] - df.loc[i, "MACD_signal"])
    if df.loc[i, 'cross_type'] == 0:
      try:
        df.loc[i, 'tmp_power'] = df.loc[i-1, 'tmp_power']
      except:
        df.loc[i, 'tmp_power'] = -1
      if (df.loc[i, 'tmp_power'] > 0):
        df.loc[i, 'power'] = 1 - power
      elif (df.loc[i, 'tmp_power'] < 0):
        df.loc[i, 'power'] = -1 + power
    else:
      if df.loc[i, 'cross_type'] == 1:
        df.loc[i, 'power'] = 1 + power
        df.loc[i, 'tmp_power'] = 1 + power
      else:
        df.loc[i, 'power'] = -1 - power
        df.loc[i, 'tmp_power'] = -1 - power

  # print(df["power"].head())
  # print(tabulate(df.loc[220:270], headers='keys', tablefmt='psql'))
  # print(df["cross_type"].value_counts())
  print("MACD:", df["power"].min(), df["power"].max())
  df.rename(columns={"power": f"MACD_{type}"}, inplace=True)
  return (df[["id", f"MACD_{type}"]].copy())
