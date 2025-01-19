from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

import statsmodels.api as sm

import internal.services.analytics.indicators as indicators
import internal.services.analytics.params as params
import internal.services.analytics.helpers as helpers


def condition(x):
  # TODO: <= 0,
  if 0.16 <= x:
    return 1  # 57928
  elif x < 0.16:
    return 0  # 52820


def condition_row(row):
  # TODO: <= 0,
  if 0.2 <= row["pred_high"] and row["pred_low"] >= -0.08:
    return 1  # 57928
  elif 0.2 > row["pred_high"] or row["pred_low"] < -0.08:
    return 0  # 52820
  else:
    return -1


def condition_validation(x):
  # TODO: <= 0,
  if 0.16 <= x:
    return 1  # 57928
  elif x < 0.16:
    return 0  # 52820


def draw_hist(up, down, range):
  growth = (up / down - 1) * 100
  plt.hist(growth, bins=100, range=range, color='skyblue', edgecolor='black')
  plt.show()


def make_prediction(is_validation=False):
  cnx = sqlite3.connect('./storage/sqlite/shares.db')
  df = pd.read_sql_query(
      "SELECT id, time, open, high, low, close, volume FROM candles", cnx)

  df["max_high"] = df["high"].rolling(
      window=20, closed='right').max().shift(-20).fillna(0)
  df["min_low"] = df["low"].rolling(
      window=20, closed='right').min().shift(-20).fillna(0)
  # draw_hist(df["max_high"], df["close"], (-0.3, 1))
  df["pred_high"] = (df["max_high"] / df["close"] - 1) * 100
  df["pred_low"] = (df["min_low"] / df["close"] - 1) * 100

  if is_validation:
    df["class"] = df["pred_high"].apply(condition_validation)
  else:
    # df["class"] = df["pred_high"].apply(condition)
    df["class"] = df.apply(condition_row, axis=1)

  print(df["class"].value_counts())
  df.rename(columns={"id": "candle_id"}, inplace=True)
  return df[["candle_id", "class"]]
