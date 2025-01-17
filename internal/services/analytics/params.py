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


def power_condition(row):
  if row['cross_type'] == 0:
    if (row["hist"] > 0):  # Показатель может быть отрицательным сам по себе!
      return 1 - row["base_power"]
    elif (row["hist"] < 0):
      return -1 + row["base_power"]
  else:
    if row['cross_type'] == 1:
      return 1 + row["base_power"]
    else:
      return -1 - row["base_power"]


def power_condition_dis_balanced(row):
  if row['cross_type'] == 0:
    if (row["hist"] > 0):
      return 1 - row["base_power"] / 3
    elif (row["hist"] < 0):
      return -1 + row["base_power"] / 3
  else:
    if row['cross_type'] == 1:
      return 1 + row["base_power"]
    else:
      return -1 - row["base_power"]


def macd(df, type="min"):
  df.insert(loc=1, column='MACD_signal_pred', value=df["MACD_signal"].shift(1))
  df.insert(loc=3, column='MACD_pred', value=df["MACD"].shift(1))
  df["hist"] = df["MACD"] - df["MACD_signal"]
  df["hist_pred"] = df["hist"].shift(1)
  df["cross_type"] = df.apply(lambda row: -1 if (row["hist_pred"] > 0 and row["hist"] <= 0) else 1 if (row["hist_pred"] <= 0 and row["hist"] > 0) else 0, axis=1)
  df["base_power"] = np.abs(df["hist"])
  df["power"] = df.apply(power_condition, axis=1)

  print("MACD:", df["power"].min(), df["power"].max())
  df.rename(columns={"power": f"MACD_{type}"}, inplace=True)
  return (df[["id", f"MACD_{type}"]].copy())


def rsi(df, type="min"):
  df.insert(loc=1, column='RSI_pred', value=df["RSI"].shift(1))
  df["cross_type"] = df.apply(lambda row: cross(row["RSI_pred"], row["RSI"], 70, 70), axis=1)
  df["counter"] = 0
  df["ASS"] = 70
  df["hist"] = df["RSI"] - 70
  df["hist_pred"] = df["hist"].shift(1)
  df["cross_type"] = df.apply(lambda row: -1 if (row["hist_pred"] > 0 and row["hist"] <= 0) else 1 if (row["hist_pred"] <= 0 and row["hist"] > 0) else 0, axis=1)
  df["base_power"] = df.apply(lambda row: np.abs(np.mean([row["RSI_pred"], row["RSI"]]) / 70 - 1), axis=1)
  df["power"] = df.apply(power_condition, axis=1)

  print("RSI:", df["power"].min(), df["power"].max())
  df.rename(columns={"power": f"RSI_{type}"}, inplace=True)
  return (df[["id", f"RSI_{type}"]].copy())


def adx(df, type="min"):  # Растущий и высокий тренд (больше 25 и чем больше - тем лучше) = подтверждение продажи и покупки. "Для ADX все что не рост - все слабость тренда"
  df.insert(loc=1, column='ADX_pred', value=df["ADX"].shift(1))
  df["ASS"] = 25
  df["base_power"] = df.apply(lambda row: np.nan if row['ADX_pred'] == 0 else np.abs(row['ADX'] / row['ADX_pred'] - 1), axis=1)
  df["power"] = df.apply(
      lambda row: np.nan if row['ADX_pred'] == 0 else 1 + (row['ADX'] / row['ADX_pred'] - 1) if row['ADX'] >= 25 else -1 - np.abs(row['ADX'] / row['ADX_pred'] - 1),
      axis=1)

  print("ADX:", df["power"].min(), df["power"].max())
  df.rename(columns={"power": f"ADX_{type}"}, inplace=True)
  return (df[["id", f"ADX_{type}"]].copy())


def ema(df, type="min"):
  df.insert(loc=1, column='EMA_pred', value=df["EMA"].shift(1))
  df.insert(loc=3, column='close_pred', value=df["close"].shift(1))
  df["hist"] = df["close"] - df["EMA"]
  df["hist_pred"] = df["hist"].shift(1)
  df["cross_type"] = df.apply(lambda row: -1 if (row["hist_pred"] > 0 and row["hist"] <= 0) else 1 if (row["hist_pred"] <= 0 and row["hist"] > 0) else 0, axis=1)
  df["base_power"] = df.apply(lambda row: np.abs(np.mean([row["close_pred"], row["close"]]) / np.mean([row["EMA_pred"], row["EMA"]])-1) * 100, axis=1)
  df["power"] = df.apply(power_condition, axis=1)

  print("сEMA:", df["power"].min(), df["power"].max())
  df.rename(columns={"power": f"сEMA_{type}"}, inplace=True)
  return (df[["id", f"сEMA_{type}"]].copy())
