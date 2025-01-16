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


def draw_hist(up, down, range):
  growth = (up / down - 1) * 100
  plt.hist(growth, bins=100, range=range, color='skyblue', edgecolor='black')
  plt.show()


def resample_candles(df, min):
  df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df['time'])))
  df.drop(columns=["time"], inplace=True)
  df = df.resample(f'{str(min)}min', label='right', closed='right').agg({
      'id': 'max',
      'open': 'first',
      'high': 'max',
      'low': 'min',
      'close': 'last',
      'volume': 'sum',
  })
  return df.dropna()


def test_pacf(ts):  # Принимает 1 параметр Графика + time
  ts = ts.set_index(pd.DatetimeIndex(pd.to_datetime(ts['time'])))
  ts.drop(columns=["time"], inplace=True)

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
  sm.graphics.tsa.plot_acf(ts, ax=ax1, lags=50)
  sm.graphics.tsa.plot_pacf(ts, ax=ax2, lags=50)
  plt.show()


def calc_min(candles):
  ind = indicators.ta_ind_minute(candles.copy())
  macd = params.macd(ind.rename(columns={"MACD10_signal": "MACD_signal", "MACD12_24": "MACD"})[["id", "MACD_signal", "MACD"]].copy())
  rsi = params.rsi(ind.rename(columns={"RSI9": "RSI"})[["id", "RSI"]].copy())
  adx = params.adx(ind.rename(columns={"ADX9": "ADX"})[["id", "ADX"]].copy())
  ema = params.ema(ind.rename(columns={"EMA24_cl": "EMA"})[["id", "close", "EMA"]].copy())
  df_m = ind.merge(macd, on='id').merge(rsi, on='id').merge(adx, on='id').merge(ema, on='id')
  df_m.insert(loc=2, column='session', value=df_m["time"].apply(lambda x: indicators.w_session(x)))
  df_m["oc_min"] = df_m["close"] / df_m["open"] - 1
  df_m["DI_min"] = df_m["ADX9_pos"] / df_m["ADX9_neg"] - 1
  df_m["vEMA_min"] = df_m["volume"] / df_m["EMA9_vol"] - 1
  df_m["op_min"] = df_m["open"] - df_m["EMA24_op"]
  df_m["hi_min"] = df_m["high"] - df_m["EMA24_hi"]
  df_m["low_min"] = df_m["low"] - df_m["EMA24_low"]
  df_m["cl_min"] = df_m["close"] - df_m["EMA24_cl"]
  df_m.drop(df_m.columns.values[3:19], axis=1, inplace=True)
  df_m.drop(range(0, 933), inplace=True)
  return df_m


def calc_hour(candles):
  candles_1h = resample_candles(candles, 60)  # убрать всякие опен клоузы и тд, смержить по id
  candles_1h.insert(loc=1, column='time', value=candles_1h.index)
  candles_1h = candles_1h.set_index(pd.RangeIndex(0, len(candles_1h.index)))
  ind = indicators.ta_ind_hour(candles_1h.copy())
  macd = params.macd(ind.rename(columns={"MACD9_signal": "MACD_signal", "MACD12_26": "MACD"})[["id", "MACD_signal", "MACD"]].copy(), type="hour")
  rsi = params.rsi(ind.rename(columns={"RSI14": "RSI"})[["id", "RSI"]].copy(), type="hour")
  adx = params.adx(ind.rename(columns={"ADX14": "ADX"})[["id", "ADX"]].copy(), type="hour")
  ema = params.ema(ind.rename(columns={"EMA20_cl": "EMA"})[["id", "close", "EMA"]].copy(), type="hour")
  df_h = ind.merge(macd, on='id').merge(rsi, on='id').merge(adx, on='id').merge(ema, on='id')
  df_h["oc_hour"] = df_h["close"] / df_h["open"] - 1
  df_h["DI_hour"] = df_h["ADX14_pos"] / df_h["ADX14_neg"] - 1
  df_h["vEMA_hour"] = df_h["volume"] / df_h["EMA9_vol"] - 1
  df_h["op_hour"] = df_h["open"] - df_h["EMA20_op"]
  df_h["hi_hour"] = df_h["high"] - df_h["EMA20_hi"]
  df_h["low_hour"] = df_h["low"] - df_h["EMA20_low"]
  df_h["cl_hour"] = df_h["close"] - df_h["EMA20_cl"]
  df_h.drop(df_h.columns.values[2:18], axis=1, inplace=True)
  df_h.drop(range(0, 33), inplace=True)

  return df_h


def get_candle_analytics(candles, in_base=False):
  df_m = calc_min(candles)
  df_h = calc_hour(candles)

  df = pd.merge(df_m, df_h, on='id', how="outer")
  # print(tabulate(df.loc[0:100], headers='keys', tablefmt='psql'))

  df.fillna(method='ffill', inplace=True)
  df.drop(columns=["time_y"], inplace=True)
  df.dropna(inplace=True)
  df.rename(columns={"time_x": "time"}, inplace=True)
  if in_base:
    df.to_sql(name='params', con=engine, if_exists='replace')

  return df


if __name__ == '__main__':
  cnx = sqlite3.connect('./storage/sqlite/shares.db')
  engine = create_engine('sqlite:///storage/sqlite/params.db')
  candles = pd.read_sql_query(
      "SELECT id, time, open, high, low, close, volume FROM candles", cnx)
  # draw_hist(df["high"], df["low"], (-0.1, 0.8))
  get_candle_analytics(candles, in_base=True)

  # Sandbox
  # df = ind.merge(adx, on='id')
  # df.drop(columns=["time"], inplace=True)  # Временно, потом восстановить

  # df["prediction"] = df["high"].rolling(
  #     window=20, closed='right').max().shift(-20).fillna(0)
  # df.drop(columns=["time"], inplace=True)  # Временно, потом восстановить

  # Расчет параметров для часовых свечей

  # ind_1h = indicators.ta_ind_hour(candles_1h.copy())
  # df = pd.merge(ind, ind_1h, on='id', how="outer")

  # print(tabulate(df_h.iloc[:10], headers='keys', tablefmt='psql'))  # 933
  # print(tabulate(df.iloc[0:30], headers='keys', tablefmt='psql'))
  # print(tabulate(df.iloc[20:25], headers='keys', tablefmt='psql'))
  # print(tabulate(candles_1h.iloc[:15], headers='keys', tablefmt='psql'))
  # print(tabulate(candles.iloc[:15], headers='keys', tablefmt='psql'))
