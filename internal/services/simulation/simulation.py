from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm
from pprint import pp

import sqlite3
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

import internal.services.simulation.strategy as simulate
import internal.services.simulation.drawing as draw
import internal.services.simulation.data as stack
import internal.services.analytics.analytics as anal
import internal.services.learning.preprocessing as prep
import internal.services.data.data as data

import warnings


def start_simulation():
  warnings.filterwarnings(action='ignore')
  df_candles, df_cid, df_f_cid = data.load_day_candle(1, 16)
  df_param = anal.get_candle_analytics(df_candles.copy(), in_base=False)
  df_normal = prep.calc_normal_data(df_param)
  df_candles.rename(columns={"id": "candle_id"}, inplace=True)

  df = pd.merge(df_candles, df_normal, on='candle_id', how="outer")
  df.dropna(inplace=True)
  df = df[(df["candle_id"] <= df_f_cid) & (df["candle_id"] >= df_cid)]

  df_model = df.iloc[:, :7].copy()
  X = df.iloc[:, 7:].copy().to_numpy()

  model = joblib.load("./ml_models/v115/main_model_1.pkl")
  # val_model = joblib.load("./ml_models/val_model_3.pkl")
  counter = 0
  for index, x in enumerate(X):
    df_model.loc[df_model.index.min()+index, ["0", "target"]] = (
        model.predict_proba(x.reshape(1, -1))).ravel()

  profile_1 = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.93, stop_loss=0.0016, take_profit=0.002, target="target", limit=6)  # TOP
  profile_2 = simulate.strategy(df_model, accuracy=0.8, max_accuracy=1, stop_loss=0.002, take_profit=0.002, target="target", limit=6)  # TOP

  fig = make_subplots(
      rows=2, cols=1,
      subplot_titles=("profile_1", "profile_2"))

  fig = draw.draw_candles(fig, df_model, profile_1, 1)
  fig = draw.draw_candles(fig, df_model, profile_2, 2)
  fig.show()


# TODO: Поработать с данными в динамике (нормализовать на ходу тииипа)
# TODO: Проверить анализ показателей
if __name__ == '__main__':
  start_simulation()
  # warnings.filterwarnings(action='ignore')
  # cnx_shares = sqlite3.connect('./storage/sqlite/shares.db')
  # df_candles, df_cid, df_f_cid = data.load_day_candle(1, 15)
  # # df_candles = pd.read_sql_query("SELECT id, time, open, high, low, close, volume from candles where time >= '2024-12-15 07:02:00.000' and time <= '2024-12-17 15:40:00.00'", cnx_shares)
  # # df_cid = pd.read_sql_query("SELECT id, time, open, high, low, close, volume from candles where time >= '2024-12-17 07:02:00.000' and time <= '2024-12-17 07:04:00.00'", cnx_shares)
  # df_param = anal.get_candle_analytics(df_candles.copy(), in_base=False)
  # df_normal = prep.calc_normal_data(df_param)
  # df_candles.rename(columns={"id": "candle_id"}, inplace=True)

  # df = pd.merge(df_candles, df_normal, on='candle_id', how="outer")
  # df.dropna(inplace=True)
  # df = df[(df["candle_id"] <= df_f_cid) & (df["candle_id"] >= df_cid)]

  # df_model = df.iloc[:, :7].copy()
  # X = df.iloc[:, 7:].copy().to_numpy()

  # model = joblib.load("./ml_models/v115/main_model_1.pkl")
  # # val_model = joblib.load("./ml_models/val_model_3.pkl")
  # counter = 0
  # for index, x in enumerate(X):
  #   df_model.loc[df_model.index.min()+index, ["0", "target"]] = (
  #       model.predict_proba(x.reshape(1, -1))).ravel()
  #   # df_model.loc[df_model.index.min()+index, ["val_0", "val_target"]] = (
  #   #     val_model.predict_proba(x.reshape(1, -1))).ravel()

  #   # if (model.predict(x.reshape(1, -1)) == 1):
  #   #   counter += 1
  #   #   print(model.predict_proba(x.reshape(1, -1)).ravel())
  #   #   print(index)
  #   # if counter == 100:
  #   #   break

  # # profile_1 = simulate.strategy(df_model, accuracy=0.5, max_accuracy=1, stop_loss=0.0016, take_profit=0.0016, target="target", limit=6)  # TOP
  # # profile_2 = simulate.strategy(df_model, accuracy=0.5, max_accuracy=1, stop_loss=0.0016, take_profit=0.0016, target="val_target", limit=6)  # TOP
  # profile_1 = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.93, stop_loss=0.02, take_profit=0.002, target="target", limit=6)  # TOP
  # profile_2 = simulate.strategy(df_model, accuracy=0.65, max_accuracy=0.99, stop_loss=0.02, take_profit=0.002, target="target", limit=6)  # TOP
  # # profile_2 = simulate.strategy(df_model, accuracy=0.6, max_accuracy=1, stop_loss=0.002, take_profit=0.002, target="val_target", limit=6)  # TOP
  # # profile_1 = simulate.strategy(df_model, accuracy=0.61, max_accuracy=0.80, stop_loss=0.0025, take_profit=0.0035, target="target", limit=6)  # TOP
  # # profile_2 = simulate.strategy(df_model, accuracy=0.62, max_accuracy=0.75, stop_loss=0.0025, take_profit=0.0035, target="val_target", limit=6)  # TOP
  # # profile_1 = simulate.strategy(df_model, accuracy=0.61, max_accuracy=0.80, stop_loss=0.0025, take_profit=0.0035, target="target", limit=6)  # TOP
  # # profile_2 = simulate.strategy(df_model, accuracy=0.62, max_accuracy=0.75, stop_loss=0.0025, take_profit=0.0035, target="val_target", limit=6)  # TOP

  # fig = make_subplots(
  #     rows=2, cols=1,
  #     subplot_titles=("profile_1", "profile_2"))

  # fig = draw.draw_candles(fig, df_model, profile_1, 1)
  # fig = draw.draw_candles(fig, df_model, profile_2, 2)
  # fig.show()
