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

# Должен брать не нормал, а нормализовать?


def get_static_data(period='week'):
  np.seterr(divide='ignore', invalid='ignore')
  cnx_shares = sqlite3.connect('./storage/sqlite/shares.db')
  cnx_params = sqlite3.connect('./storage/sqlite/params.db')

  if period == 'week':
    df1 = pd.read_sql_query(
        "SELECT * from candles where time >= '2024-12-16 07:02:00.000' and time <= '2024-12-16 15:40:00.00'", cnx_shares)
    data1 = pd.read_sql_query(
        "SELECT * from normal where candle_id >= (?) and candle_id <= (?)", cnx_params, params=(str(df1["id"].min()), str(df1["id"].max())))
    df2 = pd.read_sql_query(
        "SELECT * from candles where time >= '2024-12-17 07:02:00.000' and time <= '2024-12-17 15:40:00.00'", cnx_shares)
    data2 = pd.read_sql_query(
        "SELECT * from normal where candle_id >= (?) and candle_id <= (?)", cnx_params, params=(str(df2["id"].min()), str(df2["id"].max())))
    df3 = pd.read_sql_query(
        "SELECT * from candles where time >= '2024-12-18 07:02:00.000' and time <= '2024-12-18 15:40:00.00'", cnx_shares)
    data3 = pd.read_sql_query(
        "SELECT * from normal where candle_id >= (?) and candle_id <= (?)", cnx_params, params=(str(df3["id"].min()), str(df3["id"].max())))
    df4 = pd.read_sql_query(
        "SELECT * from candles where time >= '2024-12-19 07:02:00.000' and time <= '2024-12-19 15:40:00.00'", cnx_shares)
    data4 = pd.read_sql_query(
        "SELECT * from normal where candle_id >= (?) and candle_id <= (?)", cnx_params, params=(str(df4["id"].min()), str(df4["id"].max())))
    df5 = pd.read_sql_query(
        "SELECT * from candles where time >= '2024-12-20 07:02:00.000' and time <= '2024-12-20 15:40:00.00'", cnx_shares)
    data5 = pd.read_sql_query(
        "SELECT * from normal where candle_id >= (?) and candle_id <= (?)", cnx_params, params=(str(df5["id"].min()), str(df5["id"].max())))
    df = pd.concat([df1, df2, df3, df4, df5])
    df = df.reset_index(drop=True)
    data = pd.concat([data1, data2, data3, data4, data5])
    data = data.reset_index(drop=True)
  else:
    df = pd.read_sql_query(
        "SELECT * from candles where time >= '2024-12-18 07:02:00.000' and time <= '2024-12-18 15:40:00.00'", cnx_shares)
    data = pd.read_sql_query(
        "SELECT * from normal where candle_id >= (?) and candle_id <= (?)", cnx_params, params=(str(df["id"].min()), str(df["id"].max())))

  data.drop(columns=["index", "candle_id"], inplace=True)
  X = data.to_numpy()

  print("\n[УСПЕШНО] Данные успешно получены")
  return df, X
