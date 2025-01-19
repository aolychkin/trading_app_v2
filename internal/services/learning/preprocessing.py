from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm
import time

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine


import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer

import internal.services.learning.prediction as predict


def make_normal_data():
  cnx = sqlite3.connect('./storage/sqlite/params.db')
  engine = create_engine('sqlite:///storage/sqlite/params.db')
  df = pd.read_sql_query(
      "SELECT * FROM params", cnx)

  df_param = df[df["session"] == 1].copy()
  df_param.drop(columns=["index", "time", "session"], inplace=True)

  cols = df_param.columns.values[1:]
  # Q1 = df_param[cols].quantile(0.05)
  # Q3 = df_param[cols].quantile(0.95)
  # IQR = Q3 - Q1
  # df_param = df_param[~((df_param[cols] < (Q1 - 1.5 * IQR)) | (df_param[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

  fig = make_subplots(
      rows=3, cols=4,
      subplot_titles=("MACD_min", "RSI_min_70", "ADX_min", "сEMA_min"))

  # fig = go.Figure()
  bins = 50
  fig.add_trace(go.Histogram(x=df_param["op_hour"], nbinsx=bins), row=1, col=1)
  fig.add_trace(go.Histogram(x=df_param["RSI_min_70"], nbinsx=bins), row=1, col=2)
  fig.add_trace(go.Histogram(x=df_param["ADX_min"], nbinsx=bins), row=1, col=3)
  fig.add_trace(go.Histogram(x=df_param["сEMA_min"], nbinsx=bins), row=1, col=4)

  scaler = MaxAbsScaler()  # MaxAbsScaler()
  x_scaled = scaler.fit_transform(df_param[cols].to_numpy())
  df_normal = pd.DataFrame(data=x_scaled.reshape(df_param["ADX_min"].count(), -1), columns=df_param[cols].columns)  # train_y.values.ravel()
  df_normal.insert(loc=0, column='candle_id', value=df_param[df_param.columns.values[0:1]].values)
  fig.add_trace(go.Histogram(x=df_normal["op_hour"], nbinsx=bins), row=2, col=1)
  fig.add_trace(go.Histogram(x=df_normal["RSI_min_70"], nbinsx=bins), row=2, col=2)
  fig.add_trace(go.Histogram(x=df_normal["ADX_min"], nbinsx=bins), row=2, col=3)
  fig.add_trace(go.Histogram(x=df_normal["сEMA_min"], nbinsx=bins), row=2, col=4)

  quantile_transformer = QuantileTransformer(output_distribution='normal')
  x_scaled = quantile_transformer.fit_transform(df_normal[cols].to_numpy())
  df_quant = pd.DataFrame(data=x_scaled.reshape(df_normal["ADX_min"].count(), -1), columns=df_normal[cols].columns)  # train_y.values.ravel()
  df_quant.insert(loc=0, column='candle_id', value=df_normal[df_normal.columns.values[0:1]].values)

  fig.add_trace(go.Histogram(x=df_quant["op_hour"], nbinsx=bins), row=3, col=1)
  fig.add_trace(go.Histogram(x=df_quant["RSI_min_70"], nbinsx=bins), row=3, col=2)
  fig.add_trace(go.Histogram(x=df_quant["ADX_min"], nbinsx=bins), row=3, col=3)
  fig.add_trace(go.Histogram(x=df_quant["сEMA_min"], nbinsx=bins), row=3, col=4)

  fig.show()

  df_normal.to_sql(name='normal', con=engine, if_exists='replace')

  print(tabulate(df_normal.iloc[20:25], headers='keys', tablefmt='psql'))


def calc_normal_data(df_param):
  df_param.drop(columns=["time", "session"], inplace=True)
  cols = df_param.columns.values[1:]
  scaler = MaxAbsScaler()  # StandardScaler
  x_scaled = scaler.fit_transform(df_param[cols].to_numpy())
  df_normal = pd.DataFrame(data=x_scaled.reshape(df_param["ADX_min"].count(), -1), columns=df_param[cols].columns)  # train_y.values.ravel()
  df_normal.insert(loc=0, column='candle_id', value=df_param[df_param.columns.values[0:1]].values)

  return df_normal


def get_normal_data():
  cnx = sqlite3.connect('./storage/sqlite/params.db')
  df = pd.read_sql_query(
      "SELECT * FROM normal", cnx)
  df.drop(columns=["index"], inplace=True)
  # print(tabulate(df.iloc[20:25], headers='keys', tablefmt='psql'))
  return df
