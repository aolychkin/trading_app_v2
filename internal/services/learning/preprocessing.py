from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

import internal.services.learning.prediction as predict


def make_normal_data():
  cnx = sqlite3.connect('./storage/sqlite/params.db')
  engine = create_engine('sqlite:///storage/sqlite/params.db')
  df = pd.read_sql_query(
      "SELECT * FROM params", cnx)

  df_param = df[df["session"] == 1].copy()
  df_param.drop(columns=["index", "time", "session"], inplace=True)

  cols = df_param.columns.values[1:]
  Q1 = df_param[cols].quantile(0.1)
  Q3 = df_param[cols].quantile(0.9)
  IQR = Q3 - Q1
  df_param = df_param[~((df_param[cols] < (Q1 - 1.5 * IQR)) | (df_param[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

  scaler = MaxAbsScaler()
  x_scaled = scaler.fit_transform(df_param[cols].to_numpy())
  df_normal = pd.DataFrame(data=x_scaled.reshape(df_param["MACD_min"].count(), -1), columns=df_param[cols].columns)  # train_y.values.ravel()
  df_normal.insert(loc=0, column='candle_id', value=df_param[df_param.columns.values[0:1]].values)

  df_normal.to_sql(name='normal', con=engine, if_exists='replace')

  print(tabulate(df_normal.iloc[20:25], headers='keys', tablefmt='psql'))


def get_normal_data():
  cnx = sqlite3.connect('./storage/sqlite/params.db')
  df = pd.read_sql_query(
      "SELECT * FROM normal", cnx)
  df.drop(columns=["index"], inplace=True)
  # print(tabulate(df.iloc[20:25], headers='keys', tablefmt='psql'))
  return df
