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


def draw_candles(fig, df_model, profile, i_row):
  fig.add_trace(
      go.Candlestick(
          x=df_model['time'],
          open=df_model['open'],
          high=df_model['high'],
          low=df_model['low'],
          close=df_model['close']),
      row=i_row, col=1)

  for index, row in tqdm(profile.iterrows()):
    if row["result"] == "/":
      fig.add_trace(
          go.Scatter(
              x=[row["time"]],
              y=[row["price"]],
              mode='markers',
              marker={'size': 15, 'color': 'blue'},
              name=f'enter_{row["id"]}_acc_{row["accuracy"]}'),
          row=i_row, col=1)
    else:
      if row["cause"] == "expired":
        if row["result"] == "+":
          fig.add_trace(
              go.Scatter(
                  x=[row["time"]],
                  y=[row["price"]],
                  mode='markers',
                  marker={'size': 15, 'color': 'green', 'symbol': 300, 'line_width': 3},
                  name=f'close_{row["id"]}'),
              row=i_row, col=1
          )
        elif row["result"] == "-":
          fig.add_trace(
              go.Scatter(
                  x=[row["time"]],
                  y=[row["price"]],
                  mode='markers',
                  marker={'size': 15, 'color': 'red', 'symbol': 300, 'line_width': 3},
                  name=f'close_{row["id"]}'),
              row=i_row, col=1
          )
        elif row["result"] == "=":
          fig.add_trace(
              go.Scatter(
                  x=[row["time"]],
                  y=[row["price"]],
                  mode='markers',
                  marker={'size': 15, 'color': 'yellow', 'symbol': 300, 'line_width': 3},
                  name=f'close_{row["id"]}'),
              row=i_row, col=1
          )
      else:
        if row["result"] == "+":
          fig.add_trace(
              go.Scatter(
                  x=[row["time"]],
                  y=[row["price"]],
                  mode='markers',
                  marker={'size': 15, 'color': 'green'},
                  name=f'close_{row["id"]}'),
              row=i_row, col=1
          )
        elif row["result"] == "-":
          fig.add_trace(
              go.Scatter(
                  x=[row["time"]],
                  y=[row["price"]],
                  mode='markers',
                  marker={'size': 15, 'color': 'red'},
                  name=f'close_{row["id"]}'),
              row=i_row, col=1
          )
        elif row["result"] == "=":
          fig.add_trace(
              go.Scatter(
                  x=[row["time"]],
                  y=[row["price"]],
                  mode='markers',
                  marker={'size': 15, 'color': 'yellow'},
                  name=f'close_{row["id"]}'),
              row=i_row, col=1
          )
  return fig
