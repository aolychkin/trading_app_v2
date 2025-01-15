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
import warnings


# TODO: Поработать с данными в динамике (нормализовать на ходу тииипа)
# TODO: Проверить анализ показателей
if __name__ == '__main__':
  warnings.filterwarnings(action='ignore')
  df_model, X = stack.get_static_data(period='day')

  # print(tabulate(df_model.iloc[0:200], headers='keys', tablefmt='psql'))

  model = joblib.load("./ml_models/main_model_4.pkl")
  val_model = joblib.load("./ml_models/val_model_4.pkl")
  counter = 0
  for index, x in enumerate(X):
    df_model.loc[index, ["0", "target"]] = (
        model.predict_proba(x.reshape(1, -1))).ravel()
    df_model.loc[index, ["val_0", "val_target"]] = (
        val_model.predict_proba(x.reshape(1, -1)))
    # df_model.loc[index, ["0", "1"]] = (model.predict_proba(x.reshape(1, -1)))
    # if (model.predict(x.reshape(1, -1)) == 1):
    #   counter += 1
    #   print(model.predict_proba(x.reshape(1, -1)))
    # if counter == 20:
    #   break

  # profile = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.8, stop_loss=0.016, take_profit=0.004, target="val_target", limit=6)
  # profile = simulate.strategy(df_model, accuracy=0.69, max_accuracy=0.8, stop_loss=0.016, take_profit=0.004, target="val_target", limit=6)  # TOP
  profile_1 = simulate.strategy(df_model, accuracy=0.62, max_accuracy=0.75, stop_loss=0.0025, take_profit=0.004, target="val_target", limit=6)  # TOP
  profile_2 = simulate.strategy(df_model, accuracy=0.61, max_accuracy=0.8, stop_loss=0.0025, take_profit=0.004, target="target", limit=6)  # TOP

  fig = make_subplots(
      rows=2, cols=1,
      subplot_titles=("Ttitle 1", "Title 2"))

  fig = draw.draw_candles(fig, df_model, profile_1, 1)
  fig = draw.draw_candles(fig, df_model, profile_2, 2)
  fig.show()
