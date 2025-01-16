from datetime import datetime
from datetime import time
from tabulate import tabulate
from tqdm import tqdm
from pprint import pp

import sqlite3
import pandas as pd
import joblib
import plotly.graph_objects as go
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report


def strategy(
        data, accuracy, stop_loss, take_profit, max_accuracy=1, wait=20,
        is_val=False, target="target", val_accuracy=0.9, val_max_accuracy=1.0, val_target="val_target",
        debug=False, limit=5):
  np.seterr(divide='ignore', invalid='ignore')
  indx = 0
  profile = pd.DataFrame()
  transaction_id = 0
  profile["id"] = [transaction_id]
  profile["transaction"] = [700]
  profile["balance"] = [700]
  profile["candle_id"] = [0]
  profile["price"] = [700]
  profile["is_closed"] = [1]
  profile["cause"] = ["money"]
  profile["accuracy"] = [0.00]
  profile["result"] = [""]
  profile["time"] = [np.datetime64(datetime.now())]

  # TODO: Подкрепить показателями теми же самыми за период в 10 минут для каждой свечи минутной внутри (более длинный тренд)
  # TODO: Обучение норм, нужно работать с данными

  for index, row in tqdm(data.iterrows()):
    # Сигнал на покупку акции
    if (accuracy <= row[target] <= max_accuracy) and (profile[profile["is_closed"] == 0]["is_closed"].count() < limit):
      if datetime.strptime(row["time"], '%Y-%m-%d %H:%M:%S.%f').strftime("%H:%M") >= datetime(2024, 12, 1, 15, 38).strftime("%H:%M"):
        continue
      if is_val:
        if (val_accuracy <= row[val_target] <= val_max_accuracy):
          transaction_id += 1
          balance = round(profile.loc[indx, "balance"], 2)
          price = data.loc[index+1, "open"]
          indx += 1
          profile.loc[indx, "id"] = transaction_id
          profile.loc[indx, "transaction"] = -round(price * (1+0.0004), 2)
          profile.loc[indx, "balance"] = round(balance-round(price * (1+0.0004), 2), 2)
          profile.loc[indx, "candle_id"] = row["candle_id"]  # data.loc[index+1, "id"]
          profile.loc[indx, "price"] = price
          profile.loc[indx, "is_closed"] = 0
          profile.loc[indx, "accuracy"] = row[target]
          profile.loc[indx, "time"] = data.loc[index+1, "time"]  # row["time"]
      else:
        transaction_id += 1
        balance = round(profile.loc[indx, "balance"], 2)
        price = data.loc[index+1, "open"]
        indx += 1
        profile.loc[indx, "id"] = transaction_id
        profile.loc[indx, "transaction"] = -round(price * (1+0.0004), 2)
        profile.loc[indx, "balance"] = round(balance-round(price * (1+0.0004), 2), 2)
        profile.loc[indx, "candle_id"] = row["candle_id"]  # data.loc[index+1, "id"]
        profile.loc[indx, "price"] = price
        profile.loc[indx, "is_closed"] = 0
        profile.loc[indx, "accuracy"] = row[target]
        profile.loc[indx, "time"] = data.loc[index+1, "time"]  # row["time"]

    for index, p_row in profile.iterrows():
      if (p_row["is_closed"] == 0):
        if (0 <= row["candle_id"] - p_row["candle_id"] <= wait + 1) and (row["high"]/p_row["price"] - 1 >= take_profit):
          indx += 1
          profile.loc[indx, "id"] = p_row["id"]
          profile.loc[indx, "transaction"] = round(p_row['price'] * (1-0.0004-stop_loss), 2)
          profile.loc[indx, "balance"] = round(profile.loc[indx-1, "balance"]+round(p_row['price'] * (1-0.0004+take_profit), 2), 2)
          profile.loc[indx, "candle_id"] = row["candle_id"]
          profile.loc[indx, "price"] = round(p_row['price'] * (1+take_profit), 2)
          profile.loc[indx, "is_closed"] = 1
          profile.loc[index, "is_closed"] = 1
          profile.loc[indx, "cause"] = "take_profit"
          profile.loc[index, "cause"] = "take_profit"
          profile.loc[index, "result"] = "/"
          profile.loc[indx, "result"] = "+"
          profile.loc[indx, "time"] = row["time"]
        elif (0 < row["candle_id"] - p_row["candle_id"] <= wait + 1) and (row["low"]/p_row["price"] - 1 <= -stop_loss):
          indx += 1
          profile.loc[indx, "id"] = p_row["id"]
          profile.loc[indx, "transaction"] = round(p_row['price'] * (1-0.0004-stop_loss), 2)
          profile.loc[indx, "balance"] = round(profile.loc[indx-1, "balance"]+round(p_row['price'] * (1-0.0004-stop_loss), 2), 2)
          profile.loc[indx, "candle_id"] = row["candle_id"]
          profile.loc[indx, "price"] = round(p_row['price'] * (1-stop_loss), 2)
          profile.loc[indx, "is_closed"] = 1
          profile.loc[index, "is_closed"] = 1
          profile.loc[indx, "cause"] = "stop_loss"
          profile.loc[index, "cause"] = "stop_loss"
          profile.loc[index, "result"] = "/"
          profile.loc[indx, "result"] = "-"
          profile.loc[indx, "time"] = row["time"]
        elif (row["candle_id"] - p_row["candle_id"] > wait+1) or (datetime.strptime(row["time"], '%Y-%m-%d %H:%M:%S.%f').strftime("%H:%M") == datetime(2024, 12, 1, 15, 38).strftime("%H:%M")):
          indx += 1
          # sell = round(row['close'] * (1-0.0004), 2)
          profile.loc[indx, "id"] = p_row["id"]
          profile.loc[indx, "transaction"] = round(row['close'] * (1-0.0004), 2)
          profile.loc[indx, "balance"] = round(profile.loc[indx-1, "balance"]+round(row['close'] * (1-0.0004), 2), 2)
          profile.loc[indx, "candle_id"] = row["candle_id"]
          profile.loc[indx, "price"] = row['close']
          profile.loc[indx, "is_closed"] = 1
          profile.loc[index, "is_closed"] = 1
          profile.loc[indx, "cause"] = "expired"
          profile.loc[index, "cause"] = "expired"
          profile.loc[indx, "time"] = row["time"]
          profile.loc[index, "result"] = "/"
          if (profile.loc[indx, "price"] / profile.loc[index, "price"] - 1 > 0):
            profile.loc[indx, "result"] = "+"
          elif (profile.loc[indx, "price"] / profile.loc[index, "price"] - 1 == 0):
            profile.loc[indx, "result"] = "="
          else:
            profile.loc[indx, "result"] = "-"

  # delta_money = round((profile.tail(1)["balance"].values[0] / profile.head(1)["balance"].values[0] - 1)*100, 2)
  delta_money = round(((profile.tail(1)["balance"].values[0] - profile["balance"].min())/(profile.head(1)["balance"].values[0] - profile["balance"].min())-1)*100, 2)
  stop_order = profile["cause"].value_counts().sort_index()
  profit = profile["result"].value_counts().sort_index()
  print(profile["balance"].min())
  try:
    print(f"\
      Параметры: acc={accuracy}, sl={stop_loss}, tp={take_profit} \n \
      Итоговый баланс: {profile.tail(1)["balance"].values[0]} \n \
      Обернуто в сделках: {round((profile.head(1)["balance"].values[0] - profile["balance"].min()), 2)} \n \
      Рост обернутого: {"+" if delta_money > 0 else ""}{delta_money}% \n \
      Всего сделок: {np.floor(profile[profile["is_closed"] == 1]["is_closed"].count() / 2)} \n \
      Соотношение сделок (+/-): {round(profit["+"]/profit["-"], 2)} \n \
      take_profit: {stop_order["take_profit"]/2}  | |  stop_loss: {stop_order["stop_loss"]/2}  | |  expired: {stop_order["expired"]/2}\n")
  except:
    try:
      print(f"\
        Параметры: acc={accuracy}, sl={stop_loss}, tp={take_profit} \n \
        Итоговый баланс: {profile.tail(1)["balance"].values[0]} \n \
        Обернуто в сделках: {round((profile.head(1)["balance"].values[0] - profile["balance"].min()), 2)} \n \
        Рост обернутого: {"+" if delta_money > 0 else ""}{delta_money}% \n \
        Всего сделок: {np.floor(profile[profile["is_closed"] == 1]["is_closed"].count() / 2)} \n \
        Соотношение сделок (+/-): {round(profit["+"]/profit["-"], 2)} \n \
        take_profit: {stop_order["take_profit"]/2}  | |  stop_loss: {stop_order["stop_loss"]/2}  | |  expired: {0}\n")
    except:
      try:
        print(f"\
          Параметры: acc={accuracy}, sl={stop_loss}, tp={take_profit} \n \
          Итоговый баланс: {profile.tail(1)["balance"].values[0]} \n \
          Обернуто в сделках: {round((profile.head(1)["balance"].values[0] - profile["balance"].min()), 2)} \n \
          Рост обернутого: {"+" if delta_money > 0 else ""}{delta_money}% \n \
          Всего сделок: {np.floor(profile[profile["is_closed"] == 1]["is_closed"].count() / 2)} \n \
          Соотношение сделок (+/-): {round(profit["+"]/profit["-"], 2)} \n \
          take_profit: {0}  | |  stop_loss: {stop_order["stop_loss"]/2}  | |  expired: {stop_order["expired"]/2}\n")
      except:
        try:
          print(f"\
          Параметры: acc={accuracy}, sl={stop_loss}, tp={take_profit} \n \
          Итоговый баланс: {profile.tail(1)["balance"].values[0]} \n \
          Обернуто в сделках: {round((profile.head(1)["balance"].values[0] - profile["balance"].min()), 2)} \n \
          Рост обернутого: {"+" if delta_money > 0 else ""}{delta_money}% \n \
          Всего сделок: {np.floor(profile[profile["is_closed"] == 1]["is_closed"].count() / 2)} \n \
          Соотношение сделок (+/-): {round((stop_order["take_profit"]/2) / (stop_order["expired"]/2), 2)} \n \
          take_profit: {stop_order["take_profit"]/2}  | |  stop_loss: {0}  | |  expired: {stop_order["expired"]/2}\n")
        except:
          print(f"\
          Параметры: acc={accuracy}, sl={stop_loss}, tp={take_profit} \n \
          Итоговый баланс: {profile.tail(1)["balance"].values[0]} \n \
          Обернуто в сделках: {round((profile.head(1)["balance"].values[0] - profile["balance"].min()), 2)} \n \
          Рост обернутого: {"+" if delta_money > 0 else ""}{delta_money}% \n \
          Всего сделок: {np.floor(profile[profile["is_closed"] == 1]["is_closed"].count() / 2)} \n ")

  profile.to_csv('out.csv', index=False)
  if (debug):
    # profile.to_csv('out.csv', index=False)
    print(tabulate(profile.head(10), headers='keys', tablefmt='psql'))
    print(profile["is_closed"].value_counts())
    print(profile["result"].value_counts().sort_index())
    print(profile["cause"].value_counts().sort_index())
    print(profile["balance"].min())

  return profile
