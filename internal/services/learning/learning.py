from datetime import datetime
from tabulate import tabulate

import sqlite3
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

import internal.services.learning.ml_model as mlm
import internal.services.learning.preprocessing as prep
import internal.services.learning.prediction as predict

# классификация
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import accuracy_score as acc

if __name__ == '__main__':
  df_predict = predict.make_prediction()
  df_normal_data = prep.make_normal_data()
  df_normal_data = prep.get_normal_data()

  df = df_predict.merge(df_normal_data, on='candle_id')
  df = df.groupby("class").head(int(df["class"].value_counts().values[-1]))
  df_X = df[df.columns.values[2:]]
  df_y = df[df.columns.values[1:2]]

  X = df_X.to_numpy()
  y = df_y.to_numpy()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

  model = mlm.create_model_SVC("xgbc", X_train, y_train.ravel())

  # Оценка производительности модели
  mlm.model_score(model, X_test, y_test)

  # Сохранение модели
  # main: lgbm = 0.8446
  # validation: xgbc=0.8485
  joblib.dump(model, "./ml_models/val_model_4.pkl")
