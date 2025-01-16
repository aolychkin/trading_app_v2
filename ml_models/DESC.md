# Основные модели
## main_model_1
Сложная модель, которая обучалась долго. Входные параметры для прогноза - рост на протяжении 20 баров по границе 0.16%
SVC(kernel='rbf', C=1000.0, gamma="scale", cache_size=500, class_weight="balanced", random_state=17, verbose=1), max_samples=1, n_estimators=100, random_state=17))
0.71 (72/69) test: 0.2 | head | estim: 100 | samples: 0.5 | OneVsOneClassifier | 0.7098711614840647


## main_model_2
Похожая модель по параметрам, но обучается проще и быстрее.
OneVsOneClassifier(BaggingClassifier(SVC(kernel='rbf', C=1000.0, gamma="scale", cache_size=500, class_weight="balanced", random_state=17, verbose=1), max_samples=0.5, n_estimators=10, random_state=17))
0.71 (72/69) test: 0.2 | head | estim: 10 | samples: 0.5

# Валидационные модели
## val_model_1
### 64/61 = 63
### 56/64 = 61 (без BaggingClassifier | max_depth=5, max_features=5)

## val_model_2 = LGBMClassifier
76/76  = 76
LGBMClassifier(random_state=17)

Base = 0.7616971810520198

#### num_leaves 
31: 0.7616971810520198
50: 0.7846556233653008
100: 0.8183667538506249
[WIN] 150: 0.8328974135425748

#### boosting_type
gbdt

#### max_depth
-1

#### learning_rate
0.7: 0.8040
[WIN] 0.5: 0.8098
0.3: 0.8045
0.1: 0.7619

#### min_child_samples
20: 0.8334
50: 0.8357
200: 0.84

#### colsample_bytree
1: 0.84

### best
0.8445
boosting_type='gbdt', num_leaves=200, max_depth=-1, learning_rate=0.5, class_weight='balanced',
        min_child_samples=100, colsample_bytree=1, random_state=17)

### после парамс
Лучшие параметры: {'boosting_type': 'gbdt', 'class_weight': 'balanced', 'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': -1, 'min_child_samples': 20, 'num_leaves': 280}
Accuracy: 0.8442

0.8459
0.8466

## 0.8473
boosting_type='gbdt', num_leaves=280, max_depth=-1, learning_rate=0.1, class_weight='balanced',
        min_child_samples=20, colsample_bytree=1, reg_alpha=0.5, min_data_in_leaf=100,


# Random forest
pre analisys: 0.6694
default = 0.8459


### criterion
entropy = 0.8547
log_loss = 0.8547


### max_features
log2 = 0.847

### min_samples_split
2 = 0.847
5 = 0.8413
1 = 

### max_features
6 = 0.8504
1.0 = 0.8529 !
0.8 = 0.8495


# XGBClassifier
0.8053

gamma=0.2: 0.8109

#### max_depth
100 = 0.8513

#### min_child_weight
1->2 = 0.8555


#### reg_lambda



1195it [00:00, 2135.52it/s] main_model_1 (lgbm)
140.57
          Параметры: acc=0.7, sl=0.016, tp=0.002 
           Итоговый баланс: 703.26 
           Обернуто в сделках: 559.43 
           Рост обернутого: +0.58% 
           Всего сделок: 39.0 
           Соотношение сделок (+/-): 100% 
           take_profit: 29.0  | |  stop_loss: 0  | |  expired: 10.0


1195it [00:00, 2053.30it/s] val_model_1 (xgbc)
139.22
          Параметры: acc=0.7, sl=0.016, tp=0.002 
           Итоговый баланс: 703.52 
           Обернуто в сделках: 560.78 
           Рост обернутого: +0.63% 
           Всего сделок: 36.0 
           Соотношение сделок (+/-): 100% 
           take_profit: 30.0  | |  stop_loss: 0  | |  expired: 6.0





  # profile = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.8, stop_loss=0.016, take_profit=0.003, target="target")
  # profile = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.8, stop_loss=0.016, take_profit=0.005, target="target")
  # profile = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.8, stop_loss=0.01, take_profit=0.005, target="target")

  # profile = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.8, stop_loss=0.016, take_profit=0.002, target="target",
  #                             val_accuracy=0.6, val_max_accuracy=0.8, val_target="val_target")  # 699.46
  # profile = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.8, stop_loss=0.016, take_profit=0.002, target="target",
  #                             is_val=True, val_accuracy=0.5, val_max_accuracy=0.8, val_target="val_target")  # 699.46

  # profile = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.8, stop_loss=0.016, take_profit=0.002, target="target",
  #                             is_val=True, val_accuracy=0.6, val_max_accuracy=0.8, val_target="val_target")  # TOP

  # profile = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.8, stop_loss=0.016, take_profit=0.002, target="target")
  # profile = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.8, stop_loss=0.016, take_profit=0.003, target="target")
  # profile = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.8, stop_loss=0.016, take_profit=0.005, target="target")
  # profile = simulate.strategy(df_model, accuracy=0.7, max_accuracy=0.8, stop_loss=0.01, take_profit=0.005, target="target")