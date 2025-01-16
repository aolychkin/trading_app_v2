from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score

from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier


def create_model_SVC(type: str, X_train, y_train):  # Создание модели SVM
  if type == "fast":
    # Если нужно расчитать параллельно
    model = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', C=1000.0, gamma="scale", cache_size=500, class_weight="balanced", random_state=17, verbose=1), max_samples=0.5, n_estimators=10, random_state=17))
    model.fit(X_train, y_train)  # Обучение модели
    return model
  elif type == "xgbc":
    model = XGBClassifier(
        random_state=17, booster='gbtree', device='cuda', validate_parameters=False,
        eta=0.3, gamma=0.2, max_depth=100, min_child_weight=2,
        max_delta_step=0, subsample=1,
        sampling_method='uniform', tree_method='hist',
        colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1,
        reg_lambda=1, alpha=0, max_bin=256, num_parallel_tree=1
    )
    model.fit(X_train, y_train)  # Обучение модели
    return model
  elif type == "rf":
    # max_depth=10,
    model = RandomForestClassifier(
        random_state=17, verbose=3, n_jobs=-1, criterion='log_loss', n_estimators=100, class_weight='balanced',
        min_samples_split=2, max_features=1.0, ccp_alpha=0.0)
    model.fit(X_train, y_train)  # Обучение модели
    return model
  elif type == "lgbm":
    model = LGBMClassifier(
        boosting_type='gbdt', num_leaves=280, max_depth=-1, learning_rate=0.4, class_weight='balanced',
        min_child_samples=20, colsample_bytree=1, reg_alpha=0.5, min_data_in_leaf=100, num_iterations=100,
        random_state=17, verbose=-1, n_jobs=-1)
    model.fit(X_train, y_train)  # Обучение модели
    return model
    model = LGBMClassifier(random_state=17)
    param_grid = {
        'boosting_type': ['gbdt'],
        'num_leaves': list(range(100, 400, 50)),
        "max_depth": [-1],
        'learning_rate': [0.1, 0.3, 0.5],
        'class_weight': ['balanced'],
        'min_child_samples': [10, 20, 50],
        'reg_alpha': [0.1, 0.3, 0.5],
        'min_data_in_leaf': [30, 50, 100, 300],
        'colsample_bytree': [1],
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)  # Поиск лучших параметров модели
    print("Лучшие параметры: {}".format(grid_search.best_params_))
    return grid_search
  elif type == "tree":
    n_estimators = 10
    model = DecisionTreeClassifier(max_depth=5, max_features=5, random_state=17)
    model.fit(X_train, y_train)  # Обучение модели
    return model
  elif type == "forest":
    n_estimators = 10
    model = OneVsRestClassifier(BaggingClassifier(RandomForestClassifier(max_depth=10, max_features=6, n_estimators=100, n_jobs=-1, random_state=17), max_samples=1.0 / n_estimators, n_estimators=n_estimators, random_state=17))
    model.fit(X_train, y_train)  # Обучение модели
    return model
  elif type == "detect":
    # Если нужно определить наилучшее ядро
    model = SVC(random_state=17, verbose=1)
    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)  # Поиск лучших параметров модели
    print("Лучшие параметры: {}".format(grid_search.best_params_))
    return grid_search
  else:
    model = SVC(kernel='rbf', C=100, random_state=17, verbose=1)
    model.fit(X_train, y_train)  # Обучение модели
    return model


def model_score(model, X_test, y_test):
  # Предсказание классов для тестовых данных
  y_pred = model.predict(X_test)

  # Оценка производительности модели
  target_names = ['class 0', 'class 1']
  # target_names = ['class 0', 'class 1', 'class 2', 'class 3']
  accuracy = round(accuracy_score(y_test, y_pred), 4)
  report = classification_report(y_test, y_pred, target_names=target_names)

  print(f"Accuracy: {accuracy}")
  print(report)