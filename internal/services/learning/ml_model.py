from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from lightgbm import LGBMClassifier


def create_model_SVC(type: str, X_train, y_train):  # Создание модели SVM
  if type == "fast":
    # Если нужно расчитать параллельно
    model = OneVsOneClassifier(BaggingClassifier(SVC(kernel='rbf', C=1000.0, gamma="scale", cache_size=500, class_weight="balanced", random_state=17, verbose=1), max_samples=0.5, n_estimators=10, random_state=17))
    model.fit(X_train, y_train)  # Обучение модели
    return model
  elif type == "lgbm":
    model = LGBMClassifier(random_state=17)
    model.fit(X_train, y_train)  # Обучение модели
    return model
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
  accuracy = accuracy_score(y_test, y_pred)
  report = classification_report(y_test, y_pred, target_names=target_names)

  print(f"Accuracy: {accuracy}")
  print(report)
