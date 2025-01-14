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
### 
