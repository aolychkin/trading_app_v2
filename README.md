# trading_app_v2
## Инструкция по гиту (Указан тег уже существующего коммита)
git add *
git commit -m ""
git tag v1.1.0
git branch -M main
git remote add origin https://github.com/aolychkin/trading_app_v2.git
git push -u origin main --tags

## Экспортировать Path для импорта библиотек
export PYTHONPATH="${PYTHONPATH}:/Desktop/trading_app_v2/"