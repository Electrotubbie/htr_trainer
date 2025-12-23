# htr_trainer

Данные скрипты используются для запуска пайплайна дообучения HTR модели на архитектуре TrOCR.

# Описание файлов

- [./bash_scripts/prepare_server.sh](./bash_scripts/prepare_server.sh) - скрипт подготовки сервера к запуску пайплайна
- [./train.sh](./train.sh) - bash скрипт запуска пайплайна дообучения, в котором регулируются параметры пайплайна
- [./train.py](./train.py) - python скрипт подготовки датасета и запуска обучения на параметрах, указанных в [./train.sh](./train.sh)
- [./scripts/args.py](./scripts/args.py) - python скрипт парсинга параметров из команды запуска в [./train.sh](./train.sh)
- [./scripts/train_funcs.py](./scripts/train_funcs.py) - основные функции для запуска экспериментов

# Запуск

1) Подготовьте файлы к запуску командой `chmod -R u+x <filename>`
2) Запустите [./bash_scripts/prepare_server.sh](./bash_scripts/prepare_server.sh) для установки нужной версии python, развёртывания venv и установки необходимых пакетов
3) Пропишите параметры обучения в файле [./train.sh](./train.sh)
4) Запустите [./train.sh](./train.sh)
5) Результаты сохраняются в директории `./experiments_hist/` (история обучения по эпохам) и в `./htr_experiments/` сохраняются веса лучшей по метрике CER модели
