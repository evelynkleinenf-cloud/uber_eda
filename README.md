# Projeto Uber EDA

Este repositorio ficou organizado com apenas a estrutura final do projeto:

- `notebooks/uber_eda_ml_atualizado.ipynb`: notebook principal pronto para exploracao, regressao, classificacao e clusterizacao.
- `dados/ncr_ride_bookings.csv`: dataset usado no projeto.
- `src/uber_ml/`: modulos Python com limpeza, features e treino dos modelos.
- `requirements.txt`: dependencias do projeto.

## Estrutura

```text
uber_eda/
|-- dados/
|   |-- ncr_ride_bookings.csv
|-- notebooks/
|   |-- uber_eda_ml_atualizado.ipynb
|-- src/
|   |-- uber_ml/
|       |-- __init__.py
|       |-- config.py
|       |-- data.py
|       |-- features.py
|       |-- train_supervised.py
|       |-- train_classification.py
|       |-- train_unsupervised.py
|-- requirements.txt
|-- README.md
```

## O que o projeto faz

- EDA com graficos e analise descritiva.
- Regressao para prever `Booking Value`.
- Classificacao para prever `Booking Status`.
- Clusterizacao com KMeans e DBSCAN.

## Como rodar

```bash
pip install -r requirements.txt
python -m src.uber_ml.train_supervised
python -m src.uber_ml.train_classification
python -m src.uber_ml.train_unsupervised
```

O notebook tambem tenta localizar o CSV automaticamente no Kaggle ou em `dados/ncr_ride_bookings.csv`.
