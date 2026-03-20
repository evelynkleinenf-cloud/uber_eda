# Projeto Uber: EDA + Machine Learning

Este repositorio agora esta alinhado com a estrutura do GitHub:

- `notebooks/uber_eda_ml_atualizado.ipynb`: notebook principal para exploracao e modelos.
- `dados/ncr_ride_bookings.csv`: dataset usado no projeto.
- `src/uber_ml/`: codigo reutilizavel de preparacao, treino e clusterizacao.
- `data/raw/`: copia local para execucao dos scripts em Python.
- `data/processed/`: saídas intermediárias, se você quiser salvar dados tratados.
- `models/`: modelos treinados com `joblib`.
- `reports/figures/`: gráficos exportados.

## Estrutura sugerida

```text
New project/
|-- dados/
|   |-- ncr_ride_bookings.csv
|-- notebooks/
|   |-- uber_eda_ml_atualizado.ipynb
|-- requirements.txt
|-- README.md
|-- data/
|   |-- raw/
|   |-- processed/
|-- models/
|-- reports/
|   |-- figures/
|-- src/
|   |-- uber_ml/
|       |-- __init__.py
|       |-- config.py
|       |-- data.py
|       |-- features.py
|       |-- train_supervised.py
|       |-- train_unsupervised.py
```

## Como organizar o que voce aprendeu

### 1. Notebook para estudo e comunicacao

No notebook, vale manter:

- carregamento dos dados;
- limpeza inicial;
- EDA;
- testes rapidos de ideias;
- interpretacao dos graficos;
- comparacao visual dos resultados.

### 2. Scripts em `src/` para codigo reaproveitavel

No `src/uber_ml/`, o ideal e separar por responsabilidade:

- `data.py`: leitura e limpeza basica;
- `features.py`: criacao de colunas como hora, dia, mes;
- `train_supervised.py`: regressao linear e random forest;
- `train_unsupervised.py`: KMeans e DBSCAN;
- `config.py`: nomes padrao de colunas e caminhos.

Assim, voce evita deixar toda a logica presa no notebook.

## Modelos incluidos

### Supervisionados

- Regressao linear para prever uma coluna alvo numerica.
- Random Forest Regressor para capturar relacoes nao lineares.
- Regressao logistica e Random Forest Classifier para classificar corridas em faixas, como baixo, medio e alto valor.

### Nao supervisionados

- KMeans para segmentar grupos parecidos de corridas.
- DBSCAN para encontrar agrupamentos por densidade e detectar outliers.

## Passos para usar

1. Instale as dependencias:

```bash
pip install -r requirements.txt
```

2. Coloque seu dataset em `data/raw/ncr_ride_bookings.csv`.

3. Rode o treino supervisionado:

```bash
python -m src.uber_ml.train_supervised
```

4. Rode a classificacao:

```bash
python -m src.uber_ml.train_classification
```

5. Rode a clusterizacao:

```bash
python -m src.uber_ml.train_unsupervised
```

## Como adaptar ao seu dataset

Para este dataset, o projeto usa por padrao:

- `Date` e `Time` para gerar `datetime`, `hour`, `day_of_week` e `month`;
- `Booking Value` como alvo de regressao;
- `Booking Status` como alvo de classificacao;
- colunas como `Vehicle Type`, `Payment Method`, `Pickup Location` e `Drop Location` como categorias codificadas.

Se quiser reaproveitar a estrutura para outro CSV, ajuste as constantes em `src/uber_ml/config.py`.

## Ideia de evolucao no notebook

Depois da EDA, voce pode adicionar secoes como:

- `11. Regressao Linear`
- `12. Random Forest`
- `13. KMeans`
- `14. DBSCAN`
- `15. Comparacao e conclusoes`

Isso deixa o projeto com cara de portfolio: com exploracao, modelagem e interpretacao.
