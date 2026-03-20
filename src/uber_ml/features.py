from __future__ import annotations

import pandas as pd

from .config import (
    CLASSIFICATION_TARGET_CANDIDATES,
    CLUSTER_FEATURE_CANDIDATES,
    MODEL_CATEGORICAL_FEATURES,
    REGRESSION_TARGET_CANDIDATES,
)
from .data import detect_date_column


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    date_column = detect_date_column(featured)

    if not date_column:
        return featured

    date_series = featured[date_column]
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series, errors="coerce")

    featured["hour"] = date_series.dt.hour
    featured["day_of_week"] = date_series.dt.dayofweek
    featured["month"] = date_series.dt.month

    return featured


def select_regression_target(df: pd.DataFrame) -> str:
    for column in REGRESSION_TARGET_CANDIDATES:
        if column in df.columns:
            return column

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        raise ValueError("Nenhuma coluna numerica encontrada para regressao.")

    return numeric_columns[-1]


def select_numeric_features(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number").copy()
    if target_column and target_column in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_column])
    return numeric_df.dropna()


def build_model_matrix(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    base_df = df.dropna(subset=[target_column]).copy()

    numeric_df = base_df.select_dtypes(include="number").drop(columns=[target_column], errors="ignore")
    categorical_columns = [column for column in MODEL_CATEGORICAL_FEATURES if column in base_df.columns]
    categorical_df = pd.get_dummies(base_df[categorical_columns], dummy_na=True)

    feature_df = pd.concat([numeric_df, categorical_df], axis=1).dropna()
    target_series = base_df.loc[feature_df.index, target_column]

    return feature_df, target_series


def select_classification_target(df: pd.DataFrame) -> str:
    for column in CLASSIFICATION_TARGET_CANDIDATES:
        if column in df.columns:
            return column

    regression_target = select_regression_target(df)
    generated_target = "fare_category"

    if generated_target not in df.columns:
        valid_target = df[regression_target].dropna()
        if valid_target.nunique() < 3:
            raise ValueError(
                "Nao foi possivel gerar classes suficientes para classificacao a partir do alvo numerico."
            )

        quantile_labels = ["baixo", "medio", "alto"]
        categorized = pd.qcut(valid_target, q=3, labels=quantile_labels, duplicates="drop")
        df[generated_target] = categorized.astype("string")

    return generated_target


def select_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in CLUSTER_FEATURE_CANDIDATES if column in df.columns]

    if available:
        return df[available].select_dtypes(include="number").dropna()

    numeric_df = df.select_dtypes(include="number").dropna()
    if numeric_df.empty:
        raise ValueError("Nao ha colunas numericas suficientes para clusterizacao.")

    return numeric_df
