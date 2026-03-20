from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATE_CANDIDATES, RAW_DATA_PATH


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset nao encontrado em {path}. Coloque o arquivo uber.csv em data/raw/."
        )

    return pd.read_csv(path)


def detect_date_column(df: pd.DataFrame) -> str | None:
    for column in DATE_CANDIDATES:
        if column in df.columns:
            return column
    return None


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()

    if "Booking ID" in cleaned.columns:
        cleaned["Booking ID"] = cleaned["Booking ID"].astype(str).str.replace('"', "", regex=False)

    if "Customer ID" in cleaned.columns:
        cleaned["Customer ID"] = cleaned["Customer ID"].astype(str).str.replace('"', "", regex=False)

    if "Date" in cleaned.columns and "Time" in cleaned.columns:
        cleaned["datetime"] = pd.to_datetime(
            cleaned["Date"].astype(str) + " " + cleaned["Time"].astype(str),
            errors="coerce",
        )

    date_column = detect_date_column(cleaned)
    if date_column:
        cleaned[date_column] = pd.to_datetime(cleaned[date_column], errors="coerce")

    return cleaned
