from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .config import MODELS_DIR
from .data import basic_cleaning, load_data
from .features import add_time_features, build_model_matrix, select_regression_target


def train_models() -> dict[str, dict[str, float]]:
    df = load_data()
    df = basic_cleaning(df)
    df = add_time_features(df)

    target_column = select_regression_target(df)
    X, y = build_model_matrix(df, target_column)

    if X.empty:
        raise ValueError("Nao foi possivel montar as features para o treino supervisionado.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results[name] = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
        }

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    return results


def main() -> None:
    results = train_models()
    print("Resultados dos modelos supervisionados:")
    for model_name, metrics in results.items():
        print(
            f"- {model_name}: "
            f"MAE={metrics['mae']:.4f} | "
            f"R2={metrics['r2']:.4f}"
        )


if __name__ == "__main__":
    main()
