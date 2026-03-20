from __future__ import annotations

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import MODELS_DIR
from .data import basic_cleaning, load_data
from .features import add_time_features, build_model_matrix, select_classification_target


def train_models() -> dict[str, dict[str, float | str]]:
    df = load_data()
    df = basic_cleaning(df)
    df = add_time_features(df)

    target_column = select_classification_target(df)
    X, y = build_model_matrix(df, target_column)
    y = y.astype(str)

    if X.empty:
        raise ValueError("Nao foi possivel montar as features para o treino de classificacao.")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded if len(set(y_encoded)) > 1 else None,
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float | str]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro")),
            "report": classification_report(
                y_test,
                predictions,
                target_names=encoder.classes_,
                zero_division=0,
            ),
        }

        artifact = {
            "model": model,
            "label_encoder": encoder,
            "target_column": target_column,
        }
        joblib.dump(artifact, MODELS_DIR / f"{name}.joblib")

    return results


def main() -> None:
    results = train_models()
    print("Resultados dos modelos de classificacao:")
    for model_name, metrics in results.items():
        print(
            f"- {model_name}: "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )
        print(metrics["report"])


if __name__ == "__main__":
    main()
