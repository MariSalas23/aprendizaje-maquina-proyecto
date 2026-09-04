"""
Interpretación del modelo final de CatBoost.

Este módulo NO entrena nuevamente el modelo.
Carga el modelo final previamente entrenado y guardado en models/
y obtiene la importancia de las variables utilizadas por CatBoost.
"""

import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from config.config import (
    FINAL_MODEL_FILE,
    FINAL_MODEL_FIGURES_DIR,
    REPORTS_DIR,
    BINARY_COLS,
    ORDINAL_COLS,
    NUMERIC_COLS,
)


def load_final_model():
    """
    Carga el modelo final de CatBoost previamente entrenado.
    """

    if not FINAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo final en: {FINAL_MODEL_FILE}\n"
            "Primero debe guardarse el modelo final de CatBoost."
        )

    model = joblib.load(FINAL_MODEL_FILE)

    print("Modelo final cargado correctamente.")
    print(f"Ruta: {FINAL_MODEL_FILE}")

    return model


def get_all_features():
    """
    Obtiene las 21 variables predictoras utilizadas
    por el escenario A de CatBoost.

    Se excluye la variable ID porque no es un predictor.
    """

    return BINARY_COLS + ORDINAL_COLS + NUMERIC_COLS


def get_feature_importance(model):
    """
    Obtiene y ordena la importancia de las variables
    del modelo final de CatBoost.
    """

    features = get_all_features()

    importance_values = model.get_feature_importance()

    if len(features) != len(importance_values):
        raise ValueError(
            "El número de variables no coincide con el número "
            "de importancias proporcionadas por CatBoost."
        )

    feature_importance = pd.DataFrame(
        {
            "Variable": features,
            "Importancia": importance_values,
        }
    )

    feature_importance = feature_importance.sort_values(
        by="Importancia",
        ascending=False,
    ).reset_index(drop=True)

    return feature_importance


def save_feature_importance(feature_importance):
    """
    Guarda la importancia de las variables en formato JSON.
    """

    output_file = (
        REPORTS_DIR
        / "final_model"
        / "feature_importance.json"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = feature_importance.to_dict(
        orient="records"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"\nResultados guardados en: {output_file}"
    )


def plot_feature_importance(feature_importance):
    """
    Genera la gráfica de importancia de variables.
    """

    FINAL_MODEL_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 8))

    sns.barplot(
        data=feature_importance,
        x="Importancia",
        y="Variable",
    )

    plt.title(
        "Importancia de variables - CatBoost final"
    )

    plt.xlabel("Importancia")
    plt.ylabel("Variable")

    plt.tight_layout()

    output_file = (
        FINAL_MODEL_FIGURES_DIR
        / "feature_importance_catboost.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Gráfica guardada en: {output_file}"
    )


def main():
    """
    Ejecuta el proceso completo de interpretación.
    """

    print(
        "\n=== INTERPRETACIÓN DEL MODELO FINAL ==="
    )

    # 1. Cargar modelo ya entrenado
    model = load_final_model()

    # 2. Obtener importancia
    feature_importance = get_feature_importance(
        model
    )

    # 3. Mostrar resultados
    print(
        "\nImportancia de variables:"
    )

    print(
        feature_importance.to_string(
            index=False
        )
    )

    # 4. Guardar resultados
    save_feature_importance(
        feature_importance
    )

    # 5. Generar gráfica
    plot_feature_importance(
        feature_importance
    )

    print(
        "\nInterpretación finalizada correctamente."
    )


if __name__ == "__main__":
    main()