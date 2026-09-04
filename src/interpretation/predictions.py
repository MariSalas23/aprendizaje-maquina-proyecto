"""
Predicciones utilizando el modelo final de CatBoost.

Este módulo NO entrena nuevamente el modelo.
Carga el modelo final previamente entrenado y genera
predicciones sobre el conjunto de prueba.

El modelo final corresponde al Escenario A:
todas las variables predictoras disponibles (21 variables).

El umbral utilizado corresponde al threshold seleccionado
durante la etapa de evaluación del modelo CatBoost.
"""

import pandas as pd
import joblib

from catboost import Pool

from config.config import (
    DATA_FILE,
    FINAL_MODEL_FILE,
    REPORTS_DIR,
    TARGET,
    BINARY_COLS,
    ORDINAL_COLS,
    NUMERIC_COLS,
)

from src.preparation.cleaning import clean_dataset
from src.preparation.data_partition import split_data
from src.preparation.preprocessing import prepare_catboost_data


# Threshold seleccionado durante la evaluación
# para CatBoost - Escenario A
FINAL_THRESHOLD = 0.51


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
    por el modelo final de CatBoost.

    Se excluye ID porque es únicamente un identificador.
    """

    return BINARY_COLS + ORDINAL_COLS + NUMERIC_COLS


def prepare_test_data():
    """
    Prepara el conjunto de prueba utilizando las mismas
    variables y transformación utilizadas durante el
    entrenamiento del modelo final.

    Corresponde al Escenario A de CatBoost:
    todas las variables predictoras.
    """

    # ---------------------------------------------------------
    # 1. Cargar datos originales
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_FILE)

    # ---------------------------------------------------------
    # 2. Limpiar datos
    # ---------------------------------------------------------

    df = clean_dataset(df)

    # ---------------------------------------------------------
    # 3. Obtener las 21 variables predictoras
    # ---------------------------------------------------------

    features = get_all_features()

    # Verificación para evitar utilizar ID
    if "ID" in features:
        raise ValueError(
            "ERROR: ID no debe utilizarse como predictor."
        )

    if len(features) != 21:
        raise ValueError(
            f"Se esperaban 21 variables predictoras, "
            f"pero se encontraron {len(features)}."
        )

    # ---------------------------------------------------------
    # 4. Crear la misma partición train/test
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        df=df,
        features=features,
        target=TARGET,
    )

    # ---------------------------------------------------------
    # 5. Preparar datos para CatBoost
    # ---------------------------------------------------------

    (
        X_train_catboost,
        X_test_catboost,
        categorical_features,
    ) = prepare_catboost_data(
        X_train,
        X_test,
        features,
    )

    print("\nDatos de prueba preparados correctamente.")

    print(
        f"Variables utilizadas: {len(features)}"
    )

    print(
        f"X_test: {X_test_catboost.shape}"
    )

    print(
        f"Variables categóricas: {categorical_features}"
    )

    print(
        f"Threshold final: {FINAL_THRESHOLD}"
    )

    return (
        X_test_catboost,
        y_test,
        categorical_features,
    )


def generate_predictions(
    model,
    X_test,
    categorical_features,
):
    """
    Genera probabilidades y predicciones utilizando
    el threshold seleccionado durante la evaluación.

    Se utiliza CatBoost Pool para garantizar que las
    variables categóricas sean interpretadas de la misma
    manera que durante el entrenamiento.
    """

    # Crear Pool con las variables categóricas
    test_pool = Pool(
        data=X_test,
        cat_features=categorical_features,
    )

    # Obtener probabilidad de clase 1
    probabilities = model.predict_proba(
        test_pool
    )[:, 1]

    # Aplicar threshold seleccionado
    predictions = (
        probabilities >= FINAL_THRESHOLD
    ).astype(int)

    results = X_test.copy()

    results["Probability_Class_1"] = probabilities
    results["Prediction"] = predictions

    return results


def save_predictions(results):
    """
    Guarda las predicciones del modelo final.
    """

    output_file = (
        REPORTS_DIR
        / "final_model"
        / "predictions.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nPredicciones guardadas en: {output_file}"
    )


def main():
    """
    Ejecuta el proceso completo de generación
    de predicciones del modelo final.
    """

    print(
        "\n=== PREDICCIONES DEL MODELO FINAL ==="
    )

    # ---------------------------------------------------------
    # 1. Cargar modelo final
    # ---------------------------------------------------------

    model = load_final_model()

    # ---------------------------------------------------------
    # 2. Preparar conjunto de prueba
    # ---------------------------------------------------------

    (
        X_test,
        y_test,
        categorical_features,
    ) = prepare_test_data()

    # ---------------------------------------------------------
    # 3. Generar predicciones
    # ---------------------------------------------------------

    results = generate_predictions(
        model,
        X_test,
        categorical_features,
    )

    # ---------------------------------------------------------
    # 4. Mostrar primeras predicciones
    # ---------------------------------------------------------

    print("\nPrimeras predicciones:")

    print(
        results[
            [
                "Probability_Class_1",
                "Prediction",
            ]
        ].head(10)
    )

    # ---------------------------------------------------------
    # 5. Mostrar distribución
    # ---------------------------------------------------------

    print(
        "\nDistribución de predicciones:"
    )

    print(
        results["Prediction"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # 6. Guardar resultados
    # ---------------------------------------------------------

    save_predictions(results)

    print(
        "\nPredicciones finalizadas correctamente."
    )


if __name__ == "__main__":
    main()