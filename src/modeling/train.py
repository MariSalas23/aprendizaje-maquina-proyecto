"""
Entrenamiento del modelo final del proyecto.

Este módulo utiliza el modelo CatBoost ya construido en
catboost.py para entrenar el modelo final con todas las
variables predictoras.

El modelo final corresponde al Escenario A:
todas las variables predictoras disponibles (21 variables).

La variable ID se excluye porque es únicamente un identificador
de las observaciones y no constituye una variable predictora.

El modelo entrenado se guarda en models/ para ser utilizado
posteriormente por los módulos de interpretación y predicción.
"""

import joblib

from config.config import (
    DATA_FILE,
    FINAL_MODEL_FILE,
    TARGET,
)

from src.data.data_loader import load_data

from src.preparation.cleaning import clean_dataset

from src.preparation.data_partition import split_data

from src.preparation.preprocessing import prepare_catboost_data

from src.modeling.catboost import train_catboost_model


def get_all_features(df):
    """
    Obtiene todas las variables predictoras disponibles.

    Se excluyen:
    - TARGET: variable objetivo.
    - ID: identificador de la observación.

    Returns
    -------
    list
        Lista con las 21 variables predictoras.
    """

    features = [
        column
        for column in df.columns
        if column not in [TARGET, "ID"]
    ]

    return features


def train_final_model():
    """
    Entrena el modelo final de CatBoost utilizando
    todas las variables predictoras del Escenario A.
    """

    print("\n=== ENTRENAMIENTO DEL MODELO FINAL ===")
    print("Modelo: CatBoost")
    print("Escenario: A - Todas las variables")

    # ---------------------------------------------------------
    # 1. Cargar datos
    # ---------------------------------------------------------

    df = load_data(DATA_FILE)

    print(
        f"\nObservaciones cargadas: {len(df):,}"
    )

    print(
        f"Variables cargadas: {df.shape[1]}"
    )

    # ---------------------------------------------------------
    # 2. Limpieza
    # ---------------------------------------------------------

    df = clean_dataset(df)

    # ---------------------------------------------------------
    # 3. Definir variables predictoras
    # ---------------------------------------------------------

    features = get_all_features(df)

    print(
        f"\nVariables predictoras utilizadas: {len(features)}"
    )

    print(features)

    # Verificación para asegurar que ID no entre al modelo
    if "ID" in features:

        raise ValueError(
            "ERROR: la variable ID no debe utilizarse "
            "como predictor."
        )

    if len(features) != 21:

        raise ValueError(
            f"Se esperaban 21 variables predictoras, "
            f"pero se encontraron {len(features)}."
        )

    # ---------------------------------------------------------
    # 4. Partición train/test
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        df=df,
        features=features,
        target=TARGET,
    )

    print("\nPartición realizada:")

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"y_test:  {y_test.shape}"
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

    print(
        "\nDatos preparados para CatBoost."
    )

    print(
        f"Variables categóricas: {categorical_features}"
    )

    # ---------------------------------------------------------
    # 6. Entrenar CatBoost
    # ---------------------------------------------------------

    model = train_catboost_model(
        X_train_catboost,
        y_train,
        categorical_features,
    )

    print(
        "\nModelo CatBoost entrenado correctamente."
    )

    # ---------------------------------------------------------
    # 7. Guardar modelo final
    # ---------------------------------------------------------

    FINAL_MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        FINAL_MODEL_FILE,
    )

    print(
        "\nModelo final guardado en:"
    )

    print(
        FINAL_MODEL_FILE
    )

    return model


if __name__ == "__main__":
    train_final_model()