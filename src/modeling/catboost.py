"""
Entrenamiento de modelos CatBoost para los escenarios A y B.

Escenario A:
    Todas las 21 variables predictoras.

Escenario B:
    Las 12 variables seleccionadas a partir del análisis exploratorio.

CatBoost recibe las variables categóricas como texto (str).
"""

from catboost import CatBoostClassifier

from config.config import RANDOM_STATE


# ============================================================
# CONSTRUCCIÓN DEL MODELO
# ============================================================

def build_catboost_model():
    """
    Construye el modelo CatBoost con los hiperparámetros
    utilizados en el proyecto.
    """

    model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1
    )

    return model


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train_catboost_model(
    X_train,
    y_train,
    categorical_features
):
    """
    Entrena un modelo CatBoost.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Variables predictoras de entrenamiento.

    y_train : pandas.Series
        Variable objetivo.

    categorical_features : list
        Lista de nombres de las variables categóricas.

    Returns
    -------
    model : CatBoostClassifier
        Modelo CatBoost entrenado.
    """

    model = build_catboost_model()

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features
    )

    return model


# ============================================================
# PREDICCIÓN
# ============================================================

def predict_catboost_model(
    model,
    X
):
    """
    Genera predicciones y probabilidades para clase 1.

    Returns
    -------
    predictions : array
        Predicciones de clase.

    probabilities : array
        Probabilidades estimadas para la clase 1.
    """

    predictions = (
        model
        .predict(X)
        .ravel()
    )

    probabilities = (
        model
        .predict_proba(X)[:, 1]
    )

    return (
        predictions,
        probabilities
    )


# ============================================================
# ESCENARIOS A Y B
# ============================================================

def train_catboost_scenarios(
    partition,
    categorical_features_all,
    categorical_features_selected
):
    """
    Entrena CatBoost para los escenarios A y B.

    Escenario A:
        Todas las variables predictoras.

    Escenario B:
        Variables seleccionadas.

    Returns
    -------
    models : dict
        Diccionario con los modelos entrenados.
    """

    # --------------------------------------------------------
    # Escenario A
    # --------------------------------------------------------

    scenario_a = partition[
        "A_todas_las_variables"
    ]

    model_a = train_catboost_model(
        scenario_a["X_train"],
        scenario_a["y_train"],
        categorical_features_all
    )

    # --------------------------------------------------------
    # Escenario B
    # --------------------------------------------------------

    scenario_b = partition[
        "B_variables_seleccionadas"
    ]

    model_b = train_catboost_model(
        scenario_b["X_train"],
        scenario_b["y_train"],
        categorical_features_selected
    )

    models = {
        "A_todas_las_variables": model_a,
        "B_variables_seleccionadas": model_b
    }

    return models


# ============================================================
# PREDICCIÓN DE LOS DOS ESCENARIOS
# ============================================================

def predict_catboost_scenarios(
    models,
    partition,
    X_test_all,
    X_test_selected
):
    """
    Genera predicciones para los escenarios A y B.

    Parameters
    ----------
    models : dict
        Modelos CatBoost entrenados.

    partition : dict
        Particiones de los datos.

    X_test_all : pandas.DataFrame
        Datos de prueba del escenario A preparados para CatBoost.

    X_test_selected : pandas.DataFrame
        Datos de prueba del escenario B preparados para CatBoost.
    """

    predictions = {}

    predictions[
        "A_todas_las_variables"
    ] = predict_catboost_model(
        models[
            "A_todas_las_variables"
        ],
        X_test_all
    )

    predictions[
        "B_variables_seleccionadas"
    ] = predict_catboost_model(
        models[
            "B_variables_seleccionadas"
        ],
        X_test_selected
    )

    return predictions


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    from src.preparation.cleaning import (
        prepare_clean_dataset
    )

    from src.preparation.feature_selection import (
        get_all_features,
        get_selected_features
    )

    from src.preparation.data_partition import (
        create_partition_scenarios
    )

    from src.preparation.preprocessing import (
        prepare_catboost_data
    )

    # --------------------------------------------------------
    # 1. Dataset limpio
    # --------------------------------------------------------

    df = prepare_clean_dataset()

    # --------------------------------------------------------
    # 2. Variables de los escenarios
    # --------------------------------------------------------

    all_features = (
        get_all_features()
    )

    selected_features = (
        get_selected_features()
    )

    print(
        "\n=== 4.1 CATBOOST ==="
    )

    print(
        "\nEscenario A:"
    )

    print(
        f"Variables: "
        f"{len(all_features)}"
    )

    print(
        "\nEscenario B:"
    )

    print(
        f"Variables: "
        f"{len(selected_features)}"
    )

    # --------------------------------------------------------
    # 3. Partición train/test
    # --------------------------------------------------------

    partition = (
        create_partition_scenarios(
            df,
            all_features,
            selected_features
        )
    )

    # --------------------------------------------------------
    # 4. Preparación de datos para CatBoost
    # --------------------------------------------------------

    scenario_a = partition[
        "A_todas_las_variables"
    ]

    (
        X_train_all,
        X_test_all,
        categorical_features_all
    ) = prepare_catboost_data(
        scenario_a["X_train"],
        scenario_a["X_test"],
        all_features
    )

    scenario_b = partition[
        "B_variables_seleccionadas"
    ]

    (
        X_train_selected,
        X_test_selected,
        categorical_features_selected
    ) = prepare_catboost_data(
        scenario_b["X_train"],
        scenario_b["X_test"],
        selected_features
    )

    # --------------------------------------------------------
    # 5. Actualizar las particiones con los datos preparados
    # --------------------------------------------------------

    partition[
        "A_todas_las_variables"
    ]["X_train"] = X_train_all

    partition[
        "A_todas_las_variables"
    ]["X_test"] = X_test_all

    partition[
        "B_variables_seleccionadas"
    ]["X_train"] = X_train_selected

    partition[
        "B_variables_seleccionadas"
    ]["X_test"] = X_test_selected

    # --------------------------------------------------------
    # 6. Mostrar variables categóricas
    # --------------------------------------------------------

    print(
        "\nVariables categóricas - Escenario A:"
    )

    print(
        categorical_features_all
    )

    print(
        "\nVariables categóricas - Escenario B:"
    )

    print(
        categorical_features_selected
    )

    # --------------------------------------------------------
    # 7. Entrenamiento
    # --------------------------------------------------------

    models = train_catboost_scenarios(
        partition,
        categorical_features_all,
        categorical_features_selected
    )

    # --------------------------------------------------------
    # 8. Confirmación
    # --------------------------------------------------------

    print(
        "\nEscenario A:"
    )

    print(
        f"Variables utilizadas: "
        f"{len(all_features)}"
    )

    print(
        "Modelo entrenado correctamente."
    )

    print(
        "\nEscenario B:"
    )

    print(
        f"Variables utilizadas: "
        f"{len(selected_features)}"
    )

    print(
        "Modelo entrenado correctamente."
    )

    print(
        "\nProceso de entrenamiento finalizado."
    )