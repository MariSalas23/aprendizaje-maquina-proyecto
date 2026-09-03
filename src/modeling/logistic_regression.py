"""
Entrenamiento de modelos Logistic Regression
para los escenarios A y B.

Escenario A:
    Todas las 21 variables predictoras.

Escenario B:
    Las 12 variables seleccionadas a partir
    del análisis exploratorio.

El preprocesamiento se realiza mediante las
funciones definidas en src/preparation/preprocessing.py.
"""

from sklearn.linear_model import LogisticRegression

from config.config import RANDOM_STATE

from src.preparation.preprocessing import (
    fit_transform_train_test,
)


# ============================================================
# CONSTRUCCIÓN DEL MODELO
# ============================================================

def build_logistic_model():
    """
    Construye el modelo Logistic Regression
    utilizado en el proyecto.
    """

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    return model


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train_logistic_model(
    X_train,
    y_train,
):
    """
    Entrena un modelo Logistic Regression.

    Parameters
    ----------
    X_train : array-like
        Variables predictoras de entrenamiento
        después del preprocesamiento.

    y_train : pandas.Series
        Variable objetivo.

    Returns
    -------
    model : LogisticRegression
        Modelo entrenado.
    """

    model = build_logistic_model()

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# PREDICCIÓN
# ============================================================

def predict_logistic_model(
    model,
    X,
):
    """
    Genera predicciones y probabilidades
    para clase 1.
    """

    predictions = (
        model
        .predict(X)
    )

    probabilities = (
        model
        .predict_proba(X)[:, 1]
    )

    return (
        predictions,
        probabilities,
    )


# ============================================================
# ENTRENAMIENTO DE UN ESCENARIO
# ============================================================

def train_logistic_scenario(
    X_train,
    X_test,
    y_train,
    features,
):
    """
    Ajusta el preprocesador, transforma train/test
    y entrena Logistic Regression.
    """

    (
        preprocessor,
        X_train_transformed,
        X_test_transformed,
    ) = fit_transform_train_test(
        X_train,
        X_test,
        features,
        model_type="logistic",
    )

    model = train_logistic_model(
        X_train_transformed,
        y_train,
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "X_train_transformed": X_train_transformed,
        "X_test_transformed": X_test_transformed,
    }


# ============================================================
# ENTRENAMIENTO ESCENARIOS A Y B
# ============================================================

def train_logistic_scenarios(
    partition,
    all_features,
    selected_features,
):
    """
    Entrena Logistic Regression para los escenarios A y B.
    """

    # --------------------------------------------------------
    # Escenario A
    # --------------------------------------------------------

    scenario_a = partition[
        "A_todas_las_variables"
    ]

    results_a = train_logistic_scenario(
        X_train=scenario_a["X_train"],
        X_test=scenario_a["X_test"],
        y_train=scenario_a["y_train"],
        features=all_features,
    )

    # --------------------------------------------------------
    # Escenario B
    # --------------------------------------------------------

    scenario_b = partition[
        "B_variables_seleccionadas"
    ]

    results_b = train_logistic_scenario(
        X_train=scenario_b["X_train"],
        X_test=scenario_b["X_test"],
        y_train=scenario_b["y_train"],
        features=selected_features,
    )

    return {
        "A_todas_las_variables": results_a,
        "B_variables_seleccionadas": results_b,
    }


# ============================================================
# PREDICCIÓN ESCENARIOS A Y B
# ============================================================

def predict_logistic_scenarios(
    models,
):
    """
    Genera predicciones para ambos escenarios.
    """

    predictions = {}

    for scenario_name, scenario in models.items():

        (
            scenario_predictions,
            scenario_probabilities,
        ) = predict_logistic_model(
            scenario["model"],
            scenario["X_test_transformed"],
        )

        predictions[
            scenario_name
        ] = {
            "predictions": scenario_predictions,
            "probabilities": scenario_probabilities,
        }

    return predictions


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    from src.preparation.cleaning import (
        prepare_clean_dataset,
    )

    from src.preparation.feature_selection import (
        get_all_features,
        get_selected_features,
    )

    from src.preparation.data_partition import (
        create_partition_scenarios,
    )

    # --------------------------------------------------------
    # 1. Dataset limpio
    # --------------------------------------------------------

    df = prepare_clean_dataset()

    # --------------------------------------------------------
    # 2. Variables
    # --------------------------------------------------------

    all_features = (
        get_all_features()
    )

    selected_features = (
        get_selected_features()
    )

    print(
        "\n=== 4.2 LOGISTIC REGRESSION ==="
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
    # 3. Partición
    # --------------------------------------------------------

    partition = (
        create_partition_scenarios(
            df,
            all_features,
            selected_features,
        )
    )

    # --------------------------------------------------------
    # 4. Entrenamiento
    # --------------------------------------------------------

    models = train_logistic_scenarios(
        partition,
        all_features,
        selected_features,
    )

    # --------------------------------------------------------
    # 5. Predicciones
    # --------------------------------------------------------

    predictions = (
        predict_logistic_scenarios(
            models
        )
    )

    # --------------------------------------------------------
    # 6. Confirmación
    # --------------------------------------------------------

    for scenario_name, scenario in models.items():

        print(
            f"\n{scenario_name}:"
        )

        print(
            "Modelo entrenado correctamente."
        )

        print(
            "Train transformado: "
            f"{scenario['X_train_transformed'].shape}"
        )

        print(
            "Test transformado: "
            f"{scenario['X_test_transformed'].shape}"
        )

    print(
        "\nPredicciones generadas correctamente."
    )

    print(
        "\nProceso de Logistic Regression finalizado."
    )