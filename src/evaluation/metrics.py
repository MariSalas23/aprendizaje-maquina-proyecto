"""
Evaluación de los modelos Logistic Regression y CatBoost.

Se evalúan los cuatro escenarios del proyecto:

    1. Logistic Regression - Escenario A
       Todas las 21 variables.

    2. Logistic Regression - Escenario B
       12 variables seleccionadas.

    3. CatBoost - Escenario A
       Todas las 21 variables.

    4. CatBoost - Escenario B
       12 variables seleccionadas.

Las métricas se calculan sobre el conjunto de prueba.

Los resultados generales se guardan en:

    reports/comparison/metrics_all_models.json

Las métricas del modelo final se guardan en:

    reports/final_model/final_metrics.json

El modelo final corresponde a:

    CatBoost - Escenario A
    Todas las 21 variables predictoras.
"""

import json

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
)

from config.config import (
    REPORTS_DIR,
)

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

from src.preparation.preprocessing import (
    fit_transform_train_test,
    prepare_catboost_data,
)

from src.modeling.logistic_regression import (
    train_logistic_model,
)

from src.modeling.catboost import (
    train_catboost_model,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

COMPARISON_DIR = (
    REPORTS_DIR
    / "comparison"
)

METRICS_FILE = (
    COMPARISON_DIR
    / "metrics_all_models.json"
)

FINAL_MODEL_DIR = (
    REPORTS_DIR
    / "final_model"
)

FINAL_METRICS_FILE = (
    FINAL_MODEL_DIR
    / "final_metrics.json"
)

# Umbral utilizado actualmente para Logistic Regression.
LOGISTIC_THRESHOLD = 0.50

# Umbral seleccionado mediante el análisis OOF de CatBoost.
CATBOOST_THRESHOLD = 0.51


# ============================================================
# CÁLCULO DE MÉTRICAS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    """
    Calcula las métricas de evaluación utilizando
    un umbral específico.

    Parameters
    ----------
    y_true : pandas.Series
        Valores reales de la variable objetivo.

    probabilities : array
        Probabilidades estimadas para clase 1.

    threshold : float
        Umbral utilizado para convertir probabilidades
        en predicciones de clase.

    Returns
    -------
    dict
        Métricas calculadas.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "threshold": float(
            threshold
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision_class_0": float(
            precision_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),

        "recall_class_0": float(
            recall_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),

        "f1_class_0": float(
            f1_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),

        "precision_class_1": float(
            precision_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "recall_class_1": float(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "f1_class_1": float(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "f2_class_1": float(
            fbeta_score(
                y_true,
                predictions,
                beta=2,
                pos_label=1,
                zero_division=0,
            )
        ),

        "f1_macro": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_weighted": float(
            f1_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "pr_auc_class_1": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
    }

    return metrics


# ============================================================
# EVALUACIÓN LOGISTIC REGRESSION
# ============================================================

def evaluate_logistic_scenario(
    scenario_data,
    features,
):
    """
    Entrena y evalúa Logistic Regression
    para un escenario.
    """

    (
        preprocessor,
        X_train_transformed,
        X_test_transformed,
    ) = fit_transform_train_test(
        scenario_data["X_train"],
        scenario_data["X_test"],
        features,
        model_type="logistic",
    )

    model = train_logistic_model(
        X_train_transformed,
        scenario_data["y_train"],
    )

    probabilities = (
        model
        .predict_proba(
            X_test_transformed
        )[:, 1]
    )

    metrics = calculate_metrics(
        y_true=scenario_data["y_test"],
        probabilities=probabilities,
        threshold=LOGISTIC_THRESHOLD,
    )

    return metrics


# ============================================================
# EVALUACIÓN CATBOOST
# ============================================================

def evaluate_catboost_scenario(
    scenario_data,
    features,
):
    """
    Prepara, entrena y evalúa CatBoost
    para un escenario.
    """

    (
        X_train_catboost,
        X_test_catboost,
        categorical_features,
    ) = prepare_catboost_data(
        scenario_data["X_train"],
        scenario_data["X_test"],
        features,
    )

    model = train_catboost_model(
        X_train_catboost,
        scenario_data["y_train"],
        categorical_features,
    )

    probabilities = (
        model
        .predict_proba(
            X_test_catboost
        )[:, 1]
    )

    metrics = calculate_metrics(
        y_true=scenario_data["y_test"],
        probabilities=probabilities,
        threshold=CATBOOST_THRESHOLD,
    )

    return metrics


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

def save_metrics(
    results,
    output_file=METRICS_FILE,
):
    """
    Guarda las métricas de los cuatro escenarios
    en formato JSON dentro de reports/comparison.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
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


def save_final_metrics(
    final_metrics,
    output_file=FINAL_METRICS_FILE,
):
    """
    Guarda las métricas del modelo final
    en formato JSON dentro de reports/final_model.

    El modelo final corresponde a:

        CatBoost - Escenario A
        Todas las 21 variables predictoras.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_metrics,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------

    print(
        "\n=== CARGA Y PREPARACIÓN DEL DATASET ==="
    )

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

    # Validación para asegurar que el ID
    # no se utilice como predictor.

    if "ID" in all_features:
        raise ValueError(
            "ERROR: la variable ID no debe utilizarse "
            "como predictor."
        )

    if len(all_features) != 21:
        raise ValueError(
            "ERROR: se esperaban 21 variables predictoras "
            f"en el Escenario A, pero se encontraron "
            f"{len(all_features)}."
        )

    # --------------------------------------------------------
    # 3. Partición
    # --------------------------------------------------------

    print(
        "\n=== CREACIÓN DE PARTICIONES ==="
    )

    partition = (
        create_partition_scenarios(
            df,
            all_features,
            selected_features,
        )
    )

    # --------------------------------------------------------
    # 4. Resultados
    # --------------------------------------------------------

    results = {}

    # ========================================================
    # LOGISTIC REGRESSION - ESCENARIO A
    # ========================================================

    print(
        "\n=== EVALUACIÓN LOGISTIC REGRESSION ==="
    )

    print(
        "\nEscenario A - 21 variables"
    )

    logistic_a_metrics = (
        evaluate_logistic_scenario(
            partition[
                "A_todas_las_variables"
            ],
            all_features,
        )
    )

    results[
        "Logistic_Regression_A"
    ] = logistic_a_metrics

    # ========================================================
    # LOGISTIC REGRESSION - ESCENARIO B
    # ========================================================

    print(
        "\nEscenario B - 12 variables"
    )

    logistic_b_metrics = (
        evaluate_logistic_scenario(
            partition[
                "B_variables_seleccionadas"
            ],
            selected_features,
        )
    )

    results[
        "Logistic_Regression_B"
    ] = logistic_b_metrics

    # ========================================================
    # CATBOOST - ESCENARIO A
    # ========================================================

    print(
        "\n=== EVALUACIÓN CATBOOST ==="
    )

    print(
        "\nEscenario A - 21 variables"
    )

    catboost_a_metrics = (
        evaluate_catboost_scenario(
            partition[
                "A_todas_las_variables"
            ],
            all_features,
        )
    )

    results[
        "CatBoost_A"
    ] = catboost_a_metrics

    # ========================================================
    # CATBOOST - ESCENARIO B
    # ========================================================

    print(
        "\nEscenario B - 12 variables"
    )

    catboost_b_metrics = (
        evaluate_catboost_scenario(
            partition[
                "B_variables_seleccionadas"
            ],
            selected_features,
        )
    )

    results[
        "CatBoost_B"
    ] = catboost_b_metrics

    # ========================================================
    # GUARDAR MÉTRICAS DE LOS CUATRO ESCENARIOS
    # ========================================================

    save_metrics(
        results
    )

    print(
        "\nResultados de los cuatro escenarios "
        "guardados en:"
    )

    print(
        METRICS_FILE
    )

    # ========================================================
    # GUARDAR MÉTRICAS DEL MODELO FINAL
    # ========================================================

    save_final_metrics(
        catboost_a_metrics
    )

    print(
        "\nMétricas del modelo final guardadas en:"
    )

    print(
        FINAL_METRICS_FILE
    )

    # ========================================================
    # MOSTRAR RESUMEN
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RESUMEN DE MÉTRICAS - 4 ESCENARIOS"
    )

    print(
        "=" * 70
    )

    for scenario_name, metrics in (
        results.items()
    ):

        print(
            f"\n{scenario_name}"
        )

        print(
            f"Threshold: "
            f"{metrics['threshold']:.2f}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Recall clase 0: "
            f"{metrics['recall_class_0']:.4f}"
        )

        print(
            f"Recall clase 1: "
            f"{metrics['recall_class_1']:.4f}"
        )

        print(
            f"Precision clase 1: "
            f"{metrics['precision_class_1']:.4f}"
        )

        print(
            f"F1 clase 1: "
            f"{metrics['f1_class_1']:.4f}"
        )

        print(
            f"F2 clase 1: "
            f"{metrics['f2_class_1']:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC clase 1: "
            f"{metrics['pr_auc_class_1']:.4f}"
        )

    # ========================================================
    # RESUMEN DEL MODELO FINAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "MODELO FINAL"
    )

    print(
        "=" * 70
    )

    print(
        "\nModelo: CatBoost"
    )

    print(
        "Escenario: A - Todas las variables"
    )

    print(
        f"Variables predictoras: "
        f"{len(all_features)}"
    )

    print(
        f"Threshold: "
        f"{catboost_a_metrics['threshold']:.2f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{catboost_a_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Recall clase 1: "
        f"{catboost_a_metrics['recall_class_1']:.4f}"
    )

    print(
        f"Precision clase 1: "
        f"{catboost_a_metrics['precision_class_1']:.4f}"
    )

    print(
        f"F1 clase 1: "
        f"{catboost_a_metrics['f1_class_1']:.4f}"
    )

    print(
        f"F2 clase 1: "
        f"{catboost_a_metrics['f2_class_1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{catboost_a_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC clase 1: "
        f"{catboost_a_metrics['pr_auc_class_1']:.4f}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EVALUACIÓN FINALIZADA CORRECTAMENTE"
    )

    print(
        "=" * 70
    )