"""
Optimización de hiperparámetros para Logistic Regression V3.

El modelo utiliza:

    - Variables binarias como 0/1.
    - Variables ordinales mediante One-Hot Encoding.
    - Variables numéricas mediante SplineTransformer
      seguido de StandardScaler.

La búsqueda de hiperparámetros utiliza:

    C:
        0.01
        0.05
        0.10
        0.50
        1.00

    class_weight:
        {0: 1, 1: 2.0}
        {0: 1, 1: 2.5}
        {0: 1, 1: 3.0}
        balanced

Se utiliza validación cruzada estratificada de 5 folds
y F1 Macro como criterio de refit.

Posteriormente se realiza una búsqueda de umbral
utilizando predicciones Out-of-Fold.

Todos los resultados se guardan en:

    reports/comparison/
"""


import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

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
    RANDOM_STATE,
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
    build_logistic_v3_preprocessor,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

COMPARISON_DIR = (
    REPORTS_DIR
    / "comparison"
)

HYPERPARAMETER_RESULTS_FILE = (
    COMPARISON_DIR
    / "logistic_v3_hyperparameter_tuning.json"
)

THRESHOLD_RESULTS_FILE = (
    COMPARISON_DIR
    / "logistic_v3_threshold_analysis.json"
)

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.60
THRESHOLD_STEP = 0.01

MIN_RECALL_CLASS_0 = 0.70


# ============================================================
# CONSTRUCCIÓN DEL PIPELINE
# ============================================================

def build_logistic_v3_pipeline(
    features,
):
    """
    Construye el pipeline de Logistic Regression V3.

    El preprocesamiento se ajusta únicamente con los datos
    utilizados por GridSearchCV en cada fold.
    """

    preprocessor = (
        build_logistic_v3_preprocessor(
            features
        )
    )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# ESPACIO DE HIPERPARÁMETROS
# ============================================================

def get_hyperparameter_grid():
    """
    Define el espacio de búsqueda utilizado
    en Logistic Regression V3.
    """

    parameter_grid = {
        "classifier__C": [
            0.01,
            0.05,
            0.10,
            0.50,
            1.00,
        ],

        "classifier__class_weight": [
            {0: 1, 1: 2.0},
            {0: 1, 1: 2.5},
            {0: 1, 1: 3.0},
            "balanced",
        ],
    }

    return parameter_grid


# ============================================================
# MÉTRICAS PARA GRIDSEARCH
# ============================================================

def get_scoring_metrics():
    """
    Define las métricas utilizadas durante
    la búsqueda de hiperparámetros.
    """

    scoring = {
        "accuracy": "accuracy",

        "balanced_accuracy": (
            "balanced_accuracy"
        ),

        "f1_macro": "f1_macro",

        "f1_weighted": "f1_weighted",

        "roc_auc": "roc_auc",

        "average_precision": (
            "average_precision"
        ),

        "precision_class_0": (
            "precision_macro"
        ),

        "recall_class_0": (
            "recall_macro"
        ),

        "f1_class_0": (
            "f1_macro"
        ),
    }

    return scoring


# ============================================================
# TUNING
# ============================================================

def tune_logistic_v3(
    X_train,
    y_train,
    features,
):
    """
    Ejecuta GridSearchCV para Logistic Regression V3.

    Returns
    -------
    grid_search : GridSearchCV
        Objeto ajustado.

    best_model : Pipeline
        Mejor pipeline encontrado.
    """

    pipeline = (
        build_logistic_v3_pipeline(
            features
        )
    )

    parameter_grid = (
        get_hyperparameter_grid()
    )

    scoring = (
        get_scoring_metrics()
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    return (
        grid_search,
        grid_search.best_estimator_,
    )


# ============================================================
# RESULTADOS DEL TUNING
# ============================================================

def extract_tuning_results(
    grid_search,
):
    """
    Extrae los resultados relevantes del GridSearchCV.
    """

    results = grid_search.cv_results_

    tuning_results = []

    for i in range(
        len(results["params"])
    ):

        tuning_results.append(
            {
                "params": str(
                    results["params"][i]
                ),

                "mean_test_f1_macro": float(
                    results[
                        "mean_test_f1_macro"
                    ][i]
                ),

                "std_test_f1_macro": float(
                    results[
                        "std_test_f1_macro"
                    ][i]
                ),

                "mean_test_accuracy": float(
                    results[
                        "mean_test_accuracy"
                    ][i]
                ),

                "mean_test_balanced_accuracy": float(
                    results[
                        "mean_test_balanced_accuracy"
                    ][i]
                ),

                "mean_test_f1_weighted": float(
                    results[
                        "mean_test_f1_weighted"
                    ][i]
                ),

                "mean_test_roc_auc": float(
                    results[
                        "mean_test_roc_auc"
                    ][i]
                ),

                "mean_test_average_precision": float(
                    results[
                        "mean_test_average_precision"
                    ][i]
                ),
            }
        )

    return tuning_results


# ============================================================
# PREDICCIONES OOF
# ============================================================

def generate_oof_predictions(
    estimator,
    X,
    y,
):
    """
    Genera probabilidades Out-of-Fold utilizando
    el mejor estimador encontrado.

    El modelo se vuelve a entrenar en cada fold.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = np.zeros(
        len(X),
        dtype=float,
    )

    for fold, (
        train_index,
        validation_index,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print(
            f"    Fold {fold}/5"
        )

        X_train_fold = X.iloc[
            train_index
        ]

        X_validation_fold = X.iloc[
            validation_index
        ]

        y_train_fold = y.iloc[
            train_index
        ]

        estimator.fit(
            X_train_fold,
            y_train_fold,
        )

        oof_probabilities[
            validation_index
        ] = (
            estimator
            .predict_proba(
                X_validation_fold
            )[:, 1]
        )

    return oof_probabilities


# ============================================================
# MÉTRICAS POR UMBRAL
# ============================================================

def calculate_threshold_metrics(
    y_true,
    probabilities,
    threshold,
):
    """
    Calcula las métricas para un umbral específico.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
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


# ============================================================
# BÚSQUEDA DE UMBRAL
# ============================================================

def evaluate_thresholds(
    y_true,
    probabilities,
):
    """
    Evalúa umbrales desde 0.20 hasta 0.60
    con pasos de 0.01.
    """

    thresholds = np.round(
        np.arange(
            THRESHOLD_MIN,
            THRESHOLD_MAX
            + THRESHOLD_STEP / 2,
            THRESHOLD_STEP,
        ),
        2,
    )

    results = []

    for threshold in thresholds:

        results.append(
            calculate_threshold_metrics(
                y_true,
                probabilities,
                threshold,
            )
        )

    return results


def select_best_threshold(
    threshold_results,
):
    """
    Selecciona el umbral según el criterio definido:

        1. Recall clase 0 >= 70%
        2. Maximizar Recall clase 1
        3. Maximizar F2 clase 1
        4. Maximizar Balanced Accuracy
    """

    eligible = [
        result
        for result in threshold_results
        if result[
            "recall_class_0"
        ] >= MIN_RECALL_CLASS_0
    ]

    if not eligible:

        raise ValueError(
            "No existe un umbral que cumpla "
            "el mínimo de Recall de clase 0."
        )

    eligible = sorted(
        eligible,
        key=lambda result: (
            result["recall_class_1"],
            result["f2_class_1"],
            result["balanced_accuracy"],
        ),
        reverse=True,
    )

    return eligible[0]


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

def save_json(
    data,
    output_file,
):
    """
    Guarda resultados en JSON.
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
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

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
    # 4. Logistic V3
    #
    # El tuning se realiza sobre el escenario A,
    # utilizando las 21 variables.
    # --------------------------------------------------------

    scenario_a = partition[
        "A_todas_las_variables"
    ]

    print(
        "\n=== 4.3 HYPERPARAMETER TUNING ==="
    )

    print(
        "\nLogistic Regression V3"
    )

    print(
        "Variables utilizadas: "
        f"{len(all_features)}"
    )

    print(
        "\nEjecutando GridSearchCV..."
    )

    (
        grid_search,
        best_model,
    ) = tune_logistic_v3(
        X_train=scenario_a["X_train"],
        y_train=scenario_a["y_train"],
        features=all_features,
    )

    # --------------------------------------------------------
    # 5. Mejor combinación
    # --------------------------------------------------------

    best_params = (
        grid_search.best_params_
    )

    best_score = (
        grid_search.best_score_
    )

    print(
        "\nMejores hiperparámetros:"
    )

    print(
        best_params
    )

    print(
        "\nMejor F1 Macro CV:"
    )

    print(
        f"{best_score:.4f}"
    )

    # --------------------------------------------------------
    # 6. Resultados completos del tuning
    # --------------------------------------------------------

    tuning_results = (
        extract_tuning_results(
            grid_search
        )
    )

    tuning_report = {
        "model": "Logistic Regression V3",

        "scenario": (
            "A_todas_las_variables"
        ),

        "number_of_features": (
            len(all_features)
        ),

        "cv_folds": 5,

        "random_state": (
            RANDOM_STATE
        ),

        "refit_metric": (
            "f1_macro"
        ),

        "parameter_grid": {
            "C": [
                0.01,
                0.05,
                0.10,
                0.50,
                1.00,
            ],

            "class_weight": [
                "{0: 1, 1: 2.0}",
                "{0: 1, 1: 2.5}",
                "{0: 1, 1: 3.0}",
                "balanced",
            ],
        },

        "best_params": str(
            best_params
        ),

        "best_cv_f1_macro": float(
            best_score
        ),

        "all_results": (
            tuning_results
        ),
    }

    save_json(
        tuning_report,
        HYPERPARAMETER_RESULTS_FILE,
    )

    # --------------------------------------------------------
    # 7. Análisis de umbral
    # --------------------------------------------------------

    print(
        "\n=== ANÁLISIS DE UMBRAL LOGISTIC V3 ==="
    )

    print(
        "\nGenerando predicciones OOF..."
    )

    oof_probabilities = (
        generate_oof_predictions(
            estimator=best_model,
            X=scenario_a["X_train"],
            y=scenario_a["y_train"],
        )
    )

    print(
        "\nEvaluando umbrales..."
    )

    threshold_results = (
        evaluate_thresholds(
            y_true=scenario_a["y_train"],
            probabilities=oof_probabilities,
        )
    )

    best_threshold = (
        select_best_threshold(
            threshold_results
        )
    )

    # --------------------------------------------------------
    # 8. Guardar análisis de umbral
    # --------------------------------------------------------

    threshold_report = {
        "model": "Logistic Regression V3",

        "scenario": (
            "A_todas_las_variables"
        ),

        "threshold_range": {
            "minimum": THRESHOLD_MIN,
            "maximum": THRESHOLD_MAX,
            "step": THRESHOLD_STEP,
        },

        "minimum_recall_class_0": (
            MIN_RECALL_CLASS_0
        ),

        "selected_threshold": (
            best_threshold["threshold"]
        ),

        "selected_metrics": (
            best_threshold
        ),

        "all_threshold_results": (
            threshold_results
        ),
    }

    save_json(
        threshold_report,
        THRESHOLD_RESULTS_FILE,
    )

    # --------------------------------------------------------
    # 9. Mostrar resultado final
    # --------------------------------------------------------

    print(
        "\nUmbral seleccionado:"
    )

    print(
        f"{best_threshold['threshold']:.2f}"
    )

    print(
        "\nResultados del umbral seleccionado:"
    )

    print(
        f"Accuracy: "
        f"{best_threshold['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{best_threshold['balanced_accuracy']:.4f}"
    )

    print(
        f"Recall clase 0: "
        f"{best_threshold['recall_class_0']:.4f}"
    )

    print(
        f"Recall clase 1: "
        f"{best_threshold['recall_class_1']:.4f}"
    )

    print(
        f"Precision clase 1: "
        f"{best_threshold['precision_class_1']:.4f}"
    )

    print(
        f"F1 clase 1: "
        f"{best_threshold['f1_class_1']:.4f}"
    )

    print(
        f"F2 clase 1: "
        f"{best_threshold['f2_class_1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{best_threshold['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC clase 1: "
        f"{best_threshold['pr_auc_class_1']:.4f}"
    )

    # --------------------------------------------------------
    # 10. Confirmación
    # --------------------------------------------------------

    print(
        "\nResultados del tuning guardados en:"
    )

    print(
        HYPERPARAMETER_RESULTS_FILE
    )

    print(
        "\nResultados del análisis de umbral guardados en:"
    )

    print(
        THRESHOLD_RESULTS_FILE
    )

    print(
        "\nHyperparameter tuning finalizado."
    )