"""
Análisis de umbral para modelos CatBoost.

El análisis utiliza probabilidades Out-of-Fold (OOF) obtenidas
mediante validación cruzada estratificada de 5 folds.

Se evalúan umbrales entre 0.20 y 0.70, con incrementos de 0.01.

Criterio de selección del umbral:
    1. Recall de clase 0 >= 70%
    2. Maximizar Recall de clase 1
    3. En caso de empate, maximizar F2 de clase 1
    4. En caso de nuevo empate, maximizar Balanced Accuracy

El análisis se realiza para:
    - Escenario A: 21 variables
    - Escenario B: 12 variables seleccionadas
"""


import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold

from config.config import (
    TARGET,
    RANDOM_STATE,
    COMPARISON_FIGURES_DIR,
)

from src.preparation.cleaning import (
    prepare_clean_dataset,
)

from src.preparation.feature_selection import (
    get_all_features,
    get_selected_features,
)

from src.preparation.preprocessing import (
    prepare_catboost_data,
)

from src.modeling.catboost import (
    build_catboost_model,
)


# ============================================================
# CONFIGURACIÓN DEL ANÁLISIS
# ============================================================

N_SPLITS = 5

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.70
THRESHOLD_STEP = 0.01

MIN_RECALL_CLASS_0 = 0.70


# ============================================================
# VALIDACIÓN CRUZADA OOF
# ============================================================

def generate_oof_predictions(
    X,
    y,
    categorical_features,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """
    Genera probabilidades Out-of-Fold (OOF) para CatBoost.

    El modelo se entrena en cada fold utilizando únicamente
    los datos de entrenamiento de ese fold.

    Returns
    -------
    oof_probabilities : numpy.ndarray
        Probabilidades OOF de la clase 1.
    """

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_probabilities = np.zeros(
        len(X),
        dtype=float,
    )

    for fold, (
        train_index,
        validation_index,
    ) in enumerate(
        skf.split(X, y),
        start=1,
    ):

        print(
            f"    Fold {fold}/{n_splits}"
        )

        X_train_fold = X.iloc[
            train_index
        ].copy()

        X_validation_fold = X.iloc[
            validation_index
        ].copy()

        y_train_fold = y.iloc[
            train_index
        ].copy()

        model = build_catboost_model()

        model.fit(
            X_train_fold,
            y_train_fold,
            cat_features=categorical_features,
        )

        fold_probabilities = (
            model
            .predict_proba(
                X_validation_fold
            )[:, 1]
        )

        oof_probabilities[
            validation_index
        ] = fold_probabilities

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

    metrics = {
        "threshold": threshold,

        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision_class_0": (
            precision_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),

        "recall_class_0": (
            recall_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),

        "f1_class_0": (
            f1_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),

        "precision_class_1": (
            precision_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "recall_class_1": (
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "f1_class_1": (
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "f2_class_1": (
            fbeta_score(
                y_true,
                predictions,
                beta=2,
                pos_label=1,
                zero_division=0,
            )
        ),

        "f1_macro": (
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_weighted": (
            f1_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "pr_auc_class_1": (
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
    }

    return metrics


# ============================================================
# EVALUACIÓN DE TODOS LOS UMBRALES
# ============================================================

def evaluate_thresholds(
    y_true,
    probabilities,
):
    """
    Evalúa todos los umbrales definidos en el análisis.
    """

    thresholds = np.round(
        np.arange(
            THRESHOLD_MIN,
            THRESHOLD_MAX + THRESHOLD_STEP / 2,
            THRESHOLD_STEP,
        ),
        2,
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_threshold_metrics(
            y_true,
            probabilities,
            threshold,
        )

        results.append(
            metrics
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# SELECCIÓN DEL UMBRAL
# ============================================================

def select_best_threshold(
    threshold_results,
):
    """
    Selecciona el mejor umbral utilizando el criterio
    definido para el proyecto.

    Prioridad:
        1. Recall clase 0 >= 70%
        2. Mayor Recall clase 1
        3. Mayor F2 clase 1
        4. Mayor Balanced Accuracy
    """

    eligible = threshold_results[
        threshold_results[
            "recall_class_0"
        ] >= MIN_RECALL_CLASS_0
    ].copy()

    if eligible.empty:

        raise ValueError(
            "No se encontró ningún umbral "
            "que cumpla el mínimo de Recall "
            "de clase 0."
        )

    eligible = eligible.sort_values(
        by=[
            "recall_class_1",
            "f2_class_1",
            "balanced_accuracy",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    best_threshold = (
        eligible
        .iloc[0]
        .copy()
    )

    return (
        best_threshold,
        eligible,
    )


# ============================================================
# GRÁFICA DEL ANÁLISIS DE UMBRAL
# ============================================================

def plot_threshold_analysis(
    threshold_results,
    selected_threshold,
    scenario_name,
):
    """
    Genera y guarda la gráfica del análisis de umbral.
    """

    COMPARISON_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        threshold_results["threshold"],
        threshold_results["recall_class_0"],
        label="Recall clase 0",
    )

    plt.plot(
        threshold_results["threshold"],
        threshold_results["recall_class_1"],
        label="Recall clase 1",
    )

    plt.plot(
        threshold_results["threshold"],
        threshold_results["balanced_accuracy"],
        label="Balanced Accuracy",
    )

    plt.axvline(
        selected_threshold,
        linestyle="--",
        label=(
            f"Umbral seleccionado = "
            f"{selected_threshold:.2f}"
        ),
    )

    plt.axhline(
        MIN_RECALL_CLASS_0,
        linestyle=":",
        label="Recall clase 0 mínimo = 70%",
    )

    plt.xlabel(
        "Umbral de clasificación"
    )

    plt.ylabel(
        "Métrica"
    )

    plt.title(
        f"Análisis de umbral - {scenario_name}"
    )

    plt.legend()

    plt.tight_layout()

    filename = (
        f"catboost_threshold_"
        f"{scenario_name}.png"
    )

    output_path = (
        COMPARISON_FIGURES_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# ============================================================
# ANÁLISIS COMPLETO DE UN ESCENARIO
# ============================================================

def analyze_catboost_threshold(
    X,
    y,
    categorical_features,
    scenario_name,
):
    """
    Ejecuta el análisis completo de umbral
    para un escenario CatBoost.
    """

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"ANÁLISIS DE UMBRAL - {scenario_name}"
    )

    print(
        f"{'=' * 60}"
    )

    print(
        "\nGenerando predicciones OOF..."
    )

    oof_probabilities = (
        generate_oof_predictions(
            X=X,
            y=y,
            categorical_features=(
                categorical_features
            ),
        )
    )

    print(
        "\nEvaluando umbrales..."
    )

    threshold_results = (
        evaluate_thresholds(
            y_true=y,
            probabilities=oof_probabilities,
        )
    )

    (
        best_threshold,
        eligible_results,
    ) = select_best_threshold(
        threshold_results
    )

    selected_value = float(
        best_threshold[
            "threshold"
        ]
    )

    print(
        "\nUmbral seleccionado:"
    )

    print(
        f"{selected_value:.2f}"
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
    # Gráfica
    # --------------------------------------------------------

    figure_path = plot_threshold_analysis(
        threshold_results,
        selected_value,
        scenario_name,
    )

    print(
        f"\nGráfica guardada en:"
    )

    print(
        figure_path
    )

    return {
        "scenario": scenario_name,
        "selected_threshold": selected_value,
        "selected_metrics": (
            best_threshold.to_dict()
        ),
        "threshold_results": threshold_results,
        "oof_probabilities": oof_probabilities,
        "figure_path": figure_path,
    }


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Dataset limpio
    # --------------------------------------------------------

    df = prepare_clean_dataset()

    y = df[
        TARGET
    ].copy()

    # --------------------------------------------------------
    # 2. Escenario A
    # --------------------------------------------------------

    all_features = (
        get_all_features()
    )

    (
        X_all,
        _,
        categorical_features_all
    ) = prepare_catboost_data(
        df[all_features],
        df[all_features],
        all_features,
    )

    # --------------------------------------------------------
    # 3. Escenario B
    # --------------------------------------------------------

    selected_features = (
        get_selected_features()
    )

    (
        X_selected,
        _,
        categorical_features_selected
    ) = prepare_catboost_data(
        df[selected_features],
        df[selected_features],
        selected_features,
    )

    # --------------------------------------------------------
    # 4. Análisis Escenario A
    # --------------------------------------------------------

    results_a = analyze_catboost_threshold(
        X=X_all,
        y=y,
        categorical_features=(
            categorical_features_all
        ),
        scenario_name=(
            "escenario_A"
        ),
    )

    # --------------------------------------------------------
    # 5. Análisis Escenario B
    # --------------------------------------------------------

    results_b = analyze_catboost_threshold(
        X=X_selected,
        y=y,
        categorical_features=(
            categorical_features_selected
        ),
        scenario_name=(
            "escenario_B"
        ),
    )

    # --------------------------------------------------------
    # 6. Resumen
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "RESUMEN ANÁLISIS DE UMBRAL CATBOOST"
    )

    print(
        "=" * 60
    )

    print(
        "\nEscenario A:"
    )

    print(
        f"Umbral seleccionado: "
        f"{results_a['selected_threshold']:.2f}"
    )

    print(
        "\nEscenario B:"
    )

    print(
        f"Umbral seleccionado: "
        f"{results_b['selected_threshold']:.2f}"
    )

    print(
        "\nAnálisis de umbral finalizado."
    )