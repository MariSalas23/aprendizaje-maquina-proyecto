"""
Comparación de modelos del proyecto.

Se comparan los cuatro escenarios evaluados sobre
el conjunto de prueba:

    1. Logistic Regression - Escenario A
    2. Logistic Regression - Escenario B
    3. CatBoost - Escenario A
    4. CatBoost - Escenario B

También se incorpora información del Logistic Regression V3
optimizado mediante GridSearchCV y análisis de umbral.

Los resultados se guardan en:

    reports/comparison/model_comparison.json

Las figuras se guardan en:

    reports/comparison/figures/
"""


import json

import matplotlib.pyplot as plt
import pandas as pd

from config.config import (
    REPORTS_DIR,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

COMPARISON_DIR = (
    REPORTS_DIR
    / "comparison"
)

FIGURES_DIR = (
    COMPARISON_DIR
    / "figures"
)

METRICS_FILE = (
    COMPARISON_DIR
    / "metrics_all_models.json"
)

LOGISTIC_V3_TUNING_FILE = (
    COMPARISON_DIR
    / "logistic_v3_hyperparameter_tuning.json"
)

LOGISTIC_V3_THRESHOLD_FILE = (
    COMPARISON_DIR
    / "logistic_v3_threshold_analysis.json"
)

COMPARISON_FILE = (
    COMPARISON_DIR
    / "model_comparison.json"
)


# ============================================================
# CARGAR JSON
# ============================================================

def load_json(
    file_path,
):
    """
    Carga un archivo JSON.
    """

    if not file_path.exists():

        raise FileNotFoundError(
            f"No se encontró el archivo:\n"
            f"{file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# CONSTRUIR TABLA COMPARATIVA
# ============================================================

def build_comparison_dataframe(
    metrics_results,
):
    """
    Construye una tabla comparativa a partir
    de las métricas de los cuatro escenarios.
    """

    rows = []

    model_names = {
        "Logistic_Regression_A": (
            "Logistic Regression"
        ),

        "Logistic_Regression_B": (
            "Logistic Regression"
        ),

        "CatBoost_A": (
            "CatBoost"
        ),

        "CatBoost_B": (
            "CatBoost"
        ),
    }

    scenario_names = {
        "Logistic_Regression_A": (
            "A - 21 variables"
        ),

        "Logistic_Regression_B": (
            "B - 12 variables"
        ),

        "CatBoost_A": (
            "A - 21 variables"
        ),

        "CatBoost_B": (
            "B - 12 variables"
        ),
    }

    feature_counts = {
        "Logistic_Regression_A": 21,
        "Logistic_Regression_B": 12,
        "CatBoost_A": 21,
        "CatBoost_B": 12,
    }

    for scenario_key, metrics in (
        metrics_results.items()
    ):

        rows.append(
            {
                "scenario_key": scenario_key,

                "model": model_names[
                    scenario_key
                ],

                "scenario": scenario_names[
                    scenario_key
                ],

                "number_of_features": (
                    feature_counts[
                        scenario_key
                    ]
                ),

                "threshold": (
                    metrics[
                        "threshold"
                    ]
                ),

                "accuracy": (
                    metrics[
                        "accuracy"
                    ]
                ),

                "balanced_accuracy": (
                    metrics[
                        "balanced_accuracy"
                    ]
                ),

                "precision_class_1": (
                    metrics[
                        "precision_class_1"
                    ]
                ),

                "recall_class_1": (
                    metrics[
                        "recall_class_1"
                    ]
                ),

                "f1_class_1": (
                    metrics[
                        "f1_class_1"
                    ]
                ),

                "f2_class_1": (
                    metrics[
                        "f2_class_1"
                    ]
                ),

                "roc_auc": (
                    metrics[
                        "roc_auc"
                    ]
                ),

                "pr_auc_class_1": (
                    metrics[
                        "pr_auc_class_1"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SELECCIÓN DEL MEJOR ESCENARIO
# ============================================================

def select_best_scenario(
    comparison_df,
):
    """
    Selecciona el mejor escenario utilizando
    Balanced Accuracy como criterio principal.

    En caso de empate:
        1. Recall clase 1
        2. F2 clase 1
        3. PR-AUC clase 1
    """

    ranking = (
        comparison_df
        .sort_values(
            by=[
                "balanced_accuracy",
                "recall_class_1",
                "f2_class_1",
                "pr_auc_class_1",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return ranking.iloc[0].to_dict()


# ============================================================
# FIGURAS
# ============================================================

def create_comparison_figure(
    comparison_df,
    metric,
    ylabel,
    filename,
):
    """
    Crea una figura comparativa y la guarda
    dentro de reports/comparison/figures/.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = []

    for _, row in comparison_df.iterrows():

        labels.append(
            f"{row['model']}\n"
            f"{row['scenario']}"
        )

    values = (
        comparison_df[
            metric
        ]
        * 100
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        labels,
        values,
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        f"Comparación de modelos - {ylabel}"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
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
# GUARDAR REPORTE
# ============================================================

def save_comparison_report(
    report,
):
    """
    Guarda el reporte final de comparación
    en formato JSON.
    """

    COMPARISON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        COMPARISON_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Cargar métricas de los cuatro escenarios
    # --------------------------------------------------------

    metrics_results = load_json(
        METRICS_FILE
    )

    # --------------------------------------------------------
    # 2. Construir tabla comparativa
    # --------------------------------------------------------

    comparison_df = (
        build_comparison_dataframe(
            metrics_results
        )
    )

    # --------------------------------------------------------
    # 3. Ranking
    # --------------------------------------------------------

    ranking_df = (
        comparison_df
        .sort_values(
            by=[
                "balanced_accuracy",
                "recall_class_1",
                "f2_class_1",
                "pr_auc_class_1",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    ranking_df[
        "ranking"
    ] = range(
        1,
        len(ranking_df) + 1
    )

    # --------------------------------------------------------
    # 4. Mejor escenario
    # --------------------------------------------------------

    best_scenario = (
        select_best_scenario(
            comparison_df
        )
    )

    # --------------------------------------------------------
    # 5. Figuras
    # --------------------------------------------------------

    figure_paths = {}

    figure_paths[
        "balanced_accuracy"
    ] = str(
        create_comparison_figure(
            comparison_df,
            "balanced_accuracy",
            "Balanced Accuracy (%)",
            "model_comparison_balanced_accuracy.png",
        )
    )

    figure_paths[
        "recall_class_1"
    ] = str(
        create_comparison_figure(
            comparison_df,
            "recall_class_1",
            "Recall clase 1 (%)",
            "model_comparison_recall_class_1.png",
        )
    )

    figure_paths[
        "f1_class_1"
    ] = str(
        create_comparison_figure(
            comparison_df,
            "f1_class_1",
            "F1 clase 1 (%)",
            "model_comparison_f1_class_1.png",
        )
    )

    figure_paths[
        "pr_auc_class_1"
    ] = str(
        create_comparison_figure(
            comparison_df,
            "pr_auc_class_1",
            "PR-AUC clase 1 (%)",
            "model_comparison_pr_auc.png",
        )
    )

    # --------------------------------------------------------
    # 6. Información Logistic V3
    # --------------------------------------------------------

    logistic_v3_report = None

    if (
        LOGISTIC_V3_TUNING_FILE.exists()
        and LOGISTIC_V3_THRESHOLD_FILE.exists()
    ):

        tuning_results = load_json(
            LOGISTIC_V3_TUNING_FILE
        )

        threshold_results = load_json(
            LOGISTIC_V3_THRESHOLD_FILE
        )

        logistic_v3_report = {
            "model": (
                tuning_results[
                    "model"
                ]
            ),

            "scenario": (
                tuning_results[
                    "scenario"
                ]
            ),

            "number_of_features": (
                tuning_results[
                    "number_of_features"
                ]
            ),

            "best_params": (
                tuning_results[
                    "best_params"
                ]
            ),

            "best_cv_f1_macro": (
                tuning_results[
                    "best_cv_f1_macro"
                ]
            ),

            "selected_threshold": (
                threshold_results[
                    "selected_threshold"
                ]
            ),

            "threshold_metrics": (
                threshold_results[
                    "selected_metrics"
                ]
            ),
        }

    # --------------------------------------------------------
    # 7. Reporte final
    # --------------------------------------------------------

    report = {
        "comparison_basis": (
            "Test set metrics"
        ),

        "selection_criterion": {
            "primary": (
                "balanced_accuracy"
            ),

            "secondary": (
                "recall_class_1"
            ),

            "tertiary": (
                "f2_class_1"
            ),

            "quaternary": (
                "pr_auc_class_1"
            ),
        },

        "models_evaluated": (
            ranking_df
            .to_dict(
                orient="records"
            )
        ),

        "best_scenario": (
            best_scenario
        ),

        "logistic_v3_optimized": (
            logistic_v3_report
        ),

        "figures": (
            figure_paths
        ),
    }

    # --------------------------------------------------------
    # 8. Guardar
    # --------------------------------------------------------

    save_comparison_report(
        report
    )

    # --------------------------------------------------------
    # 9. Mostrar resultados
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "COMPARACIÓN DE MODELOS"
    )

    print(
        "=" * 75
    )

    print(
        "\nRanking:"
    )

    print(
        ranking_df[
            [
                "ranking",
                "model",
                "scenario",
                "balanced_accuracy",
                "recall_class_1",
                "f1_class_1",
                "f2_class_1",
                "roc_auc",
                "pr_auc_class_1",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\n" + "-" * 75
    )

    print(
        "Mejor escenario:"
    )

    print(
        f"{best_scenario['model']} - "
        f"{best_scenario['scenario']}"
    )

    print(
        f"Balanced Accuracy: "
        f"{best_scenario['balanced_accuracy']:.4f}"
    )

    print(
        f"Recall clase 1: "
        f"{best_scenario['recall_class_1']:.4f}"
    )

    print(
        f"F2 clase 1: "
        f"{best_scenario['f2_class_1']:.4f}"
    )

    print(
        f"PR-AUC clase 1: "
        f"{best_scenario['pr_auc_class_1']:.4f}"
    )

    # --------------------------------------------------------
    # 10. Logistic V3
    # --------------------------------------------------------

    if logistic_v3_report is not None:

        print(
            "\n" + "-" * 75
        )

        print(
            "LOGISTIC REGRESSION V3 OPTIMIZADO"
        )

        print(
            f"Mejores hiperparámetros: "
            f"{logistic_v3_report['best_params']}"
        )

        print(
            f"F1 Macro CV: "
            f"{logistic_v3_report['best_cv_f1_macro']:.4f}"
        )

        print(
            f"Umbral seleccionado: "
            f"{logistic_v3_report['selected_threshold']:.2f}"
        )

    # --------------------------------------------------------
    # 11. Ubicación de resultados
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "Resultados guardados en:"
    )

    print(
        COMPARISON_FILE
    )

    print(
        "\nFiguras guardadas en:"
    )

    print(
        FIGURES_DIR
    )

    print(
        "=" * 75
    )