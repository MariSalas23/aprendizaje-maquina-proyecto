"""
Punto de entrada principal del proyecto de Machine Learning.

El proyecto sigue la metodología CRISP-DM:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Interpretation

Este archivo permite ejecutar todo el pipeline del proyecto
utilizando únicamente:

    python main.py

Los módulos se ejecutan en el orden necesario y reutilizan
las funciones y configuraciones definidas en cada archivo.
"""

import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# FUNCIÓN PARA EJECUTAR MÓDULOS
# ============================================================

def run_module(module_name, step_number, description):
    """
    Ejecuta un módulo del proyecto como programa independiente.

    Parameters
    ----------
    module_name : str
        Nombre del módulo Python que se ejecutará.

    step_number : int
        Número de la etapa dentro del pipeline.

    description : str
        Descripción de la etapa.
    """

    print("\n" + "=" * 70)
    print(f"{step_number}. {description}")
    print("=" * 70)

    print(f"\nEjecutando: {module_name}\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module_name,
        ],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"\nEl módulo {module_name} terminó con un error."
            "\nEl pipeline se detuvo para evitar generar resultados "
            "incompletos o inconsistentes."
        )

    print(
        f"\n✓ Etapa {step_number} completada correctamente."
    )


# ============================================================
# VERIFICACIÓN DE ESTRUCTURA
# ============================================================

def check_project_structure():
    """
    Verifica que existan las carpetas principales del proyecto.
    """

    required_directories = [
        "config",
        "data",
        "model",
        "notebooks",
        "reports",
        "src",
    ]

    missing_directories = []

    for directory in required_directories:

        directory_path = PROJECT_ROOT / directory

        if not directory_path.exists():
            missing_directories.append(directory)

    if missing_directories:

        raise FileNotFoundError(
            "No se encontraron las siguientes carpetas "
            "principales del proyecto:\n"
            + "\n".join(
                f"- {directory}"
                for directory in missing_directories
            )
        )

    print(
        "✓ Estructura principal del proyecto verificada."
    )


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    """
    Ejecuta el pipeline completo del proyecto.
    """

    print("=" * 70)
    print("PROYECTO DE MACHINE LEARNING")
    print("PREDICCIÓN DE DIABETES")
    print("=" * 70)

    print(
        "\nEjecutando pipeline completo..."
    )

    # --------------------------------------------------------
    # 0. VERIFICACIÓN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("0. VERIFICACIÓN DEL PROYECTO")
    print("=" * 70)

    check_project_structure()

    # --------------------------------------------------------
    # 1. DATA UNDERSTANDING / EDA
    # --------------------------------------------------------

    run_module(
        "src.data.exploratory_analysis",
        1,
        "DATA UNDERSTANDING / EXPLORATORY DATA ANALYSIS",
    )

    # --------------------------------------------------------
    # 2. HYPERPARAMETER TUNING
    # --------------------------------------------------------

    run_module(
        "src.evaluation.hyperparameter_tuning",
        2,
        "HYPERPARAMETER TUNING",
    )

    # --------------------------------------------------------
    # 3. THRESHOLD ANALYSIS
    # --------------------------------------------------------

    run_module(
        "src.evaluation.threshold_analysis",
        3,
        "THRESHOLD ANALYSIS",
    )

    # --------------------------------------------------------
    # 4. MÉTRICAS
    # --------------------------------------------------------

    run_module(
        "src.evaluation.metrics",
        4,
        "EVALUACIÓN DE MODELOS",
    )

    # --------------------------------------------------------
    # 5. COMPARACIÓN DE MODELOS
    # --------------------------------------------------------

    run_module(
        "src.evaluation.model_comparison",
        5,
        "COMPARACIÓN Y SELECCIÓN DEL MODELO",
    )

    # --------------------------------------------------------
    # 6. ENTRENAMIENTO DEL MODELO FINAL
    # --------------------------------------------------------

    run_module(
        "src.modeling.train",
        6,
        "ENTRENAMIENTO DEL MODELO FINAL",
    )

    # --------------------------------------------------------
    # 7. FEATURE IMPORTANCE
    # --------------------------------------------------------

    run_module(
        "src.interpretation.feature_importance",
        7,
        "INTERPRETACIÓN - IMPORTANCIA DE VARIABLES",
    )

    # --------------------------------------------------------
    # 8. PREDICCIONES
    # --------------------------------------------------------

    run_module(
        "src.interpretation.predictions",
        8,
        "INTERPRETACIÓN - PREDICCIONES",
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETADO CORRECTAMENTE")
    print("=" * 70)

    print("\nPrincipales resultados generados:")

    print("\nreports/exploratory_analysis_and_features/")
    print("  └── figures/")

    print("\nreports/comparison/")
    print("  ├── logistic_v3_hyperparameter_tuning.json")
    print("  ├── logistic_v3_threshold_analysis.json")
    print("  ├── metrics_all_models.json")
    print("  ├── model_comparison.json")
    print("  └── figures/")

    print("\nreports/final_model/")
    print("  ├── feature_importance.json")
    print("  ├── predictions.csv")
    print("  └── figures/")

    print("\nmodels/")
    print("  └── modelo_diabetes_catboost_final.joblib")

    print("\n" + "=" * 70)
    print("FIN DEL PROCESO")
    print("=" * 70)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()