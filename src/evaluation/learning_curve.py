"""
Curva de aprendizaje del modelo final de CatBoost.

Esta evaluación analiza cómo cambia el Recall del modelo
al aumentar progresivamente el número de observaciones
utilizadas para el entrenamiento.

Se utiliza la misma configuración de CatBoost empleada
para el modelo final - Escenario A (21 variables).

El conjunto de prueba final NO se utiliza para construir
la curva de aprendizaje.

La figura se guarda en:

reports/final_model/figures/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

from config.config import (
    DATA_FILE,
    REPORTS_DIR,
    TARGET,
)

from src.preparation.cleaning import (
    clean_dataset,
)

from src.preparation.data_partition import (
    split_data,
)

from src.preparation.preprocessing import (
    prepare_catboost_data,
)

from src.modeling.catboost import (
    train_catboost_model,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

LEARNING_CURVE_DIR = (
    REPORTS_DIR
    / "final_model"
    / "figures"
)

LEARNING_CURVE_FILE = (
    LEARNING_CURVE_DIR
    / "learning_curve_catboost_final.png"
)

# Porcentajes del conjunto de entrenamiento
# utilizados para construir la curva.
TRAINING_SIZES = [
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
]

RANDOM_STATE = 42


# ============================================================
# OBTENER VARIABLES
# ============================================================

def get_all_features(df):
    """
    Obtiene las variables predictoras del modelo final.

    Se excluyen:
    - TARGET
    - ID
    """

    features = [
        column
        for column in df.columns
        if column not in [
            TARGET,
            "ID",
        ]
    ]

    if "ID" in features:
        raise ValueError(
            "ERROR: ID no debe utilizarse como predictor."
        )

    if len(features) != 21:
        raise ValueError(
            f"Se esperaban 21 variables predictoras, "
            f"pero se encontraron {len(features)}."
        )

    return features


# ============================================================
# CURVA DE APRENDIZAJE
# ============================================================

def generate_learning_curve():

    print(
        "\n=== CURVA DE APRENDIZAJE - CATBOOST FINAL ==="
    )

    # --------------------------------------------------------
    # 1. Cargar datos
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_FILE
    )

    # --------------------------------------------------------
    # 2. Limpiar datos
    # --------------------------------------------------------

    df = clean_dataset(
        df
    )

    # --------------------------------------------------------
    # 3. Variables predictoras
    # --------------------------------------------------------

    features = get_all_features(
        df
    )

    print(
        f"\nVariables predictoras: {len(features)}"
    )

    # --------------------------------------------------------
    # 4. Crear partición train/test
    # --------------------------------------------------------
    #
    # El test final NO se utiliza para la curva.
    #

    X_train, X_test, y_train, y_test = split_data(
        df=df,
        features=features,
        target=TARGET,
    )

    print(
        f"Observaciones de entrenamiento: {len(X_train):,}"
    )

    print(
        f"Observaciones de prueba: {len(X_test):,}"
    )

    # --------------------------------------------------------
    # 5. Crear validación interna
    # --------------------------------------------------------
    #
    # La validación se obtiene únicamente del conjunto
    # de entrenamiento.
    #

    (
        X_train_learning,
        X_validation,
        y_train_learning,
        y_validation,
    ) = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # 6. Resultados
    # --------------------------------------------------------

    training_observations = []

    training_recall = []

    validation_recall = []

    # --------------------------------------------------------
    # 7. Entrenamiento progresivo
    # --------------------------------------------------------

    for size in TRAINING_SIZES:

        n_samples = int(
            len(X_train_learning)
            * size
        )

        print(
            f"\nEntrenando con "
            f"{n_samples:,} observaciones..."
        )

        # -----------------------------------------------
        # Seleccionar subconjunto
        # -----------------------------------------------

        X_subset = (
            X_train_learning
            .iloc[:n_samples]
            .copy()
        )

        y_subset = (
            y_train_learning
            .iloc[:n_samples]
            .copy()
        )

        # -----------------------------------------------
        # Preparar datos para CatBoost
        # -----------------------------------------------

        (
            X_subset_catboost,
            X_validation_catboost,
            categorical_features,
        ) = prepare_catboost_data(
            X_subset,
            X_validation,
            features,
        )

        # -----------------------------------------------
        # Entrenar CatBoost
        # -----------------------------------------------

        model = train_catboost_model(
            X_subset_catboost,
            y_subset,
            categorical_features,
        )

        # -----------------------------------------------
        # Predicciones entrenamiento
        # -----------------------------------------------

        train_probabilities = (
            model.predict_proba(
                X_subset_catboost
            )[:, 1]
        )

        train_predictions = (
            train_probabilities >= 0.51
        ).astype(int)

        # -----------------------------------------------
        # Predicciones validación
        # -----------------------------------------------

        validation_probabilities = (
            model.predict_proba(
                X_validation_catboost
            )[:, 1]
        )

        validation_predictions = (
            validation_probabilities >= 0.51
        ).astype(int)

        # -----------------------------------------------
        # Recall
        # -----------------------------------------------

        train_recall = recall_score(
            y_subset,
            train_predictions,
            pos_label=1,
            zero_division=0,
        )

        validation_recall_value = recall_score(
            y_validation,
            validation_predictions,
            pos_label=1,
            zero_division=0,
        )

        # -----------------------------------------------
        # Guardar resultados
        # -----------------------------------------------

        training_observations.append(
            n_samples
        )

        training_recall.append(
            train_recall
        )

        validation_recall.append(
            validation_recall_value
        )

        print(
            f"Recall entrenamiento: "
            f"{train_recall:.4f}"
        )

        print(
            f"Recall validación: "
            f"{validation_recall_value:.4f}"
        )

    # ========================================================
    # 8. Crear DataFrame de resultados
    # ========================================================

    learning_curve_results = pd.DataFrame({
        "training_observations":
            training_observations,

        "training_recall":
            training_recall,

        "validation_recall":
            validation_recall,
    })

    print(
        "\nResultados de la curva de aprendizaje:"
    )

    print(
        learning_curve_results
    )

    # ========================================================
    # 9. Crear gráfica
    # ========================================================

    LEARNING_CURVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        training_observations,
        training_recall,
        marker="o",
        label="Entrenamiento",
    )

    ax.plot(
        training_observations,
        validation_recall,
        marker="o",
        label="Validación",
    )

    # --------------------------------------------------------
    # Configuración
    # --------------------------------------------------------

    ax.set_title(
        "Curva de aprendizaje - CatBoost final"
    )

    ax.set_xlabel(
        "Número de observaciones de entrenamiento"
    )

    ax.set_ylabel(
        "Recall"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    ax.set_ylim(
        0,
        1
    )

    plt.tight_layout()

    # ========================================================
    # 10. Guardar figura
    # ========================================================

    fig.savefig(
        LEARNING_CURVE_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "\nCurva de aprendizaje generada correctamente."
    )

    print(
        f"Figura guardada en: "
        f"{LEARNING_CURVE_FILE}"
    )

    return learning_curve_results


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    generate_learning_curve()