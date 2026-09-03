from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURACIÓN
# ============================================================

from config.config import (
    TARGET,
    RANDOM_STATE,
    TEST_SIZE
)


# ============================================================
# 3.3 PARTICIÓN DE LOS DATOS
# ============================================================

def split_data(
    df,
    features,
    target=TARGET,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
):
    """
    Divide los datos en entrenamiento y prueba.

    Se utiliza:
        - 80% entrenamiento
        - 20% prueba
        - random_state = 42
        - estratificación según la variable objetivo
    """

    X = df[
        features
    ].copy()

    y = df[
        target
    ].copy()

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def create_partition_scenarios(
    df,
    all_features,
    selected_features
):
    """
    Crea las particiones para los dos escenarios.

    Ambos escenarios utilizan la misma semilla
    y la misma estratificación.
    """

    (
        X_train_all,
        X_test_all,
        y_train_all,
        y_test_all
    ) = split_data(
        df,
        all_features
    )

    (
        X_train_selected,
        X_test_selected,
        y_train_selected,
        y_test_selected
    ) = split_data(
        df,
        selected_features
    )

    # --------------------------------------------------------
    # Verificar que ambos escenarios utilicen
    # exactamente los mismos registros
    # --------------------------------------------------------

    if not y_train_all.index.equals(
        y_train_selected.index
    ):

        raise ValueError(
            "Los conjuntos de entrenamiento "
            "de ambos escenarios no coinciden."
        )

    if not y_test_all.index.equals(
        y_test_selected.index
    ):

        raise ValueError(
            "Los conjuntos de prueba "
            "de ambos escenarios no coinciden."
        )

    return {

        "A_todas_las_variables": {
            "X_train": X_train_all,
            "X_test": X_test_all,
            "y_train": y_train_all,
            "y_test": y_test_all
        },

        "B_variables_seleccionadas": {
            "X_train": X_train_selected,
            "X_test": X_test_selected,
            "y_train": y_train_selected,
            "y_test": y_test_selected
        }
    }


def display_partition_results(
    partition
):
    """
    Muestra los resultados de la partición.
    """

    print(
        "\n=== 3.3 PARTICIÓN TRAIN / TEST ==="
    )

    for scenario_name, data in (
        partition.items()
    ):

        print(
            f"\n{scenario_name}"
        )

        print(
            f"Train: "
            f"{len(data['X_train']):,}"
        )

        print(
            f"Test: "
            f"{len(data['X_test']):,}"
        )

        print(
            "\nDistribución de clases en Train:"
        )

        print(
            data["y_train"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            .mul(100)
            .round(2)
        )

        print(
            "\nDistribución de clases en Test:"
        )

        print(
            data["y_test"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            .mul(100)
            .round(2)
        )


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

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = prepare_clean_dataset()

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    all_features = (
        get_all_features()
    )

    selected_features = (
        get_selected_features()
    )

    # --------------------------------------------------------
    # Partición
    # --------------------------------------------------------

    partition = create_partition_scenarios(
        df,
        all_features,
        selected_features
    )

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    display_partition_results(
        partition
    )