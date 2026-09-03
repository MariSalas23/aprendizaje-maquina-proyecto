from config.config import (
    TARGET,
    BINARY_COLS,
    ORDINAL_COLS,
    NUMERIC_COLS,
    SELECTED_FEATURES
)


# ============================================================
# 3.2 SELECCIÓN DE VARIABLES
# ============================================================

def get_all_features():
    """
    Retorna las 21 variables predictoras utilizadas
    en el escenario con todas las variables.
    """

    return (
        BINARY_COLS
        + ORDINAL_COLS
        + NUMERIC_COLS
    )


def get_selected_features():
    """
    Retorna las 12 variables seleccionadas
    a partir del análisis exploratorio.
    """

    return SELECTED_FEATURES.copy()


def validate_features(df, features):
    """
    Verifica que las variables seleccionadas
    estén presentes en el dataset.
    """

    missing_features = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Las siguientes variables no están "
            f"presentes en el dataset: {missing_features}"
        )

    return True


def select_features(df, features):
    """
    Construye el conjunto X utilizando
    las variables especificadas.
    """

    validate_features(
        df,
        features
    )

    X = df[
        features
    ].copy()

    return X


def get_target(df):
    """
    Obtiene la variable objetivo.
    """

    if TARGET not in df.columns:

        raise ValueError(
            f"La variable objetivo '{TARGET}' "
            "no está presente."
        )

    return df[
        TARGET
    ].copy()


def create_feature_scenarios(df):
    """
    Crea los dos escenarios de variables
    utilizados en el proyecto.
    """

    all_features = (
        get_all_features()
    )

    selected_features = (
        get_selected_features()
    )

    validate_features(
        df,
        all_features
    )

    validate_features(
        df,
        selected_features
    )

    scenarios = {

        "A_todas_las_variables": {
            "features": all_features,
            "X": select_features(
                df,
                all_features
            )
        },

        "B_variables_seleccionadas": {
            "features": selected_features,
            "X": select_features(
                df,
                selected_features
            )
        }
    }

    return scenarios


def display_feature_selection():
    """
    Muestra las variables utilizadas
    en cada escenario.
    """

    all_features = (
        get_all_features()
    )

    selected_features = (
        get_selected_features()
    )

    print(
        "\n=== 3.2 SELECCIÓN DE VARIABLES ==="
    )

    print(
        "\nEscenario A - Todas las variables:"
    )

    print(
        f"Cantidad: {len(all_features)}"
    )

    print(
        all_features
    )

    print(
        "\nEscenario B - Variables seleccionadas:"
    )

    print(
        f"Cantidad: {len(selected_features)}"
    )

    print(
        selected_features
    )


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    display_feature_selection()