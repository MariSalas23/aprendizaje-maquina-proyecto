import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

from config.config import (
    DATA_FILE,
    TARGET,
    BINARY_COLS,
    ORDINAL_COLS,
    NUMERIC_COLS
)


# ============================================================
# 3.1 LIMPIEZA Y TRANSFORMACIÓN INICIAL
# ============================================================

def load_raw_data(data_path=DATA_FILE):
    """
    Carga el dataset original desde el archivo CSV.
    """

    df = pd.read_csv(data_path)

    return df


def create_binary_target(df):
    """
    Convierte la variable original Diabetes_012
    en la variable objetivo binaria Diabetes_binary.

    Diabetes_binary:
        0 = Sin diabetes
        1 = Prediabetes/Diabetes
    """

    df = df.copy()

    if "Diabetes_012" in df.columns:

        df[TARGET] = (
            df["Diabetes_012"] > 0
        ).astype(int)

        df = df.drop(
            columns=["Diabetes_012"]
        )

    elif TARGET not in df.columns:

        raise ValueError(
            "El dataset no contiene "
            "'Diabetes_012' ni 'Diabetes_binary'."
        )

    return df


def check_required_columns(df):
    """
    Verifica que todas las variables utilizadas
    en el proyecto estén presentes.
    """

    required_columns = (
        BINARY_COLS
        + ORDINAL_COLS
        + NUMERIC_COLS
        + [TARGET]
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Faltan las siguientes variables "
            f"requeridas: {missing_columns}"
        )

    return True


def check_missing_values(df):
    """
    Verifica si existen valores faltantes.
    """

    missing_values = (
        df.isna()
        .sum()
    )

    total_missing = (
        missing_values.sum()
    )

    if total_missing > 0:

        report = (
            missing_values[
                missing_values > 0
            ]
            .sort_values(
                ascending=False
            )
        )

        raise ValueError(
            "Se encontraron valores faltantes:\n"
            f"{report}"
        )

    return True


def validate_binary_variables(df):
    """
    Verifica que las variables binarias
    únicamente contengan 0 y 1.
    """

    invalid_values = {}

    for col in BINARY_COLS:

        values = set(
            df[col]
            .dropna()
            .unique()
        )

        invalid = values - {0, 1}

        if invalid:

            invalid_values[col] = sorted(
                invalid
            )

    if invalid_values:

        raise ValueError(
            "Se encontraron valores inválidos "
            "en variables binarias:\n"
            f"{invalid_values}"
        )

    return True


def validate_target(df):
    """
    Verifica que Diabetes_binary tenga
    únicamente las clases 0 y 1.
    """

    values = set(
        df[TARGET]
        .dropna()
        .unique()
    )

    if not values.issubset({0, 1}):

        raise ValueError(
            "La variable objetivo contiene "
            f"valores inválidos: {sorted(values)}"
        )

    return True


def remove_duplicates(df):
    """
    Elimina registros completamente duplicados.

    Se conserva la primera aparición.
    """

    initial_rows = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    removed_rows = (
        initial_rows
        - len(df)
    )

    return df, removed_rows


def clean_dataset(df):
    """
    Ejecuta la limpieza y validación inicial
    del dataset.
    """

    initial_rows = len(df)

    # --------------------------------------------------------
    # Transformación del target
    # --------------------------------------------------------

    df = create_binary_target(df)

    # --------------------------------------------------------
    # Validaciones
    # --------------------------------------------------------

    check_required_columns(df)

    check_missing_values(df)

    validate_binary_variables(df)

    validate_target(df)

    # --------------------------------------------------------
    # Duplicados
    # --------------------------------------------------------

    df, removed_duplicates = (
        remove_duplicates(df)
    )

    print(
        "\n=== 3.1 LIMPIEZA Y TRANSFORMACIÓN ==="
    )

    print(
        f"Observaciones iniciales: "
        f"{initial_rows:,}"
    )

    print(
        f"Duplicados eliminados: "
        f"{removed_duplicates:,}"
    )

    print(
        f"Observaciones finales: "
        f"{len(df):,}"
    )

    print(
        f"Variables finales: "
        f"{df.shape[1]}"
    )

    print(
        "Valores faltantes: 0"
    )

    print(
        "Variables binarias: válidas"
    )

    print(
        "Variable objetivo: válida"
    )

    return df


def prepare_clean_dataset(data_path=DATA_FILE):
    """
    Carga y prepara el dataset para las
    siguientes etapas del proyecto.
    """

    df = load_raw_data(
        data_path
    )

    df = clean_dataset(
        df
    )

    return df


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    df = prepare_clean_dataset()

    print(
        "\nPrimeras observaciones:"
    )

    print(
        df.head()
    )