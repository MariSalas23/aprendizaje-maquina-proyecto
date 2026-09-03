from pathlib import Path

import pandas as pd


def load_data(data_path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset de indicadores de salud y diabetes.

    Parameters
    ----------
    data_path : str | Path
        Ruta al archivo CSV.

    Returns
    -------
    pd.DataFrame
        Dataset cargado.
    """

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {data_path}"
        )

    df = pd.read_csv(data_path)

    return df


def validate_target(
    df: pd.DataFrame,
    target: str
) -> None:
    """
    Verifica que la variable objetivo exista en el dataset.
    """

    if target not in df.columns:
        raise ValueError(
            f"La variable objetivo '{target}' "
            "no se encuentra en el dataset."
        )


def get_dataset_info(
    df: pd.DataFrame
) -> dict:
    """
    Obtiene información básica del dataset.
    """

    return {
        "observations": df.shape[0],
        "variables": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum())
    }


if __name__ == "__main__":

    from config.config import DATA_FILE, TARGET

    df = load_data(DATA_FILE)

    validate_target(
        df,
        TARGET
    )

    info = get_dataset_info(df)

    print("Dataset cargado correctamente.")
    print(f"Observaciones: {info['observations']:,}")
    print(f"Variables: {info['variables']}")
    print(f"Valores faltantes: {info['missing_values']:,}")
    print(f"Filas duplicadas: {info['duplicated_rows']:,}")