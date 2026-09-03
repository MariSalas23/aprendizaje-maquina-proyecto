import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from config.config import (
    TARGET,
    BINARY_COLS,
    ORDINAL_COLS,
    NUMERIC_COLS,
    SELECTED_FEATURES,
    EDA_FIGURES_DIR
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
# VISUALIZACIÓN DE VARIABLES SELECCIONADAS
# ============================================================

def plot_selected_numeric_features(df):
    """
    Genera y guarda la distribución de las variables
    numéricas seleccionadas según Diabetes_binary.

    Variables:
        BMI
        PhysHlth
    """

    variables_numericas_seleccionadas = [
        "BMI",
        "PhysHlth"
    ]

    validate_features(
        df,
        variables_numericas_seleccionadas
    )

    EDA_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 5)
    )

    for ax, col in zip(
        axes,
        variables_numericas_seleccionadas
    ):

        sns.kdeplot(
            data=df[
                df[TARGET] == 0
            ],
            x=col,
            fill=True,
            label="Sin diabetes",
            ax=ax
        )

        sns.kdeplot(
            data=df[
                df[TARGET] == 1
            ],
            x=col,
            fill=True,
            label="Prediabetes/Diabetes",
            ax=ax
        )

        ax.set_title(
            f"Distribución de {col} según {TARGET}"
        )

        ax.set_xlabel(
            col
        )

        ax.set_ylabel(
            "Densidad"
        )

        ax.legend()

    plt.tight_layout()

    output_file = (
        EDA_FIGURES_DIR
        / "selected_numeric_distributions.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Gráfica guardada en: {output_file}"
    )


def plot_selected_ordinal_features(df):
    """
    Genera y guarda la distribución de las variables
    ordinales seleccionadas según Diabetes_binary.

    Variables:
        GenHlth
        Age
        Income
        Education
    """

    variables_ordinales_seleccionadas = [
        "GenHlth",
        "Age",
        "Income",
        "Education"
    ]

    validate_features(
        df,
        variables_ordinales_seleccionadas
    )

    EDA_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10)
    )

    axes = axes.flatten()

    for ax, col in zip(
        axes,
        variables_ordinales_seleccionadas
    ):

        sns.countplot(
            data=df,
            x=col,
            hue=TARGET,
            order=sorted(
                df[col].unique()
            ),
            ax=ax
        )

        ax.set_title(
            f"Distribución de {col} según {TARGET}"
        )

        ax.set_xlabel(
            col
        )

        ax.set_ylabel(
            "Número de participantes"
        )

        ax.legend(
            title="Estado",
            labels=[
                "Sin diabetes",
                "Prediabetes/Diabetes"
            ]
        )

    plt.tight_layout()

    output_file = (
        EDA_FIGURES_DIR
        / "selected_ordinal_distributions.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Gráfica guardada en: {output_file}"
    )


def plot_selected_binary_features(df):
    """
    Genera y guarda la prevalencia de las variables
    binarias seleccionadas según Diabetes_binary.

    Variables:
        HighBP
        DiffWalk
        HighChol
        HeartDiseaseorAttack
        PhysActivity
        Stroke
    """

    variables_binarias_seleccionadas = [
        "HighBP",
        "DiffWalk",
        "HighChol",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Stroke"
    ]

    validate_features(
        df,
        variables_binarias_seleccionadas
    )

    EDA_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prevalencia_binarias = []

    for col in variables_binarias_seleccionadas:

        tabla = (
            df.groupby(TARGET)[col]
            .mean()
            .mul(100)
        )

        prevalencia_binarias.append({
            "Variable": col,
            "Sin diabetes (%)": tabla.get(
                0,
                0
            ),
            "Prediabetes/Diabetes (%)": tabla.get(
                1,
                0
            )
        })

    prevalencia_binarias = pd.DataFrame(
        prevalencia_binarias
    )

    print(
        "\n=== PREVALENCIA DE VARIABLES BINARIAS SELECCIONADAS ==="
    )

    print(
        prevalencia_binarias.round(2)
    )

    prevalencia_grafico = (
        prevalencia_binarias
        .set_index("Variable")
    )

    ax = prevalencia_grafico.plot(
        kind="bar",
        figsize=(14, 6)
    )

    ax.set_title(
        f"Prevalencia de variables seleccionadas "
        f"según {TARGET}"
    )

    ax.set_xlabel(
        "Variable"
    )

    ax.set_ylabel(
        "Porcentaje (%)"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    ax.legend(
        title="Estado"
    )

    plt.tight_layout()

    output_file = (
        EDA_FIGURES_DIR
        / "selected_binary_prevalence.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Gráfica guardada en: {output_file}"
    )


def plot_selected_features(df):
    """
    Genera y guarda las visualizaciones de las 12
    variables seleccionadas durante el análisis exploratorio.
    """

    selected_features = (
        get_selected_features()
    )

    validate_features(
        df,
        selected_features
    )

    print(
        "\n=== VISUALIZACIÓN DE VARIABLES SELECCIONADAS ==="
    )

    plot_selected_numeric_features(
        df
    )

    plot_selected_ordinal_features(
        df
    )

    plot_selected_binary_features(
        df
    )


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    display_feature_selection()

    # --------------------------------------------------------
    # Cargar dataset limpio
    # --------------------------------------------------------

    from src.preparation.cleaning import (
        prepare_clean_dataset
    )

    df = prepare_clean_dataset()

    # --------------------------------------------------------
    # Generar y guardar visualizaciones
    # --------------------------------------------------------

    plot_selected_features(
        df
    )