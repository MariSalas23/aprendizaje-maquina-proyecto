import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

from config.config import (
    DATA_FILE,
    TARGET,
    BINARY_COLS,
    ORDINAL_COLS,
    NUMERIC_COLS,
    EDA_FIGURES_DIR
)


# ============================================================
# 2.1 CARGA DE LOS DATOS
# ============================================================

def load_data(data_path=DATA_FILE):
    """
    Carga el dataset CDC Diabetes Health Indicators
    desde el archivo CSV almacenado en data/.
    """

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {data_path}"
        )

    df = pd.read_csv(data_path)

    return df


def load_and_display_data(data_path=DATA_FILE):
    """
    Carga los datos y muestra sus dimensiones y
    primeras observaciones.
    """

    df = load_data(data_path)

    print(f"Observaciones: {df.shape[0]:,}")
    print(f"Variables: {df.shape[1]}")

    print("\nPrimeras observaciones:")
    print(df.head())

    return df


# ============================================================
# 2.2 CLASIFICACIÓN DE LAS VARIABLES
# ============================================================

def classify_variables(df):
    """
    Presenta la clasificación utilizada en el análisis:
    variables binarias, ordinales, numéricas y target.
    """

    target = TARGET
    binary_cols = BINARY_COLS
    ordinal_cols = ORDINAL_COLS
    numeric_cols = NUMERIC_COLS

    print("Dimensiones:", df.shape)

    variable_report = pd.DataFrame({
        "Variable": df.columns,
        "Tipo pandas": df.dtypes.values,
        "Valores únicos": df.nunique().values
    })

    print("\nClasificación y características de las variables:")
    print(variable_report.to_string(index=False))

    print("\nInformación del DataFrame:")
    df.info()

    return variable_report


# ============================================================
# 2.3 CALIDAD DE LOS DATOS
# ============================================================

def missing_values_analysis(df):
    """
    Analiza los valores faltantes.
    """

    missing = pd.DataFrame({
        "Nulos": df.isna().sum(),
        "Porcentaje": df.isna().mean() * 100
    })

    return missing.sort_values(
        "Nulos",
        ascending=False
    )


def unique_values_analysis(df):
    """
    Analiza valores únicos, mínimos y máximos.
    """

    unique_report = pd.DataFrame({
        "Variable": df.columns,
        "Valores_unicos": [
            df[col].nunique()
            for col in df.columns
        ],
        "Minimo": [
            df[col].min()
            for col in df.columns
        ],
        "Maximo": [
            df[col].max()
            for col in df.columns
        ]
    })

    return unique_report


def validate_binary_variables(df):
    """
    Valida los valores presentes en las variables binarias
    y en la variable objetivo.
    """

    for col in BINARY_COLS + [TARGET]:

        valores = sorted(
            df[col].unique()
        )

        print(f"{col}: {valores}")


def duplicated_records_analysis(df):
    """
    Calcula cantidad y porcentaje de registros repetidos.
    """

    duplicados = df.duplicated().sum()

    print(
        f"Filas repetidas: {duplicados:,}"
    )

    print(
        f"Porcentaje: "
        f"{duplicados / len(df) * 100:.2f}%"
    )

    return duplicados


# ============================================================
# 2.4 ANÁLISIS UNIVARIADO
# ============================================================

def numerical_descriptive_analysis(df):
    """
    Estadísticos descriptivos de las variables numéricas.
    """

    descriptive = df[NUMERIC_COLS].describe().T

    return descriptive


def numerical_extended_summary(df):
    """
    Agrega mediana y asimetría al resumen numérico.
    """

    numeric_summary = (
        df[NUMERIC_COLS]
        .describe()
        .T
    )

    numeric_summary["mediana"] = (
        df[NUMERIC_COLS]
        .median()
    )

    numeric_summary["asimetria"] = (
        df[NUMERIC_COLS]
        .skew()
    )

    return numeric_summary


def plot_numerical_distributions(
    df,
    output_dir=EDA_FIGURES_DIR
):
    """
    Genera los histogramas con KDE de las variables
    BMI, MentHlth y PhysHlth, tal como en el notebook.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    for ax, col in zip(
        axes,
        NUMERIC_COLS
    ):

        sns.histplot(
            df[col],
            bins=30,
            kde=True,
            ax=ax
        )

        ax.set_title(
            f"Distribución de {col}"
        )

    plt.tight_layout()

    fig.savefig(
        output_dir / "numerical_distributions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def binary_distribution_analysis(df):
    """
    Calcula la distribución porcentual de las variables binarias.
    """

    binary_summary = []

    for col in BINARY_COLS:

        counts = df[col].value_counts()

        binary_summary.append({
            "Variable": col,
            "0 (%)": (
                counts.get(0, 0)
                / len(df)
            ) * 100,
            "1 (%)": (
                counts.get(1, 0)
                / len(df)
            ) * 100
        })

    binary_summary = pd.DataFrame(
        binary_summary
    )

    return binary_summary.round(2)


def plot_ordinal_distributions(
    df,
    output_dir=EDA_FIGURES_DIR
):
    """
    Genera los countplots de las variables ordinales:
    GenHlth, Age, Education e Income.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    for ax, col in zip(
        axes.flatten(),
        ORDINAL_COLS
    ):

        sns.countplot(
            data=df,
            x=col,
            order=sorted(
                df[col].unique()
            ),
            ax=ax
        )

        ax.set_title(
            f"Distribución de {col}"
        )

    plt.tight_layout()

    fig.savefig(
        output_dir / "ordinal_distributions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 2.5 VARIABLE OBJETIVO
# ============================================================

def target_distribution_analysis(df):
    """
    Calcula cantidad y porcentaje de cada clase
    de Diabetes_binary.
    """

    target_count = (
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    target_pct = (
        df[TARGET]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    target_summary = pd.DataFrame({
        "Cantidad": target_count,
        "Porcentaje": target_pct
    })

    target_summary.index = [
        "Sin diabetes",
        "Prediabetes/Diabetes"
    ]

    return target_summary.round(2)


def plot_target_distribution(
    df,
    output_dir=EDA_FIGURES_DIR
):
    """
    Genera el gráfico de distribución de Diabetes_binary.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(7, 5)
    )

    ax = sns.countplot(
        data=df,
        x=TARGET
    )

    plt.title(
        "Distribución de Diabetes_binary"
    )

    plt.xlabel(
        "Estado"
    )

    plt.ylabel(
        "Número de participantes"
    )

    plt.xticks(
        [0, 1],
        [
            "Sin diabetes",
            "Prediabetes/Diabetes"
        ]
    )

    fig = ax.get_figure()

    fig.savefig(
        output_dir / "target_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 2.6.1 BINARIAS VS DIABETES
# ============================================================

def binary_target_analysis(df):
    """
    Calcula la prevalencia de Diabetes_binary según
    cada variable binaria y la diferencia en puntos porcentuales.
    """

    binary_target_analysis = []

    for col in BINARY_COLS:

        rates = (
            df.groupby(col)[TARGET]
            .mean()
            * 100
        )

        binary_target_analysis.append({
            "Variable": col,
            "Diabetes cuando = 0 (%)":
                rates.get(0, np.nan),

            "Diabetes cuando = 1 (%)":
                rates.get(1, np.nan),

            "Diferencia (pp)":
                rates.get(1, np.nan)
                -
                rates.get(0, np.nan)
        })

    binary_target_analysis = pd.DataFrame(
        binary_target_analysis
    )

    return (
        binary_target_analysis
        .sort_values(
            "Diferencia (pp)",
            ascending=False
        )
        .round(2)
    )


# ============================================================
# 2.6.2 ORDINALES VS DIABETES
# ============================================================

def ordinal_target_analysis(df):
    """
    Calcula la prevalencia de diabetes según
    cada variable ordinal.
    """

    ordinal_prevalence = {}

    for col in ORDINAL_COLS:

        prevalence = (
            df.groupby(col)[TARGET]
            .mean()
            .mul(100)
        )

        ordinal_prevalence[col] = prevalence

    return ordinal_prevalence


def plot_ordinal_prevalence(
    df,
    output_dir=EDA_FIGURES_DIR
):
    """
    Genera los gráficos de prevalencia de diabetes
    según GenHlth, Age, Education e Income.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10)
    )

    for ax, col in zip(
        axes.flatten(),
        ORDINAL_COLS
    ):

        prevalence = (
            df.groupby(col)[TARGET]
            .mean()
            .mul(100)
        )

        prevalence.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            f"Prevalencia de diabetes según {col}"
        )

        ax.set_ylabel(
            "% Diabetes/Prediabetes"
        )

        ax.set_xlabel(
            col
        )

    plt.tight_layout()

    fig.savefig(
        output_dir / "ordinal_prevalence.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 2.6.3 NUMÉRICAS VS DIABETES
# ============================================================

def numerical_target_analysis(df):
    """
    Calcula media, mediana y desviación estándar
    de las variables numéricas por clase del target.
    """

    return (
        df.groupby(TARGET)[NUMERIC_COLS]
        .agg(
            ["mean", "median", "std"]
        )
        .T
    )


def plot_numerical_by_target(
    df,
    output_dir=EDA_FIGURES_DIR
):
    """
    Genera los boxplots de las variables numéricas
    según Diabetes_binary.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    for ax, col in zip(
        axes,
        NUMERIC_COLS
    ):

        sns.boxplot(
            data=df,
            x=TARGET,
            y=col,
            ax=ax
        )

        ax.set_title(
            f"{col} según estado de diabetes"
        )

        ax.set_xlabel(
            "Diabetes_binary"
        )

    plt.tight_layout()

    fig.savefig(
        output_dir / "numerical_by_target.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_numerical_density_by_target(
    df,
    output_dir=EDA_FIGURES_DIR
):
    """
    Genera gráficos de densidad para comparar
    la distribución de BMI, MentHlth y PhysHlth
    entre las dos clases de Diabetes_binary.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    for ax, col in zip(
        axes,
        NUMERIC_COLS
    ):

        sns.kdeplot(
            data=df[df[TARGET] == 0],
            x=col,
            fill=True,
            alpha=0.3,
            label="Sin diabetes",
            ax=ax
        )

        sns.kdeplot(
            data=df[df[TARGET] == 1],
            x=col,
            fill=True,
            alpha=0.3,
            label="Prediabetes/Diabetes",
            ax=ax
        )

        ax.set_title(
            f"Distribución de {col} según estado de diabetes"
        )

        ax.set_xlabel(
            col
        )

        ax.set_ylabel(
            "Densidad"
        )

        ax.legend()

    plt.tight_layout()

    fig.savefig(
        output_dir / "numerical_density_by_target.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 2.7 CORRELACIONES
# ============================================================

def spearman_correlation_analysis(df):
    """
    Calcula la matriz de correlación de Spearman
    únicamente para las variables numéricas.
    """

    corr_spearman = df[NUMERIC_COLS].corr(
        method="spearman"
    )

    return corr_spearman


def plot_spearman_correlation(
    df,
    output_dir=EDA_FIGURES_DIR
):
    """
    Genera la matriz de asociación de Spearman
    únicamente para las variables numéricas.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    corr_spearman = df[NUMERIC_COLS].corr(
        method="spearman"
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        corr_spearman,
        cmap="coolwarm",
        center=0,
        square=True,
        annot=True,
        fmt=".2f"
    )

    plt.title(
        "Matriz de correlación de Spearman - Variables numéricas"
    )

    fig = plt.gcf()

    fig.savefig(
        output_dir / "spearman_correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def target_correlation_analysis(df):
    """
    Ordena las correlaciones de las variables con
    Diabetes_binary por valor absoluto.
    """

    corr_spearman = df.corr(
        method="spearman"
    )

    target_corr = (
        corr_spearman[TARGET]
        .drop(TARGET)
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    return target_corr

# ============================================================
# 2.8 OUTLIERS
# ============================================================

def plot_outliers(
    df,
    output_dir=EDA_FIGURES_DIR
):
    """
    Genera los boxplots utilizados para identificar
    posibles valores extremos en las variables numéricas.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 4)
    )

    for ax, col in zip(
        axes,
        NUMERIC_COLS
    ):

        sns.boxplot(
            x=df[col],
            ax=ax
        )

        ax.set_title(
            f"Posibles valores extremos - {col}"
        )

    plt.tight_layout()

    fig.savefig(
        output_dir / "potential_outliers.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def outlier_analysis(df):
    """
    Calcula los posibles outliers utilizando la regla del IQR.
    """

    outlier_report = []

    for col in NUMERIC_COLS:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = (
            (df[col] < lower)
            |
            (df[col] > upper)
        ).sum()

        outlier_report.append({
            "Variable": col,
            "Limite inferior": lower,
            "Limite superior": upper,
            "Posibles outliers": n_outliers,
            "%": (
                n_outliers
                / len(df)
                * 100
            )
        })

    return (
        pd.DataFrame(
            outlier_report
        )
        .round(2)
    )


# ============================================================
# GUARDADO DE RESULTADOS DEL EDA
# ============================================================

def save_eda_results(
    df,
    output_dir=EDA_FIGURES_DIR
):
    """
    Guarda en formato JSON los principales resultados
    estructurados del análisis exploratorio.

    Se almacenan únicamente resultados complementarios
    a las gráficas para evitar duplicar información.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Información general del dataset
    # --------------------------------------------------------

    missing_total = int(
        df.isna().sum().sum()
    )

    duplicated_rows = int(
        df.duplicated().sum()
    )

    # --------------------------------------------------------
    # Distribución de la variable objetivo
    # --------------------------------------------------------

    target_counts = (
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    target_percentages = (
        df[TARGET]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    target_distribution = {
        str(int(index)): {
            "count": int(
                target_counts.loc[index]
            ),
            "percentage": round(
                float(
                    target_percentages.loc[index]
                ),
                2
            )
        }
        for index in target_counts.index
    }

    # --------------------------------------------------------
    # Resultados de outliers
    # --------------------------------------------------------

    outlier_results = {}

    for col in NUMERIC_COLS:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = int(
            (
                (df[col] < lower)
                |
                (df[col] > upper)
            ).sum()
        )

        percentage = (
            n_outliers
            / len(df)
            * 100
        )

        outlier_results[col] = {
            "Q1": round(
                float(Q1),
                2
            ),
            "Q3": round(
                float(Q3),
                2
            ),
            "IQR": round(
                float(IQR),
                2
            ),
            "lower_limit": round(
                float(lower),
                2
            ),
            "upper_limit": round(
                float(upper),
                2
            ),
            "possible_outliers": n_outliers,
            "percentage": round(
                float(percentage),
                2
            )
        }

    # --------------------------------------------------------
    # Correlaciones con la variable objetivo
    # --------------------------------------------------------

    corr_spearman = df.corr(
        method="spearman"
    )

    target_corr = (
        corr_spearman[TARGET]
        .drop(TARGET)
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    target_correlations = {
        str(variable): round(
            float(correlation),
            4
        )
        for variable, correlation
        in target_corr.items()
    }

    # --------------------------------------------------------
    # Construcción del resultado final
    # --------------------------------------------------------

    results = {
        "dataset": {
            "observations": int(
                df.shape[0]
            ),
            "variables": int(
                df.shape[1]
            ),
            "missing_values": missing_total,
            "duplicated_rows": duplicated_rows
        },

        "target": {
            "name": TARGET,
            "distribution": target_distribution
        },

        "outliers": outlier_results,

        "target_spearman_correlations":
            target_correlations
    }

    # --------------------------------------------------------
    # Guardar JSON
    # --------------------------------------------------------

    output_file = (
        output_dir.parent
        / "eda_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nResultados del EDA guardados en: "
        f"{output_file}"
    )


# ============================================================
# 2.9 HALLAZGOS DEL EDA
# ============================================================

def eda_findings():
    """
    La sección 2.9 del notebook contiene únicamente
    el encabezado 'Hallazgos del EDA' y no desarrolla
    contenido adicional.
    """

    pass


# ============================================================
# EJECUCIÓN COMPLETA DEL ANÁLISIS EXPLORATORIO
# ============================================================

def run_exploratory_analysis():
    """
    Ejecuta las secciones 2.1 a 2.9 del análisis exploratorio.
    """

    # --------------------------------------------------------
    # 2.1 Carga de los datos
    # --------------------------------------------------------

    df = load_and_display_data()

    # --------------------------------------------------------
    # Creación de la variable objetivo binaria
    # --------------------------------------------------------

    # El dataset original contiene Diabetes_012 con tres categorías:
    # 0 = sin diabetes, 1 = prediabetes, 2 = diabetes.
    #
    # Para el problema de clasificación binaria se agrupan
    # prediabetes y diabetes como clase positiva (1).

    if "Diabetes_012" not in df.columns:
        raise ValueError(
            "El dataset no contiene la variable original "
            "'Diabetes_012'."
        )

    df[TARGET] = (
        df["Diabetes_012"] > 0
    ).astype(int)

    # Diabetes_012 es la variable original de tres categorías
    # y no se utiliza como predictor en el análisis binario.
    df = df.drop(
        columns=["Diabetes_012"]
    )

    # --------------------------------------------------------
    # 2.2 Clasificación de las variables
    # --------------------------------------------------------

    classify_variables(df)

    # --------------------------------------------------------
    # 2.3 Calidad de los datos
    # --------------------------------------------------------

    print("\nValores faltantes:")
    print(
        missing_values_analysis(df)
    )

    print("\nValores únicos:")
    print(
        unique_values_analysis(df)
    )

    print("\nValidación de variables binarias:")
    validate_binary_variables(df)

    print("\nRegistros repetidos:")
    duplicated_records_analysis(df)

    # --------------------------------------------------------
    # 2.4 Análisis univariado
    # --------------------------------------------------------

    print("\nEstadísticos descriptivos:")
    print(
        numerical_descriptive_analysis(df)
    )

    print("\nResumen numérico:")
    print(
        numerical_extended_summary(df)
    )

    plot_numerical_distributions(df)

    print("\nDistribución de variables binarias:")
    print(
        binary_distribution_analysis(df)
    )

    plot_ordinal_distributions(df)

    # --------------------------------------------------------
    # 2.5 Variable objetivo
    # --------------------------------------------------------

    print("\nDistribución de la variable objetivo:")
    print(
        target_distribution_analysis(df)
    )

    plot_target_distribution(df)

    # --------------------------------------------------------
    # 2.6.1 Binarias vs Diabetes
    # --------------------------------------------------------

    print("\nBinarias vs Diabetes:")
    print(
        binary_target_analysis(df)
    )

    # --------------------------------------------------------
    # 2.6.2 Ordinales vs Diabetes
    # --------------------------------------------------------

    print("\nOrdinales vs Diabetes:")

    ordinal_results = (
        ordinal_target_analysis(df)
    )

    for variable, prevalence in ordinal_results.items():

        print(f"\n{variable}:")
        print(prevalence)

    plot_ordinal_prevalence(df)

    # --------------------------------------------------------
    # 2.6.3 Numéricas vs Diabetes
    # --------------------------------------------------------

    print("\nNuméricas vs Diabetes:")
    print(
        numerical_target_analysis(df)
    )

    plot_numerical_by_target(df)

    # --------------------------------------------------------
    # 2.7 Correlaciones
    # --------------------------------------------------------

    print("\nCorrelaciones de Spearman:")

    corr_spearman = (
        spearman_correlation_analysis(df)
    )

    print(corr_spearman)

    plot_spearman_correlation(df)

    print(
        "\nCorrelación de variables con "
        "Diabetes_binary:"
    )

    print(
        target_correlation_analysis(df)
    )

    # --------------------------------------------------------
    # 2.8 Outliers
    # --------------------------------------------------------

    plot_outliers(df)

    print("\nAnálisis de outliers:")

    print(
        outlier_analysis(df)
    )

    # --------------------------------------------------------
    # Guardado de resultados estructurados del EDA
    # --------------------------------------------------------

    save_eda_results(df)

    # --------------------------------------------------------
    # 2.9 Hallazgos del EDA
    # --------------------------------------------------------

    eda_findings()

    print(
        "\nAnálisis exploratorio completado."
    )

    print(
        f"Figuras guardadas en: "
        f"{EDA_FIGURES_DIR}"
    )

    return df


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    run_exploratory_analysis()