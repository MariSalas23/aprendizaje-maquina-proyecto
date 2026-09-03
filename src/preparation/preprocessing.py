from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    SplineTransformer
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

from config.config import (
    BINARY_COLS,
    ORDINAL_COLS,
    NUMERIC_COLS
)


# ============================================================
# 3.4 PREPROCESAMIENTO
# ============================================================

def get_preprocessing_columns(
    features
):
    """
    Identifica las variables binarias, ordinales
    y numéricas presentes en el escenario.
    """

    binary_features = [
        col
        for col in BINARY_COLS
        if col in features
    ]

    ordinal_features = [
        col
        for col in ORDINAL_COLS
        if col in features
    ]

    numeric_features = [
        col
        for col in NUMERIC_COLS
        if col in features
    ]

    return (
        binary_features,
        ordinal_features,
        numeric_features
    )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def build_logistic_preprocessor(
    features
):
    """
    Construye el preprocesador utilizado
    para Logistic Regression.

    Variables binarias:
        Se mantienen como 0/1.

    Variables ordinales:
        One-Hot Encoding.

    Variables numéricas:
        StandardScaler.
    """

    (
        binary_features,
        ordinal_features,
        numeric_features
    ) = get_preprocessing_columns(
        features
    )

    transformers = []

    # --------------------------------------------------------
    # Binarias
    # --------------------------------------------------------

    if binary_features:

        transformers.append(
            (
                "binary",
                "passthrough",
                binary_features
            )
        )

    # --------------------------------------------------------
    # Ordinales
    # --------------------------------------------------------

    if ordinal_features:

        transformers.append(
            (
                "ordinal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                ordinal_features
            )
        )

    # --------------------------------------------------------
    # Numéricas
    # --------------------------------------------------------

    if numeric_features:

        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


# ============================================================
# LOGISTIC REGRESSION V3
# ============================================================

def build_logistic_v3_preprocessor(
    features
):
    """
    Construye el preprocesador utilizado
    en Logistic Regression V3.

    Variables binarias:
        Se mantienen como 0/1.

    Variables ordinales:
        One-Hot Encoding.

    Variables numéricas:
        SplineTransformer + StandardScaler.
    """

    (
        binary_features,
        ordinal_features,
        numeric_features
    ) = get_preprocessing_columns(
        features
    )

    transformers = []

    # --------------------------------------------------------
    # Binarias
    # --------------------------------------------------------

    if binary_features:

        transformers.append(
            (
                "binary",
                "passthrough",
                binary_features
            )
        )

    # --------------------------------------------------------
    # Ordinales
    # --------------------------------------------------------

    if ordinal_features:

        transformers.append(
            (
                "ordinal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                ordinal_features
            )
        )

    # --------------------------------------------------------
    # Numéricas
    # --------------------------------------------------------

    if numeric_features:

        numeric_pipeline = [
            (
                "spline",
                SplineTransformer(
                    n_knots=5,
                    degree=3,
                    include_bias=False
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]

        from sklearn.pipeline import Pipeline

        numeric_pipeline = Pipeline(
            steps=numeric_pipeline
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


# ============================================================
# CATBOOST
# ============================================================

def prepare_catboost_data(
    X_train,
    X_test,
    features
):
    """
    Prepara los datos para CatBoost.

    Las variables categóricas se convierten a string,
    mientras que las variables numéricas permanecen
    en formato numérico.
    """

    (
        binary_features,
        ordinal_features,
        numeric_features
    ) = get_preprocessing_columns(
        features
    )

    categorical_features = (
        binary_features
        + ordinal_features
    )

    X_train_catboost = (
        X_train.copy()
    )

    X_test_catboost = (
        X_test.copy()
    )

    # --------------------------------------------------------
    # Conversión de variables categóricas
    # --------------------------------------------------------

    for col in categorical_features:

        X_train_catboost[col] = (
            X_train_catboost[col]
            .astype(str)
        )

        X_test_catboost[col] = (
            X_test_catboost[col]
            .astype(str)
        )

    return (
        X_train_catboost,
        X_test_catboost,
        categorical_features
    )


# ============================================================
# AJUSTE Y TRANSFORMACIÓN
# ============================================================

def fit_preprocessor(
    X_train,
    features,
    model_type="logistic"
):
    """
    Ajusta el preprocesador utilizando únicamente
    los datos de entrenamiento.

    model_type:
        'logistic'
        'logistic_v3'
    """

    if model_type == "logistic":

        preprocessor = (
            build_logistic_preprocessor(
                features
            )
        )

    elif model_type == "logistic_v3":

        preprocessor = (
            build_logistic_v3_preprocessor(
                features
            )
        )

    else:

        raise ValueError(
            "model_type debe ser "
            "'logistic' o 'logistic_v3'."
        )

    preprocessor.fit(
        X_train
    )

    return preprocessor


def transform_data(
    preprocessor,
    X
):
    """
    Transforma los datos usando un preprocesador
    previamente ajustado.
    """

    return preprocessor.transform(
        X
    )


def fit_transform_train_test(
    X_train,
    X_test,
    features,
    model_type="logistic"
):
    """
    Ajusta el preprocesador únicamente con X_train
    y transforma X_train y X_test.
    """

    preprocessor = fit_preprocessor(
        X_train,
        features,
        model_type=model_type
    )

    X_train_transformed = (
        transform_data(
            preprocessor,
            X_train
        )
    )

    X_test_transformed = (
        transform_data(
            preprocessor,
            X_test
        )
    )

    return (
        preprocessor,
        X_train_transformed,
        X_test_transformed
    )


def get_feature_names(
    preprocessor
):
    """
    Obtiene los nombres de las variables
    después de la transformación.
    """

    return (
        preprocessor
        .get_feature_names_out()
    )


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    from src.preparation.cleaning import (
        prepare_clean_dataset
    )

    from src.preparation.feature_selection import (
        get_all_features
    )

    from src.preparation.data_partition import (
        split_data
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = prepare_clean_dataset()

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    features = (
        get_all_features()
    )

    # --------------------------------------------------------
    # Train/Test
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        df,
        features
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    (
        preprocessor,
        X_train_transformed,
        X_test_transformed
    ) = fit_transform_train_test(
        X_train,
        X_test,
        features,
        model_type="logistic"
    )

    print(
        "\n=== 3.4 PREPROCESAMIENTO ==="
    )

    print(
        "Logistic Regression"
    )

    print(
        f"Train transformado: "
        f"{X_train_transformed.shape}"
    )

    print(
        f"Test transformado: "
        f"{X_test_transformed.shape}"
    )

    print(
        "\nVariables resultantes:"
    )

    print(
        len(
            get_feature_names(
                preprocessor
            )
        )
    )

    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------

    (
        X_train_catboost,
        X_test_catboost,
        categorical_features
    ) = prepare_catboost_data(
        X_train,
        X_test,
        features
    )

    print(
        "\nCatBoost"
    )

    print(
        f"Variables categóricas: "
        f"{len(categorical_features)}"
    )

    print(
        categorical_features
    )