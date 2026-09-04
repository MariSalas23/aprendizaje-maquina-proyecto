"""
Dashboard interactivo del proyecto de Machine Learning.

Proyecto:
Predicción de diabetes a partir de indicadores de salud
y estilo de vida.

El dashboard utiliza Streamlit para:
1. Comunicar el problema y los objetivos.
2. Visualizar hallazgos del análisis exploratorio.
3. Presentar el desempeño y comparación de modelos.
4. Permitir realizar una predicción individual utilizando
   el modelo final de CatBoost.

IMPORTANTE:
El modelo no se entrena dentro de Streamlit.
La aplicación carga el modelo final previamente generado.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
from catboost import Pool


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "diabetes_012_health_indicators_BRFSS2015.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "modelo_diabetes_catboost_final.joblib"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

EDA_DIR = (
    REPORTS_DIR
    / "exploratory_analysis_and_features"
)

EDA_FIGURES_DIR = EDA_DIR / "figures"

COMPARISON_DIR = REPORTS_DIR / "comparison"

COMPARISON_FIGURES_DIR = (
    COMPARISON_DIR / "figures"
)

FINAL_MODEL_DIR = REPORTS_DIR / "final_model"

FINAL_MODEL_FIGURES_DIR = (
    FINAL_MODEL_DIR / "figures"
)


# ============================================================
# CONFIGURACIÓN DE STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Predicción de Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VARIABLES DEL PROYECTO
# ============================================================

TARGET = "Diabetes_binary"

BINARY_COLS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]

ORDINAL_COLS = [
    "GenHlth",
    "Age",
    "Education",
    "Income",
]

NUMERIC_COLS = [
    "BMI",
    "MentHlth",
    "PhysHlth",
]

ALL_FEATURES = (
    BINARY_COLS
    + ORDINAL_COLS
    + NUMERIC_COLS
)

FINAL_THRESHOLD = 0.51


# ============================================================
# INFORMACIÓN DE LAS VARIABLES
# ============================================================

VARIABLE_LABELS = {
    "HighBP": "Presión arterial alta",
    "HighChol": "Colesterol alto",
    "CholCheck": "Control de colesterol",
    "Smoker": "Fumador",
    "Stroke": "Antecedente de accidente cerebrovascular",
    "HeartDiseaseorAttack": "Enfermedad cardíaca o ataque cardíaco",
    "PhysActivity": "Actividad física",
    "Fruits": "Consumo de frutas",
    "Veggies": "Consumo de vegetales",
    "HvyAlcoholConsump": "Consumo elevado de alcohol",
    "AnyHealthcare": "Tiene cobertura de salud",
    "NoDocbcCost": "No pudo consultar por costo",
    "DiffWalk": "Dificultad para caminar",
    "Sex": "Sexo",
    "GenHlth": "Salud general",
    "Age": "Grupo de edad",
    "Education": "Nivel educativo",
    "Income": "Nivel de ingresos",
    "BMI": "Índice de masa corporal (BMI)",
    "MentHlth": "Días con mala salud mental",
    "PhysHlth": "Días con mala salud física",
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

@st.cache_resource
def load_model():
    """
    Carga el modelo final de CatBoost.
    """
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo final en:\n{MODEL_FILE}"
        )

    return joblib.load(MODEL_FILE)


@st.cache_data
def load_data():
    """
    Carga el dataset original para visualizaciones.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    # Conversión del target utilizada en el proyecto.
    if "Diabetes_012" in df.columns:
        df[TARGET] = (
            df["Diabetes_012"] > 0
        ).astype(int)

    return df


def load_json(file_path):
    """
    Carga un archivo JSON si existe.
    """
    if not file_path.exists():
        return None

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def show_image(image_path, caption=None):
    """
    Muestra una imagen si existe.
    """
    if image_path.exists():
        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True,
        )
        return True

    return False


def format_metric(value, percentage=False):
    """
    Formatea métricas para mostrarlas en el dashboard.
    """
    if value is None:
        return "N/A"

    try:
        value = float(value)

        if percentage:
            return f"{value * 100:.2f}%"

        return f"{value:.4f}"

    except (ValueError, TypeError):
        return str(value)


def find_metric(metrics, possible_names):
    """
    Busca una métrica en un diccionario utilizando
    diferentes nombres posibles.
    """
    if not isinstance(metrics, dict):
        return None

    for name in possible_names:
        if name in metrics:
            return metrics[name]

    return None


def recursive_find(data, possible_names):
    """
    Busca recursivamente una clave dentro de un JSON.
    """
    if isinstance(data, dict):

        for key in possible_names:
            if key in data:
                return data[key]

        for value in data.values():
            result = recursive_find(
                value,
                possible_names,
            )

            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:
            result = recursive_find(
                item,
                possible_names,
            )

            if result is not None:
                return result

    return None


def prepare_prediction_data(input_values):
    """
    Prepara los datos introducidos por el usuario
    para que tengan el mismo formato utilizado por
    CatBoost durante el entrenamiento.
    """

    row = pd.DataFrame(
        [input_values],
        columns=ALL_FEATURES,
    )

    # CatBoost recibe variables binarias y ordinales
    # como variables categóricas.
    for column in BINARY_COLS + ORDINAL_COLS:
        row[column] = row[column].astype(str)

    return row


def predict_diabetes(model, input_values):
    """
    Genera la probabilidad y predicción utilizando
    el modelo final de CatBoost.
    """

    data = prepare_prediction_data(
        input_values
    )

    categorical_features = [
        data.columns.get_loc(column)
        for column in BINARY_COLS + ORDINAL_COLS
    ]

    prediction_pool = Pool(
        data=data,
        cat_features=categorical_features,
    )

    probability = float(
        model.predict_proba(
            prediction_pool
        )[:, 1][0]
    )

    prediction = int(
        probability >= FINAL_THRESHOLD
    )

    return prediction, probability


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #666666;
            margin-bottom: 25px;
        }

        .prediction-box {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #dddddd;
            margin-top: 15px;
            margin-bottom: 15px;
        }

        .small-text {
            font-size: 13px;
            color: #666666;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🩺 Predicción de Diabetes")

st.sidebar.markdown(
    """
    **Proyecto de Machine Learning 1**

    Dashboard interactivo basado en el modelo
    final de CatBoost.
    """
)

page = st.sidebar.radio(
    "Navegación",
    [
        "Inicio",
        "Análisis de datos",
        "Desempeño del modelo",
        "Predicción individual",
    ],
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Metodología: CRISP-DM"
)

st.sidebar.caption(
    "Modelo final: CatBoost"
)


# ============================================================
# CARGA DE RECURSOS
# ============================================================

try:
    model = load_model()
except Exception as error:
    model = None
    st.sidebar.error(
        "No fue posible cargar el modelo."
    )
    st.sidebar.caption(str(error))


# ============================================================
# 1. INICIO
# ============================================================

if page == "🏠 Inicio":

    st.markdown(
        '<div class="main-title">'
        "Predicción de diabetes a partir de "
        "indicadores de salud y estilo de vida"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Dashboard interactivo para análisis, "
        "interpretación y predicción."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.subheader("Problema de análisis")

    st.write(
        """
        El proyecto busca determinar en qué medida los
        indicadores de salud, estilo de vida y características
        sociodemográficas disponibles permiten caracterizar y
        posteriormente predecir la presencia de diabetes o
        prediabetes.
        """
    )

    st.subheader("Objetivo")

    st.write(
        """
        Desarrollar un modelo de clasificación capaz de
        identificar individuos con diabetes o prediabetes
        utilizando indicadores de salud, estilo de vida y
        características sociodemográficas.
        """
    )

    st.markdown("---")

    st.subheader("Resumen del proyecto")

    try:
        df = load_data()

        total_records = len(df)

        prevalence = (
            df[TARGET].mean()
            if TARGET in df.columns
            else None
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Observaciones",
            f"{total_records:,}",
        )

        col2.metric(
            "Variables predictoras",
            "21",
        )

        col3.metric(
            "Modelo final",
            "CatBoost",
        )

        if prevalence is not None:
            col4.metric(
                "Clase 1",
                f"{prevalence * 100:.2f}%",
            )
        else:
            col4.metric(
                "Umbral",
                f"{FINAL_THRESHOLD:.2f}",
            )

    except Exception as error:

        st.warning(
            f"No fue posible cargar el dataset: {error}"
        )

    st.markdown("---")

    st.subheader("Metodología CRISP-DM")

    cols = st.columns(6)

    phases = [
        ("1", "Business\nUnderstanding"),
        ("2", "Data\nUnderstanding"),
        ("3", "Data\nPreparation"),
        ("4", "Modeling"),
        ("5", "Evaluation"),
        ("6", "Interpretation"),
    ]

    for column, (number, name) in zip(
        cols,
        phases,
    ):
        with column:
            st.metric(
                number,
                name,
            )

    st.markdown("---")

    st.info(
        """
        **Interpretación del target:** en este proyecto,
        la clase 0 representa ausencia de diabetes y la
        clase 1 agrupa prediabetes/diabetes según la
        transformación definida para el proyecto.
        """
    )

    st.warning(
        """
        **Nota:** esta herramienta corresponde a un proyecto
        académico de Machine Learning y no constituye una
        herramienta de diagnóstico médico.
        """
    )


# ============================================================
# 2. ANÁLISIS DE DATOS
# ============================================================

elif page == "Análisis de datos":

    st.title("Análisis exploratorio de los datos")

    st.write(
        """
        Esta sección presenta los principales hallazgos del
        análisis exploratorio utilizado para comprender la
        estructura de los datos y seleccionar las variables
        utilizadas posteriormente en el modelado.
        """
    )

    try:
        df = load_data()

        # ----------------------------------------------------
        # Métricas generales
        # ----------------------------------------------------

        st.subheader("Resumen del dataset")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Observaciones",
            f"{len(df):,}",
        )

        col2.metric(
            "Variables",
            f"{len(df.columns):,}",
        )

        col3.metric(
            "Valores faltantes",
            f"{df.isna().sum().sum():,}",
        )

        col4.metric(
            "Duplicados",
            f"{df.duplicated().sum():,}",
        )

        st.markdown("---")

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        st.subheader(
            "Distribución de diabetes / prediabetes"
        )

        if TARGET in df.columns:

            target_counts = (
                df[TARGET]
                .value_counts()
                .sort_index()
            )

            target_df = pd.DataFrame(
                {
                    "Clase": [
                        "0 - No diabetes",
                        "1 - Prediabetes / diabetes",
                    ],
                    "Cantidad": [
                        target_counts.get(0, 0),
                        target_counts.get(1, 0),
                    ],
                }
            )

            col1, col2 = st.columns(
                [1, 1]
            )

            with col1:

                st.bar_chart(
                    target_df.set_index(
                        "Clase"
                    )
                )

            with col2:

                prevalence = (
                    df[TARGET].mean()
                )

                st.metric(
                    "Prevalencia de clase 1",
                    f"{prevalence * 100:.2f}%",
                )

                st.write(
                    f"""
                    De las **{len(df):,} observaciones**,
                    aproximadamente el **{prevalence * 100:.2f}%**
                    pertenece a la clase 1.
                    """
                )

        st.markdown("---")

        # ----------------------------------------------------
        # Variables seleccionadas
        # ----------------------------------------------------

        st.subheader(
            "Variables utilizadas por el modelo final"
        )

        selected_labels = pd.DataFrame(
            {
                "Variable": ALL_FEATURES,
                "Descripción": [
                    VARIABLE_LABELS.get(
                        variable,
                        variable,
                    )
                    for variable in ALL_FEATURES
                ],
            }
        )

        st.dataframe(
            selected_labels,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # ----------------------------------------------------
        # Figuras EDA
        # ----------------------------------------------------

        st.subheader(
            "Hallazgos visuales del análisis exploratorio"
        )

        tab1, tab2, tab3 = st.tabs(
            [
                "Variables numéricas",
                "Variables ordinales",
                "Variables binarias",
            ]
        )

        with tab1:

            show_image(
                EDA_FIGURES_DIR
                / "selected_numeric_distributions.png",
                "Distribución de variables numéricas seleccionadas por clase.",
            )

        with tab2:

            show_image(
                EDA_FIGURES_DIR
                / "selected_ordinal_distributions.png",
                "Distribución de variables ordinales seleccionadas por clase.",
            )

        with tab3:

            show_image(
                EDA_FIGURES_DIR
                / "selected_binary_prevalence.png",
                "Prevalencia de variables binarias seleccionadas según la clase objetivo.",
            )

        st.markdown("---")

        st.subheader(
            "Correlaciones entre variables"
        )

        show_image(
            EDA_FIGURES_DIR
            / "spearman_correlation_matrix.png",
            "Matriz de correlación de Spearman.",
        )

        st.markdown("---")

        st.subheader(
            "Outliers y distribución de variables numéricas"
        )

        show_image(
            EDA_FIGURES_DIR
            / "potential_outliers.png",
            "Identificación exploratoria de posibles valores atípicos mediante IQR.",
        )

    except Exception as error:

        st.error(
            f"No fue posible cargar el análisis: {error}"
        )


# ============================================================
# 3. DESEMPEÑO DEL MODELO
# ============================================================

elif page == "Desempeño del modelo":

    st.title("Desempeño y comparación de modelos")

    st.write(
        """
        En esta sección se presentan las métricas utilizadas
        para evaluar los modelos de clasificación y seleccionar
        el modelo final.
        """
    )

    metrics_file = (
        COMPARISON_DIR
        / "metrics_all_models.json"
    )

    comparison_file = (
        COMPARISON_DIR
        / "model_comparison.json"
    )

    metrics_data = load_json(
        metrics_file
    )

    comparison_data = load_json(
        comparison_file
    )

    # --------------------------------------------------------
    # Modelo final
    # --------------------------------------------------------

    st.subheader(
        "Modelo seleccionado"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Modelo",
        "CatBoost",
    )

    col2.metric(
        "Escenario",
        "A - Todas las variables",
    )

    col3.metric(
        "Threshold",
        f"{FINAL_THRESHOLD:.2f}",
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    st.subheader(
        "Métricas principales"
    )

    if metrics_data is not None:

        # Intentamos localizar CatBoost A
        # dentro de diferentes estructuras posibles.

        catboost_a = None

        if isinstance(
            metrics_data,
            dict,
        ):

            possible_keys = [
                "CatBoost_A",
                "catboost_A",
                "catboost_a",
                "CatBoost A",
                "A_todas_las_variables",
            ]

            for key in possible_keys:

                if key in metrics_data:

                    catboost_a = (
                        metrics_data[key]
                    )
                    break

        if catboost_a is None:

            catboost_a = (
                recursive_find(
                    metrics_data,
                    [
                        "CatBoost_A",
                        "catboost_A",
                        "catboost_a",
                    ],
                )
            )

        if catboost_a is not None:

            accuracy = recursive_find(
                catboost_a,
                [
                    "accuracy",
                    "Accuracy",
                ],
            )

            balanced_accuracy = recursive_find(
                catboost_a,
                [
                    "balanced_accuracy",
                    "Balanced Accuracy",
                    "balanced_accuracy_score",
                ],
            )

            recall_1 = recursive_find(
                catboost_a,
                [
                    "recall_class_1",
                    "Recall_1",
                    "recall_1",
                ],
            )

            precision_1 = recursive_find(
                catboost_a,
                [
                    "precision_class_1",
                    "Precision_1",
                    "precision_1",
                ],
            )

            f1_1 = recursive_find(
                catboost_a,
                [
                    "f1_class_1",
                    "F1_1",
                    "f1_1",
                ],
            )

            roc_auc = recursive_find(
                catboost_a,
                [
                    "roc_auc",
                    "ROC-AUC",
                    "roc_auc_score",
                ],
            )

            pr_auc = recursive_find(
                catboost_a,
                [
                    "pr_auc_class_1",
                    "PR-AUC",
                    "average_precision",
                    "average_precision_score",
                ],
            )

            metric_cols = st.columns(4)

            metric_cols[0].metric(
                "Balanced Accuracy",
                format_metric(
                    balanced_accuracy,
                    percentage=True,
                ),
            )

            metric_cols[1].metric(
                "Recall clase 1",
                format_metric(
                    recall_1,
                    percentage=True,
                ),
            )

            metric_cols[2].metric(
                "F1 clase 1",
                format_metric(
                    f1_1,
                    percentage=True,
                ),
            )

            metric_cols[3].metric(
                "PR-AUC clase 1",
                format_metric(
                    pr_auc,
                    percentage=True,
                ),
            )

            st.markdown("---")

            secondary_cols = st.columns(4)

            secondary_cols[0].metric(
                "Accuracy",
                format_metric(
                    accuracy,
                    percentage=True,
                ),
            )

            secondary_cols[1].metric(
                "Precision clase 1",
                format_metric(
                    precision_1,
                    percentage=True,
                ),
            )

            secondary_cols[2].metric(
                "ROC-AUC",
                format_metric(
                    roc_auc,
                    percentage=True,
                ),
            )

            secondary_cols[3].metric(
                "Threshold",
                f"{FINAL_THRESHOLD:.2f}",
            )

        else:

            st.info(
                """
                El archivo de métricas existe, pero su
                estructura no permite identificar automáticamente
                las métricas de CatBoost A.
                """
            )

    else:

        st.warning(
            "No se encontró metrics_all_models.json."
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Comparación visual
    # --------------------------------------------------------

    st.subheader(
        "Comparación de modelos"
    )

    comparison_images = [
        (
            "Balanced Accuracy",
            "model_comparison_balanced_accuracy.png",
        ),
        (
            "Recall clase 1",
            "model_comparison_recall_class_1.png",
        ),
        (
            "F1 clase 1",
            "model_comparison_f1_class_1.png",
        ),
        (
            "PR-AUC",
            "model_comparison_pr_auc.png",
        ),
    ]

    for title, filename in comparison_images:

        st.markdown(
            f"### {title}"
        )

        show_image(
            COMPARISON_FIGURES_DIR
            / filename,
            title,
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Interpretación
    # --------------------------------------------------------

    st.subheader(
        "Lectura analítica"
    )

    st.write(
        """
        La selección del modelo no debe basarse únicamente en
        Accuracy. Debido a que el problema presenta una clase
        minoritaria, resulta importante analizar métricas como
        Recall, F1, Balanced Accuracy y PR-AUC para determinar
        qué tan bien identifica el modelo los casos pertenecientes
        a la clase 1.
        """
    )

    st.info(
        """
        En este proyecto, el umbral fue ajustado buscando
        mantener un Recall de la clase 0 de al menos 70% y,
        posteriormente, maximizar el Recall de la clase 1,
        utilizando F2 y Balanced Accuracy como criterios de
        desempate.
        """
    )


# ============================================================
# 4. PREDICCIÓN INDIVIDUAL
# ============================================================

elif page == "Predicción individual":

    st.title(
        "Predicción individual"
    )

    st.write(
        """
        Introduce los indicadores de una persona para obtener
        la predicción del modelo final de CatBoost.
        """
    )

    if model is None:

        st.error(
            """
            El modelo final no está disponible.
            Verifica que exista:

            models/modelo_diabetes_catboost_final.joblib
            """
        )

        st.stop()

    st.info(
        """
        **Importante:** esta predicción corresponde al modelo
        desarrollado en el proyecto académico. No constituye
        un diagnóstico médico.
        """
    )

    # --------------------------------------------------------
    # Información general
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "1. Indicadores de salud"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        high_bp = st.selectbox(
            "¿Tiene presión arterial alta?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

        high_chol = st.selectbox(
            "¿Tiene colesterol alto?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

        chol_check = st.selectbox(
            "¿Se ha realizado control de colesterol?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=80.0,
            value=25.0,
            step=0.1,
        )

    with col2:

        stroke = st.selectbox(
            "¿Ha tenido un accidente cerebrovascular?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

        heart_disease = st.selectbox(
            "¿Tiene enfermedad cardíaca o ataque cardíaco?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

        gen_hlth = st.selectbox(
            "Salud general",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "1 - Excelente",
                2: "2 - Muy buena",
                3: "3 - Buena",
                4: "4 - Regular",
                5: "5 - Mala",
            }[x],
        )

        diff_walk = st.selectbox(
            "¿Tiene dificultad para caminar?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

    with col3:

        phys_activity = st.selectbox(
            "¿Realiza actividad física?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

        smoker = st.selectbox(
            "¿Es fumador?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

        fruits = st.selectbox(
            "¿Consume frutas?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

        veggies = st.selectbox(
            "¿Consume vegetales?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Estilo de vida
    # --------------------------------------------------------

    st.subheader(
        "2. Estilo de vida y acceso a salud"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        alcohol = st.selectbox(
            "¿Tiene consumo elevado de alcohol?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

        healthcare = st.selectbox(
            "¿Tiene cobertura de salud?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

    with col2:

        no_doc_cost = st.selectbox(
            "¿No pudo consultar al médico por costo?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Sí",
        )

        ment_hlth = st.number_input(
            "Días con mala salud mental",
            min_value=0,
            max_value=30,
            value=0,
            step=1,
        )

    with col3:

        phys_hlth = st.number_input(
            "Días con mala salud física",
            min_value=0,
            max_value=30,
            value=0,
            step=1,
        )

        sex = st.selectbox(
            "Sexo",
            [0, 1],
            format_func=lambda x:
                "Categoría 0" if x == 0
                else "Categoría 1",
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Características sociodemográficas
    # --------------------------------------------------------

    st.subheader(
        "3. Características sociodemográficas"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.selectbox(
            "Grupo de edad",
            list(range(1, 14)),
        )

    with col2:

        education = st.selectbox(
            "Nivel educativo",
            list(range(1, 7)),
        )

    with col3:

        income = st.selectbox(
            "Nivel de ingresos",
            list(range(1, 9)),
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Predicción
    # --------------------------------------------------------

    input_values = {
        "HighBP": high_bp,
        "HighChol": high_chol,
        "CholCheck": chol_check,
        "Smoker": smoker,
        "Stroke": stroke,
        "HeartDiseaseorAttack": heart_disease,
        "PhysActivity": phys_activity,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": alcohol,
        "AnyHealthcare": healthcare,
        "NoDocbcCost": no_doc_cost,
        "DiffWalk": diff_walk,
        "Sex": sex,
        "GenHlth": gen_hlth,
        "Age": age,
        "Education": education,
        "Income": income,
        "BMI": bmi,
        "MentHlth": ment_hlth,
        "PhysHlth": phys_hlth,
    }

    if st.button(
        "🔍 Generar predicción",
        type="primary",
        use_container_width=True,
    ):

        try:

            prediction, probability = (
                predict_diabetes(
                    model,
                    input_values,
                )
            )

            st.markdown("---")

            st.subheader(
                "Resultado de la predicción"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Probabilidad estimada de clase 1",
                    f"{probability * 100:.2f}%",
                )

            with col2:

                st.metric(
                    "Threshold utilizado",
                    f"{FINAL_THRESHOLD:.2f}",
                )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            if prediction == 1:

                st.error(
                    """
                    **Predicción: clase 1**

                    El modelo clasifica esta observación como
                    perteneciente al grupo de prediabetes/diabetes.
                    """
                )

            else:

                st.success(
                    """
                    **Predicción: clase 0**

                    El modelo clasifica esta observación como
                    perteneciente al grupo sin diabetes.
                    """
                )

            st.markdown("---")

            st.subheader(
                "Interpretación"
            )

            if prediction == 1:

                st.write(
                    f"""
                    El modelo estima una probabilidad de
                    **{probability * 100:.2f}%** de pertenecer a
                    la clase 1.

                    Dado que esta probabilidad es igual o superior
                    al threshold de **{FINAL_THRESHOLD:.2f}**,
                    la predicción corresponde a la clase 1.
                    """
                )

            else:

                st.write(
                    f"""
                    El modelo estima una probabilidad de
                    **{probability * 100:.2f}%** de pertenecer a
                    la clase 1.

                    Como esta probabilidad es inferior al threshold
                    de **{FINAL_THRESHOLD:.2f}**, la predicción
                    corresponde a la clase 0.
                    """
                )

            st.caption(
                """
                La probabilidad corresponde a la salida del
                modelo de clasificación y no debe interpretarse
                como una probabilidad clínica individual.
                """
            )

        except Exception as error:

            st.error(
                "No fue posible generar la predicción."
            )

            st.exception(error)