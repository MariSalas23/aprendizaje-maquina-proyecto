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
import plotly.graph_objects as go
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

        /* Botón principal */
        .stButton > button[kind="primary"] {
            background-color: #3c9bef !important;
            color: #FFFFFF !important;
            border: none !important;
            weight: bold !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }

        /* Hover */
        .stButton > button[kind="primary"]:hover {
            background-color: #F06F5D !important;
            color: #FFFFFF !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Predicción de Diabetes")

st.sidebar.markdown(
    """
    Utilizando de Indicadores de Salud y Estilo de Vida
    """
)

page = st.sidebar.radio(
    "**Menú**",
    [
        "Inicio",
        "Caracterización de la población",
        "Desempeño del modelo",
        "Predicción individual",
    ],
)

st.sidebar.markdown("---")

st.sidebar.caption(
    """
    Jorge Esteban Díaz Bernal

    Carmen Celeste Durán Báez

    Mariana Salas Gutiérrez
    """
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

if page == "Inicio":

    st.markdown(
        '<div class="main-title">'
        "Predicción de diabetes a partir de "
        "indicadores de salud y estilo de vida"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Dashboard interactivo para análisis, interpretación y predicción."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ========================================================
    # CONTEXTO: PROBLEMA DE LA DIABETES
    # ========================================================

    st.subheader("Contexto: El problema de la diabetes")

    st.write(
        """
        La diabetes constituye un importante problema de salud pública
        a nivel mundial. Su presencia está relacionada con diferentes
        factores de salud, estilo de vida y características
        sociodemográficas, por lo que identificar patrones asociados
        puede contribuir a detectar personas con mayor probabilidad
        de presentar la enfermedad.
        """
    )

    # Indicadores de contexto
    col1, col2 = st.columns(2)

    with col1:
        st.container(border=True).markdown(
            """
            ### Situación mundial

            **1 de cada 9 adultos**

            vive con diabetes a nivel mundial [1].
            """
        )

    # Calcular información de nuestra muestra
    try:
        df = load_data()

        total_records = len(df)

        if TARGET in df.columns:
            class_1_count = int(df[TARGET].sum())
            class_1_percentage = df[TARGET].mean() * 100
        else:
            class_1_count = None
            class_1_percentage = None

    except Exception:
        total_records = None
        class_1_count = None
        class_1_percentage = None

    with col2:

        if class_1_count is not None:

            st.container(border=True).markdown(
                f"""
                ### Nuestra muestra

                **{class_1_count:,} personas**

                pertenecen a la **clase 1 (prediabetes o diabetes)**.
                """
            )

        else:

            st.container(border=True).markdown(
                """
                ### Nuestra muestra

                Los datos de la muestra no pudieron ser cargados.
                """
            )

    st.caption(
        "Fuente [1]: International Diabetes Federation (IDF), Diabetes Atlas 2025."
    )

    st.markdown("---")

    # ========================================================
    # PROBLEMA DE ANÁLISIS Y OBJETIVO
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Problema de análisis")

        st.write(
            """
            El proyecto busca determinar en qué medida los indicadores
            de salud, estilo de vida y otras características
            disponibles permiten caracterizar y posteriormente predecir
            la presencia de diabetes o prediabetes.
            """
        )

    with col2:
        st.subheader("Objetivo")

        st.write(
            """
            Desarrollar un modelo de clasificación capaz de identificar
            individuos con diabetes o prediabetes utilizando indicadores
            de salud, estilo de vida y características sociodemográficas.
            """
        )
        
    st.markdown("---")

    # ========================================================
    # FUENTE Y CONTEXTO DE LOS DATOS
    # ========================================================

    st.subheader("Fuente y contexto de los datos")

    st.write(
        """
        El dataset utilizado en este proyecto corresponde al
        conjunto **CDC Diabetes Health Indicators**, disponible
        en el **UCI Machine Learning Repository**. El conjunto
        contiene 253.680 observaciones y 21 variables predictoras.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            """
            ### ¿Qué es el BRFSS?

            El **Behavioral Risk Factor Surveillance System (BRFSS)**
            es un sistema de vigilancia de salud pública de Estados
            Unidos que recopila información sobre factores de riesgo,
            condiciones de salud, comportamientos y prácticas de
            prevención mediante encuestas a adultos.
            """
        )

    with col2:
        st.info(
            """
            ### ¿Quién financió el dataset?

            De acuerdo con la documentación del **UCI Machine Learning
            Repository**, el dataset fue financiado por el **Centers for
            Disease Control and Prevention (CDC)**. El CDC es una agencia de salud pública de Estados Unidos
            encargada de proteger la salud y prevenir enfermedades.
            """
        )

    st.markdown("---")

    # ========================================================
    # MODELO UTILIZADO
    # ========================================================

    st.subheader("Modelo utilizado: CatBoost")

    st.write(
        """
        Para realizar la predicción se utilizó **CatBoost**, un algoritmo
        de aprendizaje automático basado en árboles de decisión que
        combina múltiples árboles para realizar predicciones.
        """
    )


    # ========================================================
    # TABLA DE VARIABLES
    # ========================================================

    st.markdown("---")

    st.subheader("Variables utilizadas")

    
    st.write(
            """
            **Interpretación del target:** en este proyecto,
            la **clase 0** representa ausencia de diabetes y la
            **clase 1** agrupa prediabetes y diabetes, de acuerdo
            con la transformación definida para el proyecto.
            """
    )

    st.write(
        """
        La siguiente tabla presenta las variables utilizadas
        en el modelo y el significado de cada una.
        """
    )

    variables_data = pd.DataFrame(
        [
            ["Diabetes_binary", "Objetivo", "Binaria", "0 = sin diabetes; 1 = prediabetes o diabetes"],
            ["HighBP", "Predictora", "Binaria", "0 = sin presión arterial alta; 1 = presión arterial alta"],
            ["HighChol", "Predictora", "Binaria", "0 = sin colesterol alto; 1 = colesterol alto"],
            ["CholCheck", "Predictora", "Binaria", "0 = no se realizó control de colesterol en 5 años; 1 = sí"],
            ["BMI", "Predictora", "Entera", "Índice de masa corporal"],
            ["Smoker", "Predictora", "Binaria", "0 = no ha fumado al menos 100 cigarrillos; 1 = sí"],
            ["Stroke", "Predictora", "Binaria", "0 = no ha tenido un accidente cerebrovascular; 1 = sí"],
            ["HeartDiseaseorAttack", "Predictora", "Binaria", "0 = sin enfermedad coronaria ni infarto; 1 = con antecedente"],
            ["PhysActivity", "Predictora", "Binaria", "0 = sin actividad física en los últimos 30 días; 1 = sí"],
            ["Fruits", "Predictora", "Binaria", "0 = no consume frutas al menos una vez al día; 1 = sí"],
            ["Veggies", "Predictora", "Binaria", "0 = no consume verduras al menos una vez al día; 1 = sí"],
            ["HvyAlcoholConsump", "Predictora", "Binaria", "0 = no consumo elevado de alcohol; 1 = consumo elevado"],
            ["AnyHealthcare", "Predictora", "Binaria", "0 = sin cobertura de salud; 1 = con algún tipo de cobertura"],
            ["NoDocbcCost", "Predictora", "Binaria", "0 = pudo acceder al médico; 1 = no pudo hacerlo por el costo"],
            ["GenHlth", "Predictora", "Ordinal", "Estado general de salud: 1 = excelente hasta 5 = deficiente"],
            ["MentHlth", "Predictora", "Entera", "Número de días de los últimos 30 con salud mental no buena"],
            ["PhysHlth", "Predictora", "Entera", "Número de días de los últimos 30 con salud física no buena"],
            ["DiffWalk", "Predictora", "Binaria", "0 = sin dificultad para caminar o subir escaleras; 1 = con dificultad"],
            ["Sex", "Predictora", "Binaria", "0 = mujer; 1 = hombre"],
            ["Age", "Predictora", "Ordinal", "Grupo de edad codificado de 1 a 13"],
            ["Education", "Predictora", "Ordinal", "Nivel educativo codificado de 1 a 6"],
            ["Income", "Predictora", "Ordinal", "Nivel de ingresos codificado de 1 a 8"],
        ],
        columns=[
            "Variable",
            "Rol",
            "Tipo",
            "Descripción",
        ],
    )

    st.dataframe(
        variables_data,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # NOTA FINAL
    # ========================================================

    st.markdown("---")

    st.error(
        """
        **Nota:** Esta herramienta corresponde a un proyecto
        académico de Machine Learning y no constituye una
        herramienta de diagnóstico médico.
        """
    )

    st.warning(
        """
        **Fuente de los datos:**  
        Los datos utilizados en este proyecto están disponibles en el
        **UCI Machine Learning Repository**.
    
        [Consultar dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
        """
    )


# ============================================================
# 2. ANÁLISIS DE DATOS
# ============================================================

elif page == "Caracterización de la población":

    st.title("Caracterización de la población")

    st.write(
        """
        Esta sección permite caracterizar la población de la base
        de datos e identificar diferencias entre las personas sin
        diabetes y aquellas pertenecientes al grupo de
        prediabetes/diabetes.
        """
    )

    try:
        df = load_data()

        # ========================================================
        # 1. RESUMEN GENERAL DE LA POBLACIÓN
        # ========================================================

        st.subheader("Resumen de la población")

        total_personas = len(df)

        personas_sin_diabetes = int(
            (df[TARGET] == 0).sum()
        )

        personas_diabetes = int(
            (df[TARGET] == 1).sum()
        )

        porcentaje_sin_diabetes = (
            personas_sin_diabetes / total_personas * 100
        )

        porcentaje_diabetes = (
            personas_diabetes / total_personas * 100
        )


        # ========================================================
        # ESTILO DE LAS TARJETAS
        # ========================================================

        st.markdown(
        """
        <style>
        .summary-card {
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 12px;
            padding: 22px;
            height: 175px;
            box-sizing: border-box;
            margin-bottom: 20px;
        }

        .summary-title {
            font-size: 17px;
            font-weight: 600;
            line-height: 1.3;
            min-height: 44px;
        }

        .summary-value {
            font-size: 34px;
            font-weight: 700;
            line-height: 1;
            margin-top: 10px;
        }

        .summary-pill-red {
            display: inline-block;
            background-color: #83c9ff;
            color: #0068c9;
            padding: 6px 13px;
            border-radius: 18px;
            font-size: 14px;
            font-weight: 600;
            margin-top: 10px;
        }

        .summary-pill-green {
            display: inline-block;
            background-color: #0068c9;
            color: #83c9ff;
            padding: 6px 13px;
            border-radius: 18px;
            font-size: 14px;
            font-weight: 600;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
        )


        # ========================================================
        # TARJETAS
        # ========================================================

        col1, col2, col3, col4 = st.columns(4)


        with col1:
            st.markdown(
                f'<div class="summary-card"><div class="summary-title">Total de personas</div><div class="summary-value">{total_personas:,}</div></div>',
                unsafe_allow_html=True
            )


        with col2:
            st.markdown(
                f'<div class="summary-card"><div class="summary-title">Sin diabetes</div><div class="summary-value">{personas_sin_diabetes:,}</div><span class="summary-pill-red">{porcentaje_sin_diabetes:.1f}%</span></div>',
                unsafe_allow_html=True
            )


        with col3:
            st.markdown(
                f'<div class="summary-card"><div class="summary-title">Prediabetes / diabetes</div><div class="summary-value">{personas_diabetes:,}</div><span class="summary-pill-green">{porcentaje_diabetes:.1f}%</span></div>',
                unsafe_allow_html=True
            )


        with col4:
            st.markdown(
                f'<div class="summary-card"><div class="summary-title">Variables predictoras</div><div class="summary-value">{len(ALL_FEATURES)}</div></div>',
                unsafe_allow_html=True
            )


        st.markdown("---")

        # ========================================================
        # 2. DISTRIBUCIÓN DEL TARGET
        # ========================================================

        st.subheader(
            "Distribución de la población según condición"
        )

        target_df = pd.DataFrame(
            {
                "Condición": [
                    "Sin diabetes",
                    "Prediabetes / diabetes",
                ],
                "Personas": [
                    personas_sin_diabetes,
                    personas_diabetes,
                ],
            }
        )

        col1, col2 = st.columns([1.5, 1])

        with col1:

            import altair as alt

            donut = (
                alt.Chart(target_df)
                .mark_arc(
                    innerRadius=65,
                    outerRadius=125
                )
                .encode(
                    theta=alt.Theta(
                        "Personas:Q",
                        stack=True
                    ),
                    color=alt.Color(
                        "Condición:N",
                        legend=alt.Legend(
                            title=None,
                            orient="bottom"
                        )
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Condición:N",
                            title="Condición"
                        ),
                        alt.Tooltip(
                            "Personas:Q",
                            title="Personas",
                            format=","
                        ),
                    ],
                )
                .properties(
                    height=380
                )
            )

            st.altair_chart(
                donut,
                use_container_width=True,
                theme="streamlit"
            )

        with col2:

            # Tarjeta: Sin diabetes
            with st.container(border=True):
                st.markdown("**Sin diabetes**")
                st.markdown(
                    f"### {porcentaje_sin_diabetes:.1f}%"
                )

            # Tarjeta: Prediabetes / diabetes
            with st.container(border=True):
                st.markdown("**Prediabetes / diabetes**")
                st.markdown(
                    f"### {porcentaje_diabetes:.1f}%"
                )

            st.write(
        """
        La mayor parte de la población analizada corresponde
        al grupo sin diabetes, mientras que el grupo de
        prediabetes/diabetes representa una proporción menor
        del total.
        """
    )


        st.markdown("---")

        # ========================================================
        # 3. PERFIL DEMOGRÁFICO
        # ========================================================

        st.subheader("Perfil demográfico")

        st.write(
            """
            Comparación de las características demográficas entre
            las personas sin diabetes y las personas pertenecientes
            al grupo de prediabetes/diabetes.
            """
        )

        # --------------------------------------------------------
        # EDAD
        # --------------------------------------------------------

        st.markdown("### Distribución por grupo de edad")

        age_labels = {
            1: "18–24",
            2: "25–29",
            3: "30–34",
            4: "35–39",
            5: "40–44",
            6: "45–49",
            7: "50–54",
            8: "55–59",
            9: "60–64",
            10: "65–69",
            11: "70–74",
            12: "75–79",
            13: "80+",
        }

        age_comparison = (
            df.groupby(
                ["Age", TARGET]
            )
            .size()
            .unstack(fill_value=0)
        )

        age_comparison = age_comparison.rename(
            columns={
                0: "Sin diabetes",
                1: "Prediabetes / diabetes",
            }
        )

        age_comparison.index = [
            age_labels.get(
                int(age),
                str(age),
            )
            for age in age_comparison.index
        ]

        st.bar_chart(
            age_comparison,
            use_container_width=True,
        )

        st.caption(
            "Cantidad de personas en cada grupo de edad, diferenciando por condición."
        )

        # --------------------------------------------------------
        # SEXO
        # --------------------------------------------------------

        st.markdown("### Distribución por sexo")

        sex_comparison = (
            df.groupby(
                ["Sex", TARGET]
            )
            .size()
            .unstack(fill_value=0)
        )

        sex_comparison = sex_comparison.rename(
            columns={
                0: "Sin diabetes",
                1: "Prediabetes / diabetes",
            }
        )

        sex_comparison.index = [
            "Mujeres" if int(sex) == 0
            else "Hombres"
            for sex in sex_comparison.index
        ]

        st.bar_chart(
            sex_comparison,
            use_container_width=True,
        )

        st.caption(
                    "Cantidad de personas en cada grupo de sexo, diferenciando por condición."
                )

        st.markdown("---")

        # ========================================================
        # 4. EDUCACIÓN E INGRESOS
        # ========================================================

        st.subheader(
            "Características socioeconómicas"
        )

        st.write(
                """
                Esta sección permite analizar las características socioeconómicas de la población, comparando los niveles educativos y de ingresos entre las personas sin diabetes y aquellas pertenecientes al grupo de prediabetes/diabetes.
                """
            )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Nivel educativo")

            education_comparison = (
                df.groupby(
                    ["Education", TARGET]
                )
                .size()
                .unstack(fill_value=0)
            )

            education_comparison = (
                education_comparison.rename(
                    columns={
                        0: "Sin diabetes",
                        1: "Prediabetes / diabetes",
                    }
                )
            )

            education_comparison.index = [
                f"Nivel {int(value)}"
                for value in education_comparison.index
            ]

            st.bar_chart(
                education_comparison,
                use_container_width=True,
            )

            st.caption(
                         "Distribución de las personas según su nivel de ingresos, diferenciando entre los grupos sin diabetes y prediabetes/diabetes."
                            )

        with col2:

            st.markdown("### Nivel de ingresos")

            income_comparison = (
                df.groupby(
                    ["Income", TARGET]
                )
                .size()
                .unstack(fill_value=0)
            )

            income_comparison = (
                income_comparison.rename(
                    columns={
                        0: "Sin diabetes",
                        1: "Prediabetes / diabetes",
                    }
                )
            )

            income_comparison.index = [
                f"Nivel {int(value)}"
                for value in income_comparison.index
            ]

            st.bar_chart(
                income_comparison,
                use_container_width=True,
            )

            st.caption(
                "Distribución de las personas según su nivel de ingresos, diferenciando entre los grupos sin diabetes y prediabetes/diabetes."
            )

        st.markdown("---")

        # ========================================================
        # 5. CONDICIONES DE SALUD
        # ========================================================

        st.subheader("Condiciones de salud")

        health_variables = [
            (
                "HighBP",
                "Presión arterial alta",
            ),
            (
                "HighChol",
                "Colesterol alto",
            ),
            (
                "HeartDiseaseorAttack",
                "Enfermedad cardíaca o ataque cardíaco",
            ),
            (
                "Stroke",
                "Antecedente de accidente cerebrovascular",
            ),
            (
                "DiffWalk",
                "Dificultad para caminar",
            ),
        ]

        health_data = []

        for variable, label in health_variables:

            for condition in [0, 1]:

                subset = df[
                    df[TARGET] == condition
                ]

                percentage = (
                    subset[variable].mean() * 100
                )

                health_data.append(
                    {
                        "Condición": label,
                        "Grupo": (
                            "Sin diabetes"
                            if condition == 0
                            else "Prediabetes / diabetes"
                        ),
                        "Porcentaje": percentage,
                    }
                )

        health_long = pd.DataFrame(
            health_data
        )

        health_comparison = (
            health_long
            .pivot(
                index="Condición",
                columns="Grupo",
                values="Porcentaje",
            )
            .fillna(0)
        )

        st.bar_chart(
            health_comparison,
            use_container_width=True,
        )

        st.caption(
            "Porcentaje de personas dentro de cada grupo que reportaron la condición."
        )

        st.markdown("---")

        # ========================================================
        # 6. HÁBITOS Y ESTILO DE VIDA
        # ========================================================

        st.subheader(
            "Hábitos y estilo de vida"
        )

        lifestyle_variables = [
            (
                "Smoker",
                "Fumadores",
            ),
            (
                "PhysActivity",
                "Realizan actividad física",
            ),
            (
                "Fruits",
                "Consumen frutas",
            ),
            (
                "Veggies",
                "Consumen vegetales",
            ),
            (
                "HvyAlcoholConsump",
                "Consumo elevado de alcohol",
            ),
        ]

        lifestyle_data = []

        for variable, label in lifestyle_variables:

            for condition in [0, 1]:

                subset = df[
                    df[TARGET] == condition
                ]

                percentage = (
                    subset[variable].mean() * 100
                )

                lifestyle_data.append(
                    {
                        "Característica": label,
                        "Grupo": (
                            "Sin diabetes"
                            if condition == 0
                            else "Prediabetes / diabetes"
                        ),
                        "Porcentaje": percentage,
                    }
                )

        lifestyle_long = pd.DataFrame(
            lifestyle_data
        )

        lifestyle_comparison = (
            lifestyle_long
            .pivot(
                index="Característica",
                columns="Grupo",
                values="Porcentaje",
            )
            .fillna(0)
        )

        st.bar_chart(
            lifestyle_comparison,
            use_container_width=True,
        )

        st.caption(
            "Porcentaje de personas dentro de cada grupo que presenta cada característica."
        )

        st.markdown("---")

        # ========================================================
        # 7. ACCESO A SERVICIOS DE SALUD
        # ========================================================

        st.subheader(
            "Acceso y prevención en salud"
        )

        healthcare_variables = [
            (
                "AnyHealthcare",
                "Tiene cobertura de salud",
            ),
            (
                "CholCheck",
                "Se realizó control de colesterol",
            ),
            (
                "NoDocbcCost",
                "No pudo consultar por costo",
            ),
        ]

        healthcare_data = []

        for variable, label in healthcare_variables:

            for condition in [0, 1]:

                subset = df[
                    df[TARGET] == condition
                ]

                percentage = (
                    subset[variable].mean() * 100
                )

                healthcare_data.append(
                    {
                        "Característica": label,
                        "Grupo": (
                            "Sin diabetes"
                            if condition == 0
                            else "Prediabetes / diabetes"
                        ),
                        "Porcentaje": percentage,
                    }
                )

        healthcare_long = pd.DataFrame(
            healthcare_data
        )

        healthcare_comparison = (
            healthcare_long
            .pivot(
                index="Característica",
                columns="Grupo",
                values="Porcentaje",
            )
            .fillna(0)
        )

        st.bar_chart(
            healthcare_comparison,
            use_container_width=True,
        )

        st.caption(
                    "Porcentaje de personas según su acceso a servicios de salud y prácticas de prevención, comparando ambos grupos."
                )

        st.markdown("---")

        # ========================================================
        # 8. SALUD GENERAL
        # ========================================================

        st.subheader(
            "Percepción de salud general"
        )

        health_labels = {
            1: "Excelente",
            2: "Muy buena",
            3: "Buena",
            4: "Regular",
            5: "Mala",
        }

        general_health_comparison = (
            df.groupby(
                ["GenHlth", TARGET]
            )
            .size()
            .unstack(fill_value=0)
        )

        general_health_comparison = (
            general_health_comparison.rename(
                columns={
                    0: "Sin diabetes",
                    1: "Prediabetes / diabetes",
                }
            )
        )

        general_health_comparison.index = [
            health_labels.get(
                int(value),
                str(value),
            )
            for value in general_health_comparison.index
        ]

        st.bar_chart(
            general_health_comparison,
            use_container_width=True,
        )

        st.caption(
            "Distribución de la percepción de salud general, diferenciando entre los grupos sin diabetes y prediabetes/diabetes."
        )

    except Exception as error:

        st.error(
            f"No fue posible cargar el análisis: {error}"
        )


# ============================================================
# 3. DESEMPEÑO DEL MODELO
# ============================================================

elif page == "Desempeño del modelo":

    st.title("Desempeño del modelo")

    st.write(
        """
        En esta sección se presentan las principales métricas de
        desempeño del modelo final de CatBoost, seleccionado para
        realizar la predicción de prediabetes/diabetes.
        """
    )

    metrics_file = (
        COMPARISON_DIR
        / "metrics_all_models.json"
    )

    metrics_data = load_json(
        metrics_file
    )

    # --------------------------------------------------------
    # Modelo seleccionado
    # --------------------------------------------------------

    st.subheader(
        "Desempeño de CatBoost"
    )

    st.markdown(
        """
        <style>
        .model-card {
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 12px;
            padding: 22px;
            height: 140px;
            box-sizing: border-box;
            margin-bottom: 20px;
        }

        .model-title {
            font-size: 17px;
            font-weight: 600;
            line-height: 1.3;
            min-height: 44px;
        }

        .model-value {
            font-size: 34px;
            font-weight: 700;
            line-height: 1;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    model_cols = st.columns(3)

    with model_cols[0]:
        st.markdown(
            '<div class="model-card">'
            '<div class="model-title">Modelo</div>'
            '<div class="model-value">CatBoost</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with model_cols[1]:
        st.markdown(
            f'<div class="model-card">'
            f'<div class="model-title">Threshold</div>'
            f'<div class="model-value">{FINAL_THRESHOLD:.2f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with model_cols[2]:
        st.markdown(
            f'<div class="model-card">'
            f'<div class="model-title">Variables utilizadas</div>'
            f'<div class="model-value">{len(ALL_FEATURES)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Métricas del modelo
    # --------------------------------------------------------

    st.subheader(
        "Métricas de desempeño"
    )

    if metrics_data is not None:

        # ----------------------------------------------------
        # Localizar CatBoost A
        # ----------------------------------------------------

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

            catboost_a = recursive_find(
                metrics_data,
                [
                    "CatBoost_A",
                    "catboost_A",
                    "catboost_a",
                ],
            )

        # ----------------------------------------------------
        # Extraer métricas
        # ----------------------------------------------------

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

            # ---------------- Clase 0 ----------------

            precision_0 = recursive_find(
                catboost_a,
                [
                    "precision_class_0",
                    "Precision_0",
                    "precision_0",
                ],
            )

            recall_0 = recursive_find(
                catboost_a,
                [
                    "recall_class_0",
                    "Recall_0",
                    "recall_0",
                ],
            )

            f1_0 = recursive_find(
                catboost_a,
                [
                    "f1_class_0",
                    "F1_0",
                    "f1_0",
                ],
            )

            # ---------------- Clase 1 ----------------

            precision_1 = recursive_find(
                catboost_a,
                [
                    "precision_class_1",
                    "Precision_1",
                    "precision_1",
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

            f1_1 = recursive_find(
                catboost_a,
                [
                    "f1_class_1",
                    "F1_1",
                    "f1_1",
                ],
            )

            # ---------------- Otras métricas ----------------

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

            # ------------------------------------------------
            # Estilo de las cajitas
            # ------------------------------------------------

            st.markdown(
                """
                <style>
                .performance-card {
                    border: 1px solid rgba(128, 128, 128, 0.35);
                    border-radius: 12px;
                    padding: 22px;
                    height: 155px;
                    box-sizing: border-box;
                    margin-bottom: 20px;
                }

                .performance-title {
                    font-size: 17px;
                    font-weight: 600;
                    line-height: 1.3;
                    min-height: 44px;
                }

                .performance-value {
                    font-size: 34px;
                    font-weight: 700;
                    line-height: 1;
                    margin-top: 10px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # =================================================
            # DESEMPEÑO GENERAL
            # =================================================

            st.markdown(
                "### Desempeño general"
            )

            metric_cols = st.columns(4)

            with metric_cols[0]:
                st.markdown(
                    f'<div class="performance-card">'
                    f'<div class="performance-title">Exactitud (Accuracy)</div>'
                    f'<div class="performance-value">'
                    f'{format_metric(accuracy, percentage=True)}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            with metric_cols[1]:
                st.markdown(
                    f'<div class="performance-card">'
                    f'<div class="performance-title">Exactitud balanceada</div>'
                    f'<div class="performance-value">'
                    f'{format_metric(balanced_accuracy, percentage=True)}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            with metric_cols[2]:
                st.markdown(
                    f'<div class="performance-card">'
                    f'<div class="performance-title">Área bajo ROC (ROC-AUC)</div>'
                    f'<div class="performance-value">'
                    f'{format_metric(roc_auc, percentage=True)}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            with metric_cols[3]:
                st.markdown(
                    f'<div class="performance-card">'
                    f'<div class="performance-title">Área bajo PR (PR-AUC)</div>'
                    f'<div class="performance-value">'
                    f'{format_metric(pr_auc, percentage=True)}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            # =================================================
            # CLASE 0
            # =================================================

            st.markdown(
                "### Desempeño — Clase 0: Sin diabetes"
            )

            metric_cols = st.columns(3)

            with metric_cols[0]:
                st.markdown(
                    f'<div class="performance-card">'
                    f'<div class="performance-title">Precisión</div>'
                    f'<div class="performance-value">'
                    f'{format_metric(precision_0, percentage=True)}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            with metric_cols[1]:
                st.markdown(
                    f'<div class="performance-card">'
                    f'<div class="performance-title">Sensibilidad (Recall)</div>'
                    f'<div class="performance-value">'
                    f'{format_metric(recall_0, percentage=True)}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            with metric_cols[2]:
                st.markdown(
                    f'<div class="performance-card">'
                    f'<div class="performance-title">Puntuación F1</div>'
                    f'<div class="performance-value">'
                    f'{format_metric(f1_0, percentage=True)}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            # =================================================
            # CLASE 1
            # =================================================

            st.markdown(
                "### Desempeño — Clase 1: Prediabetes / diabetes"
            )

            metric_cols = st.columns(3)

            with metric_cols[0]:
                st.markdown(
                    f'<div class="performance-card">'
                    f'<div class="performance-title">Precisión</div>'
                    f'<div class="performance-value">'
                    f'{format_metric(precision_1, percentage=True)}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            with metric_cols[1]:
                st.markdown(
                    f'<div class="performance-card">'
                    f'<div class="performance-title">Sensibilidad (Recall)</div>'
                    f'<div class="performance-value">'
                    f'{format_metric(recall_1, percentage=True)}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            with metric_cols[2]:
                st.markdown(
                    f'<div class="performance-card">'
                    f'<div class="performance-title">Puntuación F1</div>'
                    f'<div class="performance-value">'
                    f'{format_metric(f1_1, percentage=True)}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            # =================================================
            # ¿QUÉ SIGNIFICA CADA MÉTRICA?
            # =================================================

            st.markdown("---")

            st.subheader(
                "¿Qué significa cada métrica?"
            )

            st.write(
                """
                En este proyecto, la **clase 0** representa a las personas
                sin diabetes y la **clase 1** corresponde al grupo de
                prediabetes/diabetes. Cada métrica permite evaluar un aspecto
                diferente del desempeño del modelo.
                """
            )

            # =================================================
            # FILA 1
            # =================================================

            col1, col2 = st.columns(2)

            with col1:

                st.info(
                    f"""
                    **Exactitud (Accuracy)**

                    Indica qué porcentaje del total de personas fue clasificado
                    correctamente por el modelo.

                    Con un resultado de **{format_metric(accuracy, percentage=True)}**,
                    de cada 100 personas evaluadas, aproximadamente
                    **{accuracy * 100:.0f} fueron clasificadas correctamente**.
                    """
                )

            with col2:

                st.error(
                    f"""
                    **Exactitud balanceada (Balanced Accuracy)**

                    Mide el desempeño promedio del modelo en las dos clases,
                    dando la misma importancia a cada una.

                    Se presenta un desempeño promedio de aproximadamente
                    **{balanced_accuracy * 100:.0f} de cada 100 casos** al considerar
                    por igual ambas clases.
                    """
                )


            # =================================================
            # FILA 2 — CLASE 0
            # =================================================

            col1, col2 = st.columns(2)

            with col1:

                st.error(
                    f"""
                    **Precisión — Clase 0: Sin diabetes**

                    Indica, de todas las personas que el modelo clasificó como
                    **sin diabetes**, qué porcentaje realmente pertenecía a
                    esta clase.

                    Con una precisión de **{format_metric(precision_0, percentage=True)}**,
                    de cada 100 personas clasificadas como sin diabetes,
                    aproximadamente **{precision_0 * 100:.0f} realmente pertenecían
                    a este grupo**.
                    """
                )

            with col2:

                st.info(
                    f"""
                    **Sensibilidad (Recall) — Clase 0: Sin diabetes**

                    Indica qué porcentaje de las personas que realmente
                    pertenecían a la clase **sin diabetes** fue identificado
                    correctamente.

                    Con un resultado de **{format_metric(recall_0, percentage=True)}**,
                    de cada 100 personas que realmente pertenecían a esta clase,
                    aproximadamente **{recall_0 * 100:.0f} fueron bien identificadas**.
                    """
                )


            # =================================================
            # FILA 3 — CLASE 0 / CLASE 1
            # =================================================

            col1, col2 = st.columns(2)

            with col1:

                st.info(
                    f"""
                    **Puntuación F1 — Clase 0: Sin diabetes**

                    Combina la precisión y la sensibilidad de la clase 0
                    en una sola medida.

                    Con un resultado de **{format_metric(f1_0, percentage=True)}**,
                    el modelo presenta un equilibrio entre identificar
                    correctamente a las personas sin diabetes y realizar
                    predicciones precisas para esta clase.
                    """
                )

            with col2:

                st.error(
                    f"""
                    **Precisión — Clase 1: Prediabetes / diabetes**

                    Indica, de todos a los que se clasificaron cen
                    **prediabetes/diabetes**, qué porcentaje realmente pertenecía
                    a esta clase.

                    De cada 100 personas clasificadas como prediabetes/diabetes,
                    aproximadamente **{precision_1 * 100:.0f} realmente pertenecían
                    a este grupo**.
                    """
                )


            # =================================================
            # FILA 4 — CLASE 1
            # =================================================

            col1, col2 = st.columns(2)

            with col1:

                st.error(
                    f"""
                    **Sensibilidad (Recall) — Clase 1: Prediabetes / diabetes**

                    Indica qué porcentaje de las personas que realmente
                    pertenecían al grupo de **prediabetes/diabetes** fue bien identificado.

                    De cada 100 casos que pertenecían a la clase 1,
                    aproximadamente **{recall_1 * 100:.0f} fueron bien identificados**.
                    """
                )

            with col2:

                st.info(
                    f"""
                    **Puntuación F1 — Clase 1: Prediabetes / diabetes**

                    Combina la precisión y la sensibilidad de la clase 1
                    en una sola medida.

                    Con un resultado de **{format_metric(f1_1, percentage=True)}**,
                    esta métrica resume el equilibrio entre identificar
                    correctamente los casos de prediabetes/diabetes y
                    evitar clasificaciones incorrectas.
                    """
                )


            # =================================================
            # FILA 5 — MÉTRICAS DE DISCRIMINACIÓN
            # =================================================

            col1, col2 = st.columns(2)

            with col1:

                st.info(
                    f"""
                    **Área bajo la curva ROC (ROC-AUC)**

                    Mide la capacidad del modelo para diferenciar entre
                    las dos clases utilizando las probabilidades que genera.

                    Con un resultado de **{format_metric(roc_auc, percentage=True)}**,
                    el modelo presenta una buena capacidad para distinguir
                    entre personas sin diabetes y personas pertenecientes
                    al grupo de prediabetes/diabetes.
                    """
                )

            with col2:

                st.error(
                    f"""
                    **Área bajo la curva Precisión-Recall (PR-AUC)**

                    Evalúa el comportamiento del modelo al identificar
                    principalmente la clase 1, considerando conjuntamente
                    la precisión y la sensibilidad.

                    Con un resultado de **{format_metric(pr_auc, percentage=True)}**,
                    esta métrica resume el desempeño del modelo en la
                    identificación de personas pertenecientes al grupo
                    de prediabetes/diabetes.
                    """
                )

        else:

            st.info(
                """
                El archivo de métricas existe, pero no fue posible
                identificar automáticamente las métricas del modelo
                CatBoost A.
                """
            )

    else:

        st.warning(
            "No se encontró metrics_all_models.json."
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

        sexo_opcion = st.selectbox(
            "Sexo",
            ["Mujer", "Hombre"],
        )

        sexo = 0 if sexo_opcion == "Mujer" else 1

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
            format_func=lambda x: {
                1: "18–24 años",
                2: "25–29 años",
                3: "30–34 años",
                4: "35–39 años",
                5: "40–44 años",
                6: "45–49 años",
                7: "50–54 años",
                8: "55–59 años",
                9: "60–64 años",
                10: "65–69 años",
                11: "70–74 años",
                12: "75–79 años",
                13: "80 años o más",
            }[x],
        )

    with col2:

                education = st.selectbox(
                    "Nivel educativo",
                    list(range(1, 7)),
                    format_func=lambda x: {
                        1: "Sin escolaridad / jardín",
                        2: "Primaria",
                        3: "Secundaria incompleta",
                        4: "Bachillerato completo",
                        5: "Universidad o técnica (1–3 años)",
                        6: "Universidad (4+ años)",
                    }[x],
                )

    with col3:

        income = st.selectbox(
            "Nivel de ingresos",
            list(range(1, 9)),
            format_func=lambda x: {
                1: "Menos de $10.000",
                2: "$10.000 – $14.999",
                3: "$15.000 – $19.999",
                4: "$20.000 – $24.999",
                5: "$25.000 – $34.999",
                6: "$35.000 – $49.999",
                7: "$50.000 – $74.999",
                8: "$75.000 o más",
            }[x],
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
        "Sex": sexo,
        "GenHlth": gen_hlth,
        "Age": age,
        "Education": education,
        "Income": income,
        "BMI": bmi,
        "MentHlth": ment_hlth,
        "PhysHlth": phys_hlth,
    }

    if st.button(
        "Generar predicción",
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