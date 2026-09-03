from pathlib import Path


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ============================================================
# CARPETAS DE REPORTES
# ============================================================

EDA_FIGURES_DIR = (
    REPORTS_DIR
    / "exploratory_analysis_and_features"
    / "figures"
)

COMPARISON_FIGURES_DIR = (
    REPORTS_DIR
    / "comparison"
    / "figures"
)

FINAL_MODEL_FIGURES_DIR = (
    REPORTS_DIR
    / "final_model"
    / "figures"
)


# ============================================================
# ARCHIVOS
# ============================================================

DATA_FILE = (
    DATA_DIR
    / "diabetes_012_health_indicators_BRFSS2015.csv"
)

FINAL_METRICS_FILE = (
    REPORTS_DIR
    / "final_model"
    / "final_metrics.json"
)

FINAL_MODEL_FILE = (
    MODELS_DIR
    / "modelo_diabetes_catboost_final.joblib"
)


# ============================================================
# VARIABLE OBJETIVO
# ============================================================

TARGET = "Diabetes_binary"


# ============================================================
# CLASIFICACIÓN DE VARIABLES
# Basada en el notebook
# ============================================================

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


# ============================================================
# VARIABLES SELECCIONADAS EN EL EDA
# ============================================================

SELECTED_FEATURES = [
    "GenHlth",
    "HighBP",
    "BMI",
    "DiffWalk",
    "HighChol",
    "Age",
    "HeartDiseaseorAttack",
    "Income",
    "PhysHlth",
    "Education",
    "PhysActivity",
    "Stroke",
]


# ============================================================
# CONFIGURACIÓN DEL MODELO / PARTICIÓN
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20