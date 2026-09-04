import pandas as pd
import joblib

from catboost import Pool

from config.config import (
    DATA_FILE,
    FINAL_MODEL_FILE,
    TARGET,
    BINARY_COLS,
    ORDINAL_COLS,
    NUMERIC_COLS,
)

from src.preparation.cleaning import clean_dataset
from src.preparation.data_partition import split_data
from src.preparation.preprocessing import prepare_catboost_data


# ============================================================
# 1. CARGAR MODELO
# ============================================================

model = joblib.load(FINAL_MODEL_FILE)

print("Modelo cargado:")
print(FINAL_MODEL_FILE)


# ============================================================
# 2. CARGAR Y LIMPIAR EXACTAMENTE IGUAL
# ============================================================

df = pd.read_csv(DATA_FILE)

df = clean_dataset(df)

features = BINARY_COLS + ORDINAL_COLS + NUMERIC_COLS


# ============================================================
# 3. HACER EXACTAMENTE EL MISMO SPLIT
# ============================================================

X_train, X_test, y_train, y_test = split_data(
    df=df,
    features=features,
    target=TARGET,
)


# ============================================================
# 4. PREPARAR EXACTAMENTE COMO predictions.py
# ============================================================

(
    X_train_catboost,
    X_test_catboost,
    categorical_features,
) = prepare_catboost_data(
    X_train,
    X_test,
    features,
)


# ============================================================
# 5. PREDECIR TODO EL TEST
# ============================================================

test_pool = Pool(
    data=X_test_catboost,
    cat_features=categorical_features,
)

probabilities = model.predict_proba(test_pool)[:, 1]

predictions = (probabilities >= 0.51).astype(int)


# ============================================================
# 6. BUSCAR EL INDIVIDUO
# ============================================================

persona = {
    "HighBP": 0,
    "HighChol": 1,
    "CholCheck": 1,
    "Smoker": 1,
    "Stroke": 1,
    "HeartDiseaseorAttack": 1,
    "PhysActivity": 1,
    "Fruits": 0,
    "Veggies": 0,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "DiffWalk": 0,
    "Sex": 1,
    "GenHlth": 3,
    "Age": 10,
    "Education": 4,
    "Income": 3,
    "BMI": 31,
    "MentHlth": 0,
    "PhysHlth": 30,
}

mask = pd.Series(True, index=X_test.index)

for col, value in persona.items():
    mask &= X_test[col] == value


coincidencias = X_test.loc[mask]

print("\n========================================")
print("COINCIDENCIAS EN X_TEST")
print("========================================")

print(coincidencias)

print("\nCantidad de coincidencias:", len(coincidencias))


# ============================================================
# 7. MOSTRAR LA PREDICCIÓN REAL DEL MODELO
# ============================================================

for idx in coincidencias.index:

    posicion = X_test.index.get_loc(idx)

    print("\n========================================")
    print("RESULTADO")
    print("========================================")

    print("Índice:", idx)

    print(
        "Probabilidad:",
        probabilities[posicion]
    )

    print(
        "Porcentaje:",
        probabilities[posicion] * 100
    )

    print(
        "Predicción:",
        predictions[posicion]
    )

    print(
        "Valor real:",
        y_test.loc[idx]
    )