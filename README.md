# Predicción de diabetes a partir de indicadores de salud y estilo de vida

Proyecto final de Máquina de Aprendizaje 1  
Universidad de La Sabana

**Integrantes:**
- Jorge Esteban Díaz Bernal
- Carmen Celeste Durán Báez
- Mariana Salas Gutiérrez

---

# 1. Contexto del problema

La diabetes constituye un importante problema de salud pública a nivel
mundial. Su presencia está relacionada con diferentes factores de salud,
estilo de vida y características sociodemográficas.

De acuerdo con la Federación Internacional de Diabetes (IDF), 589 millones
de adultos entre 20 y 79 años viven con diabetes a nivel mundial, lo que
equivale aproximadamente a 1 de cada 9 adultos.

Para este proyecto se utiliza el conjunto de datos **CDC Diabetes Health
Indicators**, disponible en el UCI Machine Learning Repository. El conjunto
contiene 253.680 observaciones y 21 variables predictoras relacionadas con
indicadores de salud, estilo de vida y características sociodemográficas.

El problema se aborda como una tarea de **clasificación supervisada**. La
variable objetivo se transforma en una clasificación binaria:

- **Clase 0:** sin diabetes.
- **Clase 1:** prediabetes o diabetes.

El objetivo es utilizar las variables disponibles para identificar patrones
y realizar una clasificación de los individuos según la clase a la que
pertenecen.

**Fuentes:**

- International Diabetes Federation (IDF):  
  https://diabetesatlas.org/es/data-by-location/global/

- UCI Machine Learning Repository:  
  https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

---

# 2. Objetivo

Desarrollar un modelo de Machine Learning que permita clasificar individuos
según su probabilidad de pertenecer a la clase asociada con prediabetes o
diabetes, utilizando indicadores de salud, estilo de vida y características
sociodemográficas.

El proyecto sigue las etapas de la metodología **CRISP-DM**:

1. Comprensión del negocio.
2. Comprensión de los datos.
3. Preparación de los datos.
4. Modelado.
5. Evaluación.
6. Interpretación.

El modelo final utilizado es **CatBoost**, un algoritmo de Machine Learning
basado en árboles de decisión y Gradient Boosting.

El modelo utiliza las **21 variables predictoras** disponibles en el conjunto
de datos y emplea un umbral de clasificación de **0,51**.

Los principales resultados obtenidos fueron:

- **Accuracy:** 71,90%
- **Balanced Accuracy:** 73,25%
- **ROC-AUC:** 81,13%
- **PR-AUC:** 46,97%
- **Recall clase 1:** 75,32%
- **Precision clase 1:** 35,29%
- **F1-score clase 1:** 48,06%

---

# 3. Cómo correr este proyecto en el repositorio

## 3.1 Clonar el repositorio

Desde una terminal ejecutar:

```bash
git clone https://github.com/MariSalas23/aprendizaje-maquina-proyecto.git
```

Ingresar a la carpeta del proyecto:

```bash
cd aprendizaje-maquina-proyecto
```

---

## 3.2 Crear el entorno virtual

Se recomienda utilizar **Python 3.12**.

Crear el entorno virtual:

```bash
python -m venv .venv
```

### Windows

Activar el entorno:

```bash
.venv\Scripts\activate
```

### macOS / Linux

Activar el entorno:

```bash
source .venv/bin/activate
```

---

## 3.3 Instalar las dependencias

Con el entorno virtual activado:

```bash
pip install -r requirements.txt
```

---

## 3.4 Ejecutar `main.py`

El archivo `main.py` corresponde al pipeline principal del proyecto.

Para ejecutarlo, desde la carpeta raíz del repositorio:

```bash
python main.py
```

---

## 3.5 Ejecutar `app.py`

La interfaz de usuario se encuentra en:

```text
ui/app.py
```

Para ejecutar la aplicación:

```bash
streamlit run ui/app.py
```

Después de ejecutar el comando, Streamlit proporcionará una dirección
local para acceder a la aplicación desde el navegador.

---

# 4. Dashboard

Para ver el dashboard, ingresar al siguiente link:

[(https://marisalas23-aprendizaje-maquina-proyecto-uiapp-1bhvhq.streamlit.app](https://marisalas23-aprendizaje-maquina-proyecto-uiapp-1bhvhq.streamlit.app/)

---

# 5. Metodología CRISP-DM

El desarrollo del proyecto se realizó siguiendo la metodología CRISP-DM,
abordando de manera estructurada las diferentes etapas del proceso de
Machine Learning.

## 5.1 Comprensión del problema

Se definió como problema de análisis la identificación de patrones
relacionados con la presencia de diabetes o prediabetes a partir de
indicadores de salud, estilo de vida y características sociodemográficas.

El problema se plantea como una tarea de clasificación supervisada, donde
el objetivo es clasificar a los individuos en dos categorías.

## 5.2 Comprensión de los datos

Se utilizó el conjunto de datos **CDC Diabetes Health Indicators**, compuesto
por 253.680 observaciones y 21 variables predictoras.

Las variables contienen información relacionada con:

- Indicadores de salud.
- Estilo de vida.
- Acceso a servicios de salud.
- Características sociodemográficas.

La variable objetivo utilizada en el proyecto corresponde a una clasificación
binaria:

- **Clase 0:** sin diabetes.
- **Clase 1:** prediabetes o diabetes.

Durante esta etapa se realizó un análisis exploratorio de la estructura de
los datos, tipos de variables, valores faltantes, valores duplicados,
distribuciones y relaciones entre variables.

## 5.3 Preparación de los datos

En esta etapa se realizaron los procesos necesarios para preparar la
información para el modelado.

Se realizaron actividades como:

- Revisión de la calidad de los datos.
- Identificación de valores faltantes.
- Revisión de registros duplicados.
- Clasificación de las variables según su naturaleza.
- Preparación de las variables binarias.
- Preparación de las variables ordinales.
- Preparación de las variables numéricas.
- Transformación de la variable objetivo a una clasificación binaria.
- Selección de las variables utilizadas por el modelo.
- Preparación de los datos para el entrenamiento y evaluación.

## 5.4 Modelado

Se evaluaron diferentes modelos de clasificación durante el proceso de
Machine Learning.

El modelo seleccionado para la solución final fue **CatBoost**.

CatBoost es un algoritmo basado en árboles de decisión que utiliza Gradient
Boosting para combinar múltiples árboles y realizar predicciones.

La selección del modelo se realizó considerando el desempeño obtenido en las
métricas de evaluación y la capacidad del modelo para trabajar con las
características de las variables utilizadas.

El modelo final utiliza las **21 variables predictoras**.

## 5.5 Evaluación

El modelo final fue evaluado utilizando diferentes métricas de clasificación,
considerando tanto el desempeño general como el comportamiento de cada clase.

### Métricas generales

- **Accuracy:** 71,90%
- **Balanced Accuracy:** 73,25%
- **ROC-AUC:** 81,13%
- **PR-AUC:** 46,97%

### Clase 0 — Sin diabetes

- **Precision:** 93,25%
- **Recall:** 71,18%
- **F1-score:** 80,74%

### Clase 1 — Prediabetes / diabetes

- **Precision:** 35,29%
- **Recall:** 75,32%
- **F1-score:** 48,06%

Para la clasificación final se utilizó un umbral de **0,51**.

Este umbral permite clasificar una observación como clase 1 cuando la
probabilidad predicha por el modelo es igual o superior a 0,51.

## 5.6 Interpretación

Los resultados muestran un comportamiento diferente entre las dos clases.

Para la clase 1, correspondiente a prediabetes o diabetes, el modelo alcanza
un **Recall de 75,32%**. Esto significa que identifica correctamente una
proporción importante de las observaciones pertenecientes a esta clase.

Sin embargo, la **Precision de 35,29%** muestra que una parte importante de
las observaciones clasificadas como clase 1 pertenece realmente a la clase 0.

Para la clase 0, el modelo obtiene una **Precision de 93,25%** y un
**F1-score de 80,74%**, mostrando un mejor desempeño en la identificación de
esta clase.

Por lo tanto, el modelo presenta una capacidad relevante para identificar
casos pertenecientes a la clase 1, aunque con una cantidad importante de
falsos positivos.

---

# 6. Estructura del proyecto

El repositorio está organizado de acuerdo con las diferentes etapas y
componentes del proyecto de Machine Learning.

```text
aprendizaje-maquina-proyecto/
│
├── config/
│   ├── __init__.py
│   └── config.py
│
├── data/
│   └── diabetes_012_health_indicators_BRFSS2015.csv
│
├── models/
│   └── modelo_diabetes_catboost_final.joblib
│
├── notebooks/
│   └── CDC_Diabetes_Health_Indicators_.ipynb
│
├── reports/
│   ├── comparison/
│   ├── exploratory_analysis_and_features/
│   └── final_model/
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── interpretation/
│   ├── modeling/
│   └── preparation/
│
├── ui/
│   └── app.py
│
├── main.py
├── requirements.txt
├── README.md
├── Dockerfile
└── .gitignore
```
### Descripción de la estructura

**`config/`**

Contiene los archivos relacionados con la configuración general del
proyecto.

**`data/`**

Contiene el conjunto de datos utilizado para el desarrollo del proyecto.

**`models/`**

Contiene el modelo final entrenado utilizado para realizar las predicciones.

**`notebooks/`**

Contiene el notebook utilizado durante el proceso de exploración y análisis
de los datos.

**`reports/`**

Contiene los resultados generados durante las diferentes etapas del
proyecto, incluyendo el análisis exploratorio, la comparación de modelos y
los resultados del modelo final.

**`src/`**

Contiene el código fuente utilizado durante las diferentes etapas del
proceso de Machine Learning.

Las principales carpetas son:

- `data/`: procesos relacionados con los datos.
- `preparation/`: preparación y transformación de los datos.
- `modeling/`: construcción y entrenamiento de modelos.
- `evaluation/`: evaluación de los modelos.
- `interpretation/`: interpretación de los resultados.

**`ui/`**

Contiene la interfaz de usuario desarrollada en Streamlit.

El archivo principal es `app.py`.

**`main.py`**

Contiene el pipeline principal del proyecto y permite ejecutar el proceso
principal de Machine Learning.

**`requirements.txt`**

Contiene las librerías y dependencias necesarias para ejecutar el proyecto.

**`README.md`**

Contiene la documentación general del proyecto y las instrucciones
necesarias para reproducirlo.

**`.gitignore`**

Contiene los archivos y carpetas que no deben ser incluidos en el
repositorio.

---

# 7. Decisiones técnicas

El proyecto se desarrolló como un problema de clasificación supervisada,
utilizando como variable objetivo una clasificación binaria.

La clase 0 corresponde a individuos sin diabetes, mientras que la clase 1
agrupa las observaciones correspondientes a prediabetes o diabetes.

Se utilizaron las 21 variables predictoras disponibles en el conjunto de
datos, relacionadas con indicadores de salud, estilo de vida, acceso a
servicios de salud y características sociodemográficas.

Para el modelo final se seleccionó **CatBoost**, un algoritmo basado en
árboles de decisión y Gradient Boosting.

La selección del modelo se realizó a partir de la evaluación de diferentes
modelos de clasificación y de sus resultados en las métricas de desempeño.

Para evaluar el modelo se utilizaron diferentes métricas, incluyendo
Accuracy, Balanced Accuracy, ROC-AUC, PR-AUC, Precision, Recall y
F1-score.

Para la clasificación final se estableció un umbral de **0,51**.

---

# 8. Resultados principales

El modelo final seleccionado fue **CatBoost**, utilizando las 21 variables
predictoras.

Los principales resultados obtenidos fueron:

| Métrica | Resultado |
|---|---:|
| Accuracy | 71,90% |
| Balanced Accuracy | 73,25% |
| ROC-AUC | 81,13% |
| PR-AUC | 46,97% |
| Precision clase 0 | 93,25% |
| Recall clase 0 | 71,18% |
| F1-score clase 0 | 80,74% |
| Precision clase 1 | 35,29% |
| Recall clase 1 | 75,32% |
| F1-score clase 1 | 48,06% |

El modelo utiliza un umbral de clasificación de **0,51**.

El resultado más relevante para la clase 1 es el **Recall de 75,32%**,
indicando que el modelo identifica una proporción importante de las
observaciones pertenecientes a la clase de prediabetes o diabetes.

---

# 9. Interpretación de los resultados

El desempeño del modelo presenta diferencias entre las dos clases.

Para la **clase 1 (prediabetes / diabetes)**, el modelo obtiene un Recall
de **75,32%**, lo que indica que identifica correctamente una proporción
importante de las observaciones pertenecientes a esta clase.

Sin embargo, la Precision de **35,29%** indica que existe una proporción
considerable de predicciones de clase 1 que corresponden realmente a
observaciones de la clase 0.

Para la **clase 0 (sin diabetes)**, el modelo obtiene una Precision de
**93,25%** y un F1-score de **80,74%**, mostrando un mejor desempeño para
esta clase.

Estos resultados muestran que el modelo tiene una capacidad relevante para
identificar individuos pertenecientes a la clase 1, aunque presenta una
cantidad importante de falsos positivos.

---

# 10. Limitaciones

Los resultados obtenidos dependen de las características y calidad del
conjunto de datos utilizado.

El modelo identifica patrones presentes en los datos, pero no establece
relaciones causales entre las variables y la presencia de diabetes.

La variable objetivo utilizada en este proyecto agrupa en la clase 1 las
observaciones correspondientes a **prediabetes o diabetes**.

Además, la Precision de 35,29% para la clase 1 muestra que existe una
proporción importante de falsos positivos.

Por tratarse de un proyecto académico, las predicciones generadas por el
modelo no deben interpretarse como un diagnóstico médico.

---

# 11. Conclusiones

El proyecto permitió aplicar técnicas de Machine Learning para analizar
indicadores de salud, estilo de vida y características sociodemográficas
relacionados con la presencia de diabetes o prediabetes.

El desarrollo bajo la metodología CRISP-DM permitió estructurar las
diferentes etapas del proyecto, desde la comprensión del problema y los
datos hasta la preparación, modelado, evaluación e interpretación.

El modelo final basado en **CatBoost** obtuvo un ROC-AUC de **81,13%** y un
Recall de **75,32% para la clase 1**, mostrando una capacidad relevante
para identificar observaciones pertenecientes al grupo de prediabetes o
diabetes.

Sin embargo, la Precision de **35,29% para la clase 1** evidencia una
limitación en la precisión de las predicciones positivas.

En consecuencia, el modelo debe entenderse como una herramienta de análisis
y predicción dentro del contexto académico del proyecto y no como una
herramienta de diagnóstico médico.

---

# 12. Referencias

### International Diabetes Federation — IDF Diabetes Atlas

[Diabetes Atlas - Datos globales](https://diabetesatlas.org/es/data-by-location/global/)

### UCI Machine Learning Repository — CDC Diabetes Health Indicators

[CDC Diabetes Health Indicators - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

---
