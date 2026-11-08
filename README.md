# VectorIA

## Sistema Inteligente para la Vigilancia del Dengue

VectorIA es una plataforma de inteligencia artificial desarrollada para apoyar la vigilancia epidemiológica del dengue mediante la integración de tres modelos especializados:

- CNN para clasificación de especies de mosquitos.
- LSTM para pronóstico de casos de dengue.
- ANN para recomendación de acciones preventivas.

El sistema integra los tres modelos mediante una API desarrollada con FastAPI y un dashboard interactivo implementado con Streamlit.

---

# Objetivo General

Desarrollar una herramienta inteligente capaz de:

1. Identificar la especie de mosquito a partir de una fotografía.
2. Pronosticar la cantidad de casos de dengue para las próximas semanas.
3. Recomendar acciones preventivas según el nivel de riesgo epidemiológico.
4. Facilitar la visualización y exploración de información histórica sobre dengue.

---

# Arquitectura General

```text
Imagen del mosquito
         │
         ▼
      CNN (MobileNetV2)
         │
         ▼
 Especie detectada
         │
         ▼
      LSTM
         │
         ▼
 Casos pronosticados
         │
         ▼
       ANN
         │
         ▼
 Acción preventiva recomendada
```

---

# Estructura del Proyecto

```text
VectorIA/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── app/
│   ├── Home.py
│   └── assets/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── modelo_final_mosquitos.keras
│   ├── modelo_lstm_dengue.keras
│   ├── modelo_ann_acciones.keras
│   └── ann_config.json
│
├── notebooks/
│
├── src/
│   ├── clasificador.py
│   ├── pronosticador.py
│   ├── recomendador.py
│   └── train/
│
├── Uso_Responsable_IA.md
│
└── README.md
```

---

# Fuentes de Datos

## Mosquitos

### Kaggle

Dataset utilizado para:

- Aedes aegypti
- Aedes albopictus

### IEEE Dataport

Dataset utilizado para:

- Culex

### Mendeley

Dataset complementario utilizado para validación externa.

Las imágenes aumentadas fueron excluidas del entrenamiento para evitar fuga de información (data leakage).

---

## Dengue

### Datos nacionales

Archivo:

```text
vectoria_dengue_nacional_real.csv
```

Variables utilizadas:

- Casos de dengue
- Tasa por 100 000 habitantes
- Lluvia
- Temperatura máxima
- Temperatura mínima
- Humedad relativa

### Datos semanales

Archivo:

```text
vectoria_dengue_semanal.csv
```

Utilizado para el modelo LSTM.

### Datos cantonales

Archivo:

```text
vectoria_casos_cantonal_anual_real.csv
```

Utilizado para análisis histórico dentro del dashboard.

---

# Proceso de Desarrollo

## 1. Exploración de Datos (EDA)

Se realizaron análisis exploratorios para:

- Identificar valores faltantes.
- Analizar tendencias temporales.
- Evaluar correlaciones climáticas.
- Detectar desbalance de clases.
- Analizar la distribución geográfica del dengue.

---

## 2. Preprocesamiento

Durante la preparación de los datos se realizaron:

- Limpieza de registros.
- Eliminación de duplicados.
- Construcción de variables rezagadas (lags).
- Escalado de variables.
- Construcción de ventanas temporales.
- División train, validation y test.

---

# Modelo 1: Clasificación de Mosquitos (CNN)

## Arquitectura

Modelo final:

```text
MobileNetV2
```

Guardado en:

```text
models/modelo_final_mosquitos.keras
```

## Clases Clasificadas

- Aedes aegypti
- Aedes albopictus
- Anopheles
- Culex

## Resultados

Resultados aproximados obtenidos durante la evaluación:

```text
Accuracy ≈ 93%
F1 Macro ≈ 94%
```

---

# Modelo 2: Pronóstico de Casos de Dengue (LSTM)

## Objetivo

Pronosticar la cantidad de casos de dengue cuatro semanas hacia el futuro utilizando datos epidemiológicos y climáticos históricos.

## Configuración

```text
Ventana temporal: 26 registros
Variables utilizadas: 25
Horizonte de predicción: 4 semanas
```

## Resultados

```text
MAE ≈ 22.97
RMSE ≈ 34.08
```

Modelo almacenado en:

```text
models/modelo_lstm_dengue.keras
```

---

# Modelo 3: Recomendación Preventiva (ANN)

## Acciones Recomendadas

- Monitoreo normal
- Eliminación de criaderos
- Fumigación focalizada
- Alerta sanitaria

## Variables Utilizadas

- ratio_casos_umbral
- lluvia_14d_clip
- temp_max
- aegypti_detectado

## Modelo

Guardado en:

```text
models/modelo_ann_acciones.keras
```

Configuración de inferencia:

```text
models/ann_config.json
```

---

# API FastAPI

La API integra los tres modelos mediante los siguientes endpoints:

## Estado general

```http
GET /health
```

## Fechas disponibles para pronóstico

```http
GET /dengue/dates
```

## Clasificación de mosquito

```http
POST /mosquito/classify
```

## Pronóstico de dengue

```http
POST /dengue/predict
```

## Recomendación preventiva

```http
POST /action/recommend
```

## Flujo integrado

```http
POST /integrated/analyze
```

---

# Dashboard Streamlit

Funcionalidades implementadas:

## Clasificación de mosquito

- Carga de imágenes.
- Clasificación automática.
- Nivel de confianza.
- Probabilidades por clase.

## Pronóstico de dengue

- Selección de fecha.
- Generación de predicciones.
- Visualización gráfica.

## Recomendación preventiva

- Variables climáticas.
- Umbral epidemiológico.
- Acción sugerida.
- Explicación automática.

## Análisis histórico

- Casos por provincia.
- Ranking de cantones.
- Filtros por año.

---

# Instalación

Crear entorno virtual:

```bash
python -m venv .venv
```

Activar entorno:

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

# Ejecución

## API

```bash
python -m uvicorn api.main:app --reload
```

Documentación:

```text
http://127.0.0.1:8000/docs
```

## Dashboard

```bash
streamlit run app/Home.py
```

Dashboard:

```text
http://localhost:8501
```

---

# Documentación Complementaria

El proyecto incorpora documentación adicional:

- README.md
- Uso_Responsable_IA.md

Estos documentos describen el funcionamiento general del sistema y las consideraciones éticas relacionadas con el uso de inteligencia artificial.

---

# Tecnologías Utilizadas

- Python
- TensorFlow
- Keras
- FastAPI
- Streamlit
- Plotly
- Pandas
- NumPy
- Scikit-learn

---

# Integrantes

Sebastián Calvo Solano
Carlos Cambronero Zuñiga
Josué Francisco Redondo Gómez
Steven Zamora Reyes
---

# Consideraciones Éticas

VectorIA es una herramienta académica diseñada para apoyar actividades de vigilancia epidemiológica.

- No diagnostica dengue en personas.
- No sustituye decisiones médicas.
- No reemplaza sistemas oficiales de vigilancia sanitaria.
- Debe utilizarse únicamente con fines educativos, demostrativos o de investigación.

Para más información consultar:

```text
Uso_Responsable_IA.md
```

---