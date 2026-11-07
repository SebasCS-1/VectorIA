"""
ann.py · Módulo 3: ANN multiclase para recomendar la acción preventiva
======================================================================

Entrena una red neuronal que, a partir del pronóstico de casos, el clima y la
detección de Aedes aegypti, recomienda una de cuatro acciones de control vectorial:

    monitoreo_normal · eliminacion_criaderos · fumigacion_focalizada · alerta_sanitaria

Es la versión en script del notebook ``notebooks/05_ANN.ipynb`` (mismo pipeline,
mismos hiperparámetros). El notebook explica cada decisión; este archivo sirve para
reentrenar el modelo sin abrir Jupyter.

Uso (desde la raíz del repositorio):
    python src/train/ann.py
    python src/train/ann.py --epochs 100        # entrenamiento más corto

Entradas:
    data/raw/Acciones/vectoria_acciones.csv

Salidas:
    models/modelo_ann_acciones.keras   -> modelo entrenado
    models/ann_config.json             -> escalador, orden de variables y clases
                                          (lo lee src/recomendador.py)
    data/processed/*_ann.csv           -> splits ya escalados (para inspección)
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

# --------------------------------------------------------------------------- #
# 1. Constantes del módulo
# --------------------------------------------------------------------------- #
SEED = 42

# Rutas relativas a la raíz del repositorio (este archivo está en src/train/)
RAIZ = Path(__file__).resolve().parents[2]
RUTA_CSV = RAIZ / "data" / "raw" / "Acciones" / "vectoria_acciones.csv"
CARPETA_PROCESADOS = RAIZ / "data" / "processed"
CARPETA_MODELOS = RAIZ / "models"
RUTA_MODELO = CARPETA_MODELOS / "modelo_ann_acciones.keras"
RUTA_CONFIG = CARPETA_MODELOS / "ann_config.json"

# Clases ordenadas de menor a mayor severidad (el índice es la etiqueta numérica)
CLASES = [
    "monitoreo_normal",
    "eliminacion_criaderos",
    "fumigacion_focalizada",
    "alerta_sanitaria",
]
MAPA_CLASES = {nombre: i for i, nombre in enumerate(CLASES)}

# Umbrales de las reglas de etiquetado (diapositiva "Módulo 3" del proyecto)
UMBRALES_REGLAS = {
    "factor_alerta": 2.0,     # casos_pred > 2 x umbral endémico  -> alerta
    "lluvia_14d_mm": 80.0,    # lluvia 14 días > 80 mm            -> criaderos
    "temp_max_c": 27.0,       # temperatura máxima > 27 °C        -> criaderos
}

# Recorte de valores extremos (outliers) antes de escalar. Más allá de estos
# valores la acción no cambia (ratio > 2 ya es alerta; lluvia >> 80 ya es "alta"),
# pero sí comprimen la escala justo donde están los umbrales importantes.
CLIP_RATIO = 4.0
CLIP_LLUVIA_MM = 400.0

# Variables de entrada de la red (en este orden)
FEATURES = ["ratio_casos_umbral", "lluvia_14d_clip", "temp_max", "aegypti_detectado"]

# Hiperparámetros
EPOCHS = 200
BATCH_SIZE = 32
LR_INICIAL = 2e-3


# --------------------------------------------------------------------------- #
# 2. Datos
# --------------------------------------------------------------------------- #
def cargar_datos(ruta: Path = RUTA_CSV) -> pd.DataFrame:
    """Lee el CSV y convierte la fecha a datetime."""
    df = pd.read_csv(ruta)
    df["fecha_inicio_semana"] = pd.to_datetime(df["fecha_inicio_semana"])
    return df


def preparar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las variables derivadas y la etiqueta numérica.

    - ratio_raw:           casos pronosticados / umbral endémico (sin recortar)
    - ratio_casos_umbral:  el mismo cociente recortado a CLIP_RATIO
    - lluvia_14d_clip:     lluvia de 14 días recortada a CLIP_LLUVIA_MM
    - y:                   etiqueta numérica (0..3) según MAPA_CLASES
    """
    df = df.copy()
    df["ratio_raw"] = df["casos_pred_4sem"] / df["casos_umbral_endemico"]
    df["ratio_casos_umbral"] = df["ratio_raw"].clip(upper=CLIP_RATIO)
    df["lluvia_14d_clip"] = df["lluvia_14d_mm"].clip(upper=CLIP_LLUVIA_MM)
    df["y"] = df["recomendacion"].map(MAPA_CLASES)
    return df


def dividir_por_semana(df: pd.DataFrame, test_size: float, seed: int = SEED):
    """Divide agrupando por semana: las 5 filas (escenarios) de una misma semana
    caen siempre del mismo lado. Si se dividiera al azar por fila, la ANN vería en
    entrenamiento casi la misma información que luego se evalúa (fuga de datos)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_a, idx_b = next(gss.split(df, groups=df["fecha_inicio_semana"]))
    return df.iloc[idx_a], df.iloc[idx_b]


def escalar(df_train, df_val, df_test):
    """Estandariza (media 0, desviación 1). El escalador se ajusta SOLO con train."""
    escalador = StandardScaler().fit(df_train[FEATURES])

    def transformar(d):
        return pd.DataFrame(escalador.transform(d[FEATURES]), columns=FEATURES, index=d.index)

    return escalador, transformar(df_train), transformar(df_val), transformar(df_test)


def calcular_pesos(y_train) -> dict:
    """Pesos 'balanced' para compensar el desbalance de clases."""
    clases = np.unique(y_train)
    pesos = compute_class_weight(class_weight="balanced", classes=clases, y=y_train)
    return dict(zip(clases.tolist(), pesos))


# --------------------------------------------------------------------------- #
# 3. Modelo
# --------------------------------------------------------------------------- #
def construir_modelo(n_entradas: int, n_clases: int, pasos_totales: int) -> Sequential:
    """ANN densa 128-64-32 con softmax.

    La tasa de aprendizaje decrece con coseno desde LR_INICIAL: al inicio aprende
    rápido y al final afina las fronteras (importante porque las reglas son cortes
    exactos, por ejemplo temp_max > 27).
    """
    modelo = Sequential([
        Input(shape=(n_entradas,)),
        Dense(128, activation="relu"),
        Dropout(0.1),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(n_clases, activation="softmax"),   # una probabilidad por acción
    ])
    tasa = CosineDecay(initial_learning_rate=LR_INICIAL, decay_steps=pasos_totales, alpha=0.01)
    modelo.compile(
        loss="sparse_categorical_crossentropy",   # etiquetas enteras (0..3)
        optimizer=Adam(learning_rate=tasa),
        metrics=["accuracy"],
    )
    return modelo


def entrenar(X_train, y_train, X_val, y_val, pesos, epochs=EPOCHS, verbose=0):
    """Entrena sin EarlyStopping: con 3 % de ruido en las etiquetas, val_loss es
    inestable y detiene el entrenamiento antes de afinar las fronteras."""
    pasos = math.ceil(len(X_train) / BATCH_SIZE) * epochs
    modelo = construir_modelo(X_train.shape[1], len(CLASES), pasos)
    historial = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        class_weight=pesos,
        verbose=verbose,
    )
    return modelo, historial


# --------------------------------------------------------------------------- #
# 4. Evaluación y guardado
# --------------------------------------------------------------------------- #
def evaluar(modelo, X_test, y_test) -> dict:
    """Imprime el reporte por clase y la matriz de confusión; devuelve el reporte."""
    y_pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, labels=range(len(CLASES)),
                                target_names=CLASES, zero_division=0))
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(confusion_matrix(y_test, y_pred, labels=range(len(CLASES))))
    return classification_report(y_test, y_pred, labels=range(len(CLASES)),
                                 target_names=CLASES, zero_division=0, output_dict=True)


def guardar_artefactos(modelo, escalador) -> None:
    """Guarda el modelo (.keras) y un JSON con todo lo necesario para inferir."""
    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    modelo.save(RUTA_MODELO)
    config = {
        "features": FEATURES,
        "clases": CLASES,
        "scaler_mean": escalador.mean_.tolist(),
        "scaler_scale": escalador.scale_.tolist(),
        "clip_ratio": CLIP_RATIO,
        "clip_lluvia_mm": CLIP_LLUVIA_MM,
        "umbrales_reglas": UMBRALES_REGLAS,
    }
    with open(RUTA_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Modelo guardado en: {RUTA_MODELO}")
    print(f"Configuración guardada en: {RUTA_CONFIG}")


# --------------------------------------------------------------------------- #
# 5. Programa principal
# --------------------------------------------------------------------------- #
def main(epochs: int = EPOCHS) -> None:
    tf.keras.utils.set_random_seed(SEED)

    # Datos y variables
    df = preparar_variables(cargar_datos())
    print(f"Filas: {len(df)} | Semanas: {df['fecha_inicio_semana'].nunique()}")

    # Split agrupado por semana: 20 % test; del resto, 15 % validación
    df_trainval, df_test = dividir_por_semana(df, test_size=0.20)
    df_train, df_val = dividir_por_semana(df_trainval, test_size=0.15)
    assert not set(df_train["fecha_inicio_semana"]) & set(df_test["fecha_inicio_semana"]), \
        "Hay semanas repetidas entre train y test"
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # Escalado y pesos de clase
    escalador, X_train, X_val, X_test = escalar(df_train, df_val, df_test)
    pesos = calcular_pesos(df_train["y"].values)
    print("Pesos por clase:", {CLASES[k]: round(v, 2) for k, v in pesos.items()})

    # Entrenamiento
    modelo, _ = entrenar(X_train, df_train["y"].values, X_val, df_val["y"].values,
                         pesos, epochs=epochs, verbose=0)

    # Evaluación en test
    evaluar(modelo, X_test, df_test["y"].values)

    # Guardado de datos procesados y artefactos
    CARPETA_PROCESADOS.mkdir(parents=True, exist_ok=True)
    for nombre, X, d in [("train", X_train, df_train), ("val", X_val, df_val), ("test", X_test, df_test)]:
        X.to_csv(CARPETA_PROCESADOS / f"X_{nombre}_ann.csv", index=False)
        d[["y"]].to_csv(CARPETA_PROCESADOS / f"y_{nombre}_ann.csv", index=False)
    guardar_artefactos(modelo, escalador)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena la ANN del Módulo 3 (recomendación de acción).")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Número de épocas (por defecto 200)")
    args = parser.parse_args()
    main(epochs=args.epochs)
