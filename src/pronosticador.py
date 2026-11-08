"""Inferencia del modelo LSTM de dengue sin reentrenamiento."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ_PROYECTO / "data" / "raw" / "RNN" / "vectoria_dengue_semanal.csv"
RUTA_MODELO = RAIZ_PROYECTO / "models" / "modelo_lstm_dengue.keras"

VENTANA = 26
HORIZONTE = 4

VARIABLES_BASE = [
    "casos_dengue",
    "lluvia_mm",
    "temp_max_c",
    "temp_min_c",
    "humedad_rel_pct",
]

VARIABLES_REZAGADAS = (
    [f"casos_lag_{lag}" for lag in range(1, 5)]
    + [f"lluvia_lag_{lag}" for lag in range(1, 5)]
    + [f"temp_max_lag_{lag}" for lag in range(1, 5)]
    + [f"temp_min_lag_{lag}" for lag in range(1, 5)]
    + [f"humedad_lag_{lag}" for lag in range(1, 5)]
)
VARIABLES_MODELO = VARIABLES_BASE + VARIABLES_REZAGADAS


def validar_columnas(df: pd.DataFrame) -> None:
    requeridas = {"fecha_inicio_semana", *VARIABLES_BASE}
    faltantes = requeridas.difference(df.columns)
    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias en el CSV: "
            + ", ".join(sorted(faltantes))
        )


def crear_variables(df: pd.DataFrame) -> pd.DataFrame:
    validar_columnas(df)
    datos = df[["fecha_inicio_semana", *VARIABLES_BASE]].copy()
    datos["fecha_inicio_semana"] = pd.to_datetime(
        datos["fecha_inicio_semana"], errors="coerce"
    )
    datos = datos.dropna(subset=["fecha_inicio_semana"])
    datos = datos.sort_values("fecha_inicio_semana", kind="stable").reset_index(drop=True)

    for columna in VARIABLES_BASE:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    datos[VARIABLES_BASE] = (
        datos[VARIABLES_BASE].interpolate(method="linear").bfill().ffill()
    )

    for lag in range(1, 5):
        datos[f"casos_lag_{lag}"] = datos["casos_dengue"].shift(lag)
        datos[f"lluvia_lag_{lag}"] = datos["lluvia_mm"].shift(lag)
        datos[f"temp_max_lag_{lag}"] = datos["temp_max_c"].shift(lag)
        datos[f"temp_min_lag_{lag}"] = datos["temp_min_c"].shift(lag)
        datos[f"humedad_lag_{lag}"] = datos["humedad_rel_pct"].shift(lag)

    datos["target"] = datos["casos_dengue"].shift(-HORIZONTE)
    return datos.dropna(subset=VARIABLES_MODELO + ["target"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def cargar_artefactos() -> dict:
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No se encontro el CSV: {RUTA_DATOS}")
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(f"No se encontro el modelo LSTM: {RUTA_MODELO}")

    datos_sin_escalar = crear_variables(pd.read_csv(RUTA_DATOS))
    if len(datos_sin_escalar) < VENTANA:
        raise ValueError(
            f"Solo hay {len(datos_sin_escalar)} registros validos; se necesitan {VENTANA}."
        )

    columnas_escaladas = VARIABLES_MODELO + ["target"]
    scaler = MinMaxScaler()
    datos_escalados = datos_sin_escalar.copy()
    datos_escalados[columnas_escaladas] = scaler.fit_transform(
        datos_sin_escalar[columnas_escaladas]
    )

    modelo = tf.keras.models.load_model(RUTA_MODELO, compile=False)
    forma = modelo.input_shape[0] if isinstance(modelo.input_shape, list) else modelo.input_shape
    if len(forma) != 3:
        raise ValueError(f"Forma de entrada inesperada del modelo: {forma}")
    if forma[1] is not None and forma[1] != VENTANA:
        raise ValueError(f"El modelo espera {forma[1]} registros; el wrapper usa {VENTANA}.")
    if forma[2] is not None and forma[2] != len(VARIABLES_MODELO):
        raise ValueError(
            f"El modelo espera {forma[2]} variables; el wrapper construye {len(VARIABLES_MODELO)}."
        )

    indice_target = columnas_escaladas.index("target")
    return {
        "modelo": modelo,
        "datos_escalados": datos_escalados,
        "datos_sin_escalar": datos_sin_escalar,
        "target_minimo": float(scaler.data_min_[indice_target]),
        "target_maximo": float(scaler.data_max_[indice_target]),
    }


def fechas_disponibles() -> list[str]:
    datos = cargar_artefactos()["datos_escalados"]
    return (
        datos.loc[VENTANA - 1 :, "fecha_inicio_semana"]
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .tolist()
    )


def pronosticar(fecha: Optional[str] = None) -> dict:
    artefactos = cargar_artefactos()
    modelo = artefactos["modelo"]
    escalados = artefactos["datos_escalados"]
    originales = artefactos["datos_sin_escalar"]

    if fecha is None:
        indice_final = len(escalados) - 1
    else:
        fecha_solicitada = pd.to_datetime(fecha, errors="coerce")
        if pd.isna(fecha_solicitada):
            raise ValueError("La fecha debe usar el formato YYYY-MM-DD.")
        encontrados = escalados.index[
            escalados["fecha_inicio_semana"] == fecha_solicitada
        ].tolist()
        if not encontrados:
            raise ValueError(f"La fecha {fecha} no existe en los datos.")
        indice_final = encontrados[-1]

    indice_inicial = indice_final - VENTANA + 1
    if indice_inicial < 0:
        raise ValueError(f"Se requieren {VENTANA} registros anteriores.")

    ventana = escalados.loc[
        indice_inicial:indice_final, VARIABLES_MODELO
    ].to_numpy(dtype=np.float32)
    forma_esperada = (VENTANA, len(VARIABLES_MODELO))
    if ventana.shape != forma_esperada:
        raise ValueError(
            f"Forma de ventana incorrecta: {ventana.shape}; se esperaba {forma_esperada}."
        )

    salida = modelo.predict(np.expand_dims(ventana, axis=0), verbose=0)
    valor_escalado = float(np.asarray(salida).reshape(-1)[0])
    minimo = artefactos["target_minimo"]
    maximo = artefactos["target_maximo"]
    casos = max(0.0, valor_escalado * (maximo - minimo) + minimo)

    fecha_base = originales.loc[indice_final, "fecha_inicio_semana"]
    fecha_pronostico = fecha_base + pd.Timedelta(weeks=HORIZONTE)

    return {
        "fecha_base": fecha_base.strftime("%Y-%m-%d"),
        "fecha_pronostico": fecha_pronostico.strftime("%Y-%m-%d"),
        "casos_ultima_observacion": round(
            float(originales.loc[indice_final, "casos_dengue"]), 2
        ),
        "casos_pronosticados": int(round(casos)),
        "casos_pronosticados_decimal": round(casos, 2),
        "ventana_registros": VENTANA,
        "horizonte_semanas": HORIZONTE,
        "numero_variables": len(VARIABLES_MODELO),
        "modelo": "LSTM",
    }


def estado_pronosticador() -> dict:
    artefactos = cargar_artefactos()
    return {
        "disponible": True,
        "modelo": RUTA_MODELO.name,
        "datos": RUTA_DATOS.name,
        "registros_validos": len(artefactos["datos_escalados"]),
        "ventana_registros": VENTANA,
        "horizonte_semanas": HORIZONTE,
        "numero_variables": len(VARIABLES_MODELO),
        "forma_entrada_modelo": str(artefactos["modelo"].input_shape),
    }


if __name__ == "__main__":
    print("Probando pronosticador LSTM...")
    print("Estado:", estado_pronosticador())
    print("Pronostico:", pronosticar())
