"""
recomendador.py · Módulo 3: función de recomendación para la app
================================================================

Envuelve la ANN entrenada (``models/modelo_ann_acciones.keras``) en una función
simple, ``recomendar()``, que recibe los datos de una semana y devuelve la acción
preventiva sugerida, su confianza y el motivo en lenguaje natural.

Uso desde la app Streamlit (app/Home.py):

    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))   # raíz del repo
    from src.recomendador import recomendar

    resultado = recomendar(
        casos_pred_4sem=339,         # salida del LSTM (Módulo 2)
        casos_umbral_endemico=282,   # canal endémico de esa semana
        lluvia_14d_mm=17.4,          # clima (NASA POWER)
        temp_max=27.1,               # clima (NASA POWER)
        aegypti_detectado=1,         # salida de la CNN (Módulo 1): 1 si es Aedes aegypti
    )
    st.write(resultado["accion_legible"], resultado["motivo"])

IMPORTANTE sobre las unidades de entrada:
    ``casos_pred_4sem`` y ``casos_umbral_endemico`` deben estar en la MISMA escala
    (en el dataset de entrenamiento, ambos son totales nacionales de 4 semanas).
    El modelo solo mira el cociente entre ambos.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

# Rutas relativas a la raíz del repositorio (este archivo está en src/)
RAIZ = Path(__file__).resolve().parents[1]
RUTA_MODELO = RAIZ / "models" / "modelo_ann_acciones.keras"
RUTA_CONFIG = RAIZ / "models" / "ann_config.json"

# Nombres amigables para mostrar en la interfaz
ACCIONES_LEGIBLES = {
    "monitoreo_normal": "Monitoreo normal",
    "eliminacion_criaderos": "Eliminación de criaderos",
    "fumigacion_focalizada": "Fumigación focalizada",
    "alerta_sanitaria": "Alerta sanitaria",
}


@lru_cache(maxsize=1)
def _cargar_artefactos():
    """Carga el modelo y la configuración UNA sola vez (se reutiliza en cada llamada).

    El import de TensorFlow es perezoso: así importar este módulo es instantáneo y
    el costo se paga solo con la primera recomendación.
    """
    import tensorflow as tf

    modelo = tf.keras.models.load_model(RUTA_MODELO)
    with open(RUTA_CONFIG, encoding="utf-8") as f:
        config = json.load(f)
    return modelo, config


def _construir_entrada(config, casos_pred_4sem, casos_umbral_endemico,
                       lluvia_14d_mm, temp_max, aegypti_detectado):
    """Replica EXACTAMENTE el preprocesado del entrenamiento y devuelve (1, n_features)."""
    ratio = casos_pred_4sem / casos_umbral_endemico
    valores = {
        "ratio_casos_umbral": min(ratio, config["clip_ratio"]),         # recorte de outliers
        "lluvia_14d_clip": min(lluvia_14d_mm, config["clip_lluvia_mm"]),
        "temp_max": temp_max,
        "aegypti_detectado": aegypti_detectado,
    }
    # Se respeta el orden de variables guardado al entrenar
    x = np.array([[valores[nombre] for nombre in config["features"]]], dtype="float32")
    # Estandarización con la media y desviación del conjunto de entrenamiento
    media = np.array(config["scaler_mean"], dtype="float32")
    escala = np.array(config["scaler_scale"], dtype="float32")
    return (x - media) / escala, ratio


def _cerca_de_umbral(ratio, lluvia_14d_mm, temp_max, reglas):
    """True si algún valor está muy cerca de un corte de decisión (zona incierta)."""
    return bool(
        abs(ratio - 1.0) <= 0.10
        or abs(ratio - reglas["factor_alerta"]) <= 0.20
        or abs(lluvia_14d_mm - reglas["lluvia_14d_mm"]) <= 10
        or abs(temp_max - reglas["temp_max_c"]) <= 0.3
    )


def _redactar_motivo(accion, casos, umbral, ratio, lluvia, temp_max, aegypti, reglas):
    """Explica en una frase por qué se sugiere la acción, citando los valores de entrada."""
    if accion == "alerta_sanitaria":
        return (f"Los casos pronosticados ({casos:,.0f}) equivalen a {ratio:.1f} veces el umbral "
                f"endémico ({umbral:,.0f}); superan el doble ({reglas['factor_alerta']:.0f}×).")
    if accion == "fumigacion_focalizada":
        return (f"Los casos pronosticados ({casos:,.0f}) superan el umbral endémico ({umbral:,.0f}) "
                f"y se detectó Aedes aegypti en la zona.")
    if accion == "eliminacion_criaderos":
        return (f"La lluvia de los últimos 14 días ({lluvia:.1f} mm) y la temperatura máxima "
                f"({temp_max:.1f} °C) favorecen la reproducción del mosquito "
                f"(referencia: > {reglas['lluvia_14d_mm']:.0f} mm y > {reglas['temp_max_c']:.0f} °C).")
    # monitoreo_normal
    return (f"Los casos pronosticados ({casos:,.0f}) no superan el umbral endémico ({umbral:,.0f}) "
            f"ni se combinan lluvia y calor altos; no hay indicadores que exijan una intervención.")


def recomendar(casos_pred_4sem, casos_umbral_endemico, lluvia_14d_mm, temp_max,
               aegypti_detectado) -> dict:
    """Recomienda una acción preventiva para una semana.

    Parámetros
    ----------
    casos_pred_4sem : casos pronosticados para las próximas 4 semanas (salida del LSTM).
    casos_umbral_endemico : umbral del canal endémico (misma escala que el anterior).
    lluvia_14d_mm : lluvia acumulada de los últimos 14 días, en mm.
    temp_max : temperatura máxima promedio, en °C.
    aegypti_detectado : 1 si la CNN detectó Aedes aegypti, 0 en caso contrario.

    Devuelve un diccionario con:
        accion          -> clave interna (p. ej. 'fumigacion_focalizada')
        accion_legible  -> texto para la interfaz (p. ej. 'Fumigación focalizada')
        confianza       -> probabilidad de la acción elegida (0 a 1)
        probabilidades  -> {acción: probabilidad} para las 4 acciones
        motivo          -> explicación en una frase
        cerca_de_umbral -> True si los datos están en una zona límite (conviene revisar)
    """
    if casos_umbral_endemico <= 0:
        raise ValueError("casos_umbral_endemico debe ser mayor que 0.")
    if casos_pred_4sem < 0:
        raise ValueError("casos_pred_4sem no puede ser negativo.")
    aegypti_detectado = int(bool(aegypti_detectado))   # acepta 0/1/True/False

    modelo, config = _cargar_artefactos()
    x, ratio = _construir_entrada(config, casos_pred_4sem, casos_umbral_endemico,
                                  lluvia_14d_mm, temp_max, aegypti_detectado)

    # Llamada directa al modelo (más rápida que .predict para una sola fila)
    probabilidades = modelo(x, training=False).numpy()[0]
    indice = int(np.argmax(probabilidades))
    accion = config["clases"][indice]
    reglas = config["umbrales_reglas"]

    motivo = _redactar_motivo(accion, casos_pred_4sem, casos_umbral_endemico, ratio,
                              lluvia_14d_mm, temp_max, aegypti_detectado, reglas)
    cerca = _cerca_de_umbral(ratio, lluvia_14d_mm, temp_max, reglas)
    if cerca:
        motivo += " Atención: algún valor está muy cerca de un umbral, conviene revisar el caso."

    return {
        "accion": accion,
        "accion_legible": ACCIONES_LEGIBLES[accion],
        "confianza": float(probabilidades[indice]),
        "probabilidades": {c: float(p) for c, p in zip(config["clases"], probabilidades)},
        "motivo": motivo,
        "cerca_de_umbral": cerca,
    }


if __name__ == "__main__":
    # Prueba rápida: python src/recomendador.py
    ejemplos = [
        ("Caso de la diapositiva", dict(casos_pred_4sem=339, casos_umbral_endemico=282,
                                        lluvia_14d_mm=17.4, temp_max=27.1, aegypti_detectado=1)),
        ("Epidemia (3x el umbral)", dict(casos_pred_4sem=3000, casos_umbral_endemico=1000,
                                          lluvia_14d_mm=120, temp_max=28.5, aegypti_detectado=0)),
        ("Lluvia y calor", dict(casos_pred_4sem=400, casos_umbral_endemico=1000,
                                 lluvia_14d_mm=150, temp_max=29.0, aegypti_detectado=0)),
        ("Todo normal", dict(casos_pred_4sem=300, casos_umbral_endemico=1000,
                              lluvia_14d_mm=30, temp_max=26.0, aegypti_detectado=0)),
    ]
    for titulo, datos in ejemplos:
        r = recomendar(**datos)
        print(f"\n{titulo}: {r['accion_legible']} (confianza {r['confianza']:.0%})")
        print(f"  {r['motivo']}")
