"""
API principal de VectorIA.

Este modulo expone mediante FastAPI los tres componentes de inteligencia
artificial del proyecto:

1. CNN: clasifica la especie de un mosquito a partir de una imagen.
2. LSTM: pronostica los casos de dengue cuatro semanas adelante.
3. ANN: recomienda una accion preventiva a partir del pronostico, el clima
   y la deteccion de Aedes aegypti.

Ejecucion desde la raiz del repositorio:
    python -m uvicorn api.main:app --reload

Documentacion interactiva:
    http://127.0.0.1:8000/docs

Aviso etico:
    VectorIA es una herramienta academica de apoyo a la vigilancia
    epidemiologica. No diagnostica dengue en personas ni sustituye el
    criterio de las autoridades sanitarias.
"""

from functools import lru_cache
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

# Se reutiliza la logica de inferencia preparada en src/. La API no entrena
# modelos; solamente carga los artefactos .keras y solicita predicciones.
from src.clasificador import MosquitoClassifier
from src.pronosticador import estado_pronosticador, fechas_disponibles, pronosticar
from src.recomendador import recomendar


# ============================================================
# 1. Rutas y configuracion general
# ============================================================

# api/main.py esta dentro de api/. parents[1] corresponde a la raiz del
# repositorio, sin depender de la ubicacion particular de cada computadora.
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_MODELO_CNN = RAIZ_PROYECTO / "models" / "modelo_final_mosquitos.keras"
RUTA_MODELO_ANN = RAIZ_PROYECTO / "models" / "modelo_ann_acciones.keras"

# Extensiones admitidas por el endpoint de clasificacion. La validacion evita
# intentar abrir como imagen archivos incompatibles o potencialmente corruptos.
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# 2. Aplicacion FastAPI
# ============================================================

app = FastAPI(
    title="VectorIA API",
    description=(
        "API academica para clasificacion de mosquitos, pronostico de dengue "
        "y recomendacion de acciones preventivas. No diagnostica dengue "
        "en personas."
    ),
    version="1.0.0",
)


# ============================================================
# 3. Esquemas de entrada
# ============================================================

class SolicitudDengue(BaseModel):
    """Datos necesarios para solicitar un pronostico con la LSTM."""

    fecha: str | None = Field(
        default=None,
        description=(
            "Fecha base en formato YYYY-MM-DD. Si se omite, se utiliza "
            "la ultima fecha disponible en los datos."
        ),
    )


class SolicitudAccion(BaseModel):
    """Variables que utiliza la ANN para recomendar una accion preventiva."""

    casos_pred_4sem: float = Field(
        ge=0,
        description="Casos pronosticados para el horizonte de cuatro semanas.",
    )
    casos_umbral_endemico: float = Field(
        gt=0,
        description="Umbral endemico en la misma escala que el pronostico.",
    )
    lluvia_14d_mm: float = Field(
        ge=0,
        description="Lluvia acumulada durante los ultimos 14 dias, en mm.",
    )
    temp_max: float = Field(
        description="Temperatura maxima promedio, en grados Celsius.",
    )
    aegypti_detectado: bool = Field(
        description="True cuando la CNN detecto Aedes aegypti.",
    )


# ============================================================
# 4. Carga y manejo de imagenes
# ============================================================

@lru_cache(maxsize=1)
def obtener_clasificador() -> MosquitoClassifier:
    """
    Carga la CNN una sola vez y la reutiliza en solicitudes posteriores.

    Cargar un archivo .keras es costoso. lru_cache evita repetir esa operacion
    en cada peticion y reduce el tiempo de respuesta de la API.
    """

    if not RUTA_MODELO_CNN.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo CNN: {RUTA_MODELO_CNN}"
        )

    return MosquitoClassifier(str(RUTA_MODELO_CNN))


def guardar_imagen_temporal(archivo: UploadFile) -> Path:
    """
    Guarda la imagen recibida en un archivo temporal.

    MosquitoClassifier trabaja con una ruta de archivo. Por eso la imagen
    enviada por HTTP se guarda temporalmente y se elimina despues de predecir.
    """

    extension = Path(archivo.filename or "imagen.jpg").suffix.lower()

    if extension not in EXTENSIONES_IMAGEN:
        raise HTTPException(
            status_code=400,
            detail="Formato no admitido. Use JPG, JPEG, PNG, BMP o WEBP.",
        )

    contenido = archivo.file.read()

    if not contenido:
        raise HTTPException(
            status_code=400,
            detail="El archivo de imagen esta vacio.",
        )

    temporal = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension,
    )
    temporal.write(contenido)
    temporal.close()

    return Path(temporal.name)


# ============================================================
# 5. Endpoints informativos
# ============================================================

@app.get("/")
def inicio() -> dict:
    """Devuelve informacion basica y la ruta de la documentacion."""

    return {
        "proyecto": "VectorIA",
        "version": "1.0.0",
        "documentacion": "/docs",
        "aviso": (
            "Herramienta academica de vigilancia; no diagnostica dengue "
            "en personas."
        ),
    }


@app.get("/health")
def health() -> dict:
    """
    Comprueba la disponibilidad de la API y de los tres modelos.

    El endpoint permite al dashboard verificar el servicio antes de solicitar
    predicciones. La LSTM se valida cargando sus datos y revisando su forma.
    """

    try:
        estado_lstm = estado_pronosticador()
    except Exception as error:
        estado_lstm = {
            "disponible": False,
            "detalle": str(error),
        }

    return {
        "api": "disponible",
        "cnn_modelo_existe": RUTA_MODELO_CNN.exists(),
        "lstm": estado_lstm,
        "ann_modelo_existe": RUTA_MODELO_ANN.exists(),
    }


# ============================================================
# 6. Endpoints del modulo LSTM
# ============================================================

@app.get("/dengue/dates")
def obtener_fechas() -> dict:
    """Lista las fechas que tienen suficiente historial para pronosticar."""

    try:
        fechas = fechas_disponibles()
        return {
            "cantidad": len(fechas),
            "fechas": fechas,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


@app.post("/dengue/predict")
def predecir_dengue(solicitud: SolicitudDengue) -> dict:
    """
    Pronostica los casos de dengue cuatro semanas despues de la fecha base.

    Un error de fecha se considera una solicitud invalida (HTTP 400). Los
    errores inesperados al cargar el modelo o los datos se reportan como 500.
    """

    try:
        return pronosticar(solicitud.fecha)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


# ============================================================
# 7. Endpoint del modulo ANN
# ============================================================

@app.post("/action/recommend")
def recomendar_accion(solicitud: SolicitudAccion) -> dict:
    """Recomienda una accion preventiva mediante la ANN entrenada."""

    try:
        return recomendar(
            casos_pred_4sem=solicitud.casos_pred_4sem,
            casos_umbral_endemico=solicitud.casos_umbral_endemico,
            lluvia_14d_mm=solicitud.lluvia_14d_mm,
            temp_max=solicitud.temp_max,
            aegypti_detectado=solicitud.aegypti_detectado,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


# ============================================================
# 8. Endpoint del modulo CNN
# ============================================================

@app.post("/mosquito/classify")
def clasificar_mosquito(
    archivo: UploadFile = File(...),
) -> dict:
    """
    Clasifica una imagen y devuelve la especie y la confianza de la CNN.

    La imagen temporal se elimina siempre mediante finally, incluso si ocurre
    un error durante la inferencia.
    """

    ruta_temporal = guardar_imagen_temporal(archivo)

    try:
        resultado = obtener_clasificador().predict(
            str(ruta_temporal)
        )

        return {
            "archivo": archivo.filename,
            "clase": resultado["class"],
            "confianza": resultado["confidence"],
            "probabilidades": resultado["probabilities"]
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
    finally:
        ruta_temporal.unlink(missing_ok=True)


# ============================================================
# 9. Endpoint integrado CNN + LSTM + ANN
# ============================================================

@app.post("/integrated/analyze")
def analisis_integrado(
    archivo: UploadFile = File(...),
    fecha: str | None = Form(default=None),
    casos_umbral_endemico: float = Form(..., gt=0),
    lluvia_14d_mm: float = Form(..., ge=0),
    temp_max: float = Form(...),
) -> dict:
    """
    Ejecuta el flujo integral de VectorIA.

    Flujo:
    1. La CNN clasifica el mosquito de la imagen.
    2. La LSTM pronostica los casos cuatro semanas adelante.
    3. La ANN combina pronostico, umbral, clima y deteccion de Aedes aegypti.
    4. La API devuelve los tres resultados en una sola respuesta.
    """

    ruta_temporal = guardar_imagen_temporal(archivo)

    try:
        clasificacion = obtener_clasificador().predict(
            str(ruta_temporal)
        )

        pronostico = pronosticar(fecha)

        # La ANN utiliza una variable binaria: 1 si la CNN identifica aegypti,
        # y 0 para cualquier otra clase de mosquito.
        aegypti_detectado = (
            clasificacion["class"].lower() == "aegypti"
        )

        accion = recomendar(
            casos_pred_4sem=pronostico[
                "casos_pronosticados_decimal"
            ],
            casos_umbral_endemico=casos_umbral_endemico,
            lluvia_14d_mm=lluvia_14d_mm,
            temp_max=temp_max,
            aegypti_detectado=aegypti_detectado,
        )

        return {
            "clasificacion_mosquito": clasificacion,
            "pronostico_dengue": pronostico,
            "recomendacion": accion,
            "aviso": (
                "Resultado academico de apoyo; no es un diagnostico medico."
            ),
        }
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
    finally:
        ruta_temporal.unlink(missing_ok=True)
