"""Contratos de la API (Pydantic): qué recibe y qué devuelve cada endpoint.

FastAPI valida con estos modelos antes de tocar el agente (un body malformado responde 422 sin gastar tokens)
y los usa para generar la documentación de /docs.
"""
from pydantic import BaseModel, Field


class ConsultaRequest(BaseModel):
    # el tope de longitud "de negocio" lo aplica el guardrail (y cuenta como bloqueo en /metricas). Este
    # max_length es solo un corte de seguridad para no aceptar cuerpos gigantes (responde 422)
    pregunta: str = Field(..., max_length=20_000, description="Consulta en lenguaje natural",
                          examples=["¿Qué aerolíneas operan la ruta Madrid–Roma?"])
    thread_id: str = Field(..., min_length=1, max_length=100, pattern=r"^[\w.\-:@]+$",
                           description="Identificador de la conversación: mismo thread_id = misma memoria",
                           examples=["usuario_001"])
    user_id: str | None = Field(None, max_length=100, description="Identificador del usuario (opcional, para logs)")


class ConsultaResponse(BaseModel):
    respuesta: str
    thread_id: str
    herramientas_usadas: list[str]
    exito: bool = True
    bloqueado: bool = False
    motivo_bloqueo: str | None = None
    intencion: str | None = None
    camino: str | None = None
    latencia_ms: int
    request_id: str


class MensajeHistorial(BaseModel):
    rol: str
    contenido: str
    herramientas_llamadas: list[str] | None = None
    herramienta: str | None = None


class EstadoResponse(BaseModel):
    thread_id: str
    mensajes: list[MensajeHistorial]
    cantidad_mensajes: int
    ultima_intencion: str | None = None
    ultimo_camino: str | None = None
