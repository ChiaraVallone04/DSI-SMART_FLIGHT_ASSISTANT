"""Parte B — API REST del agente con FastAPI.

Endpoints:
    GET  /health              estado del servicio y de sus dependencias (catálogo, memoria, claves)
    POST /consultar           guardrails de entrada -> agente (LangGraph) -> guardrail de salida -> métricas
    GET  /estado/{thread_id}  historial de la conversación, leído de la memoria SQLite (enmascarado)
    GET  /metricas            consultas, errores, latencia promedio/p95, herramientas, bloqueos, tokens y costo
    GET  /docs                Swagger UI (lo genera FastAPI)

Levantar (desde la raíz del repo):
    python main.py serve                  (o: uvicorn src.api:app --port 8000)
"""
import asyncio
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from src.config import API_HOST, API_PORT, RUTA_MEMORIA, TIMEOUT_CONSULTA_SEGUNDOS
from src.guardrails import enmascarar_pii, validar_entrada
from src.observability import logger, metricas
from src.schemas import ConsultaRequest, ConsultaResponse, EstadoResponse

logger.info("api_iniciando", extra={"memoria": str(RUTA_MEMORIA)})
# importar el agente abre ChromaDB y la memoria SQLite: si la base no está construida, el servidor no arranca y
# el error dice qué comando correr (python main.py ingest)
from src.graph import conexion_memoria, obtener_historial, responder  # noqa: E402
from vector_db import obtener_coleccion  # noqa: E402

VERSION = "1.0.0"

app = FastAPI(
    title="Smart Flight Assistant — API del agente",
    description="Agente de vuelos con LangGraph (RAG + sensor de disponibilidad + actuador de reservas), "
                "memoria persistente por thread_id, guardrails y observabilidad.",
    version=VERSION,
)


# ── Middleware y manejo de errores ───────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def registrar_request(request: Request, call_next):
    """Un log por request HTTP (método, ruta, status, duración) con un request_id para correlacionar."""
    request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    inicio = time.perf_counter()
    respuesta = await call_next(request)
    logger.info("request_completado", extra={
        "request_id": request.state.request_id, "method": request.method, "path": request.url.path,
        "status_code": respuesta.status_code, "duracion_ms": round((time.perf_counter() - inicio) * 1000),
    })
    respuesta.headers["X-Request-ID"] = request.state.request_id
    return respuesta


@app.exception_handler(Exception)
async def error_no_manejado(request: Request, error: Exception):
    """Cualquier excepción inesperada: log con traceback y un JSON limpio, sin detalles internos al cliente."""
    logger.error("error_no_manejado", exc_info=error, extra={
        "request_id": getattr(request.state, "request_id", None), "path": request.url.path,
        "error_type": type(error).__name__})
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor."})


# ── Endpoints ────────────────────────────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def raiz():
    return RedirectResponse("/docs")


@app.get("/health")
async def health():
    """Verifica que el servicio y sus dependencias estén operativos (lo usan los monitores y Docker)."""
    chequeos = {}
    try:
        chequeos["catalogo_documentos"] = obtener_coleccion().count()
    except Exception as error:  # noqa: BLE001
        chequeos["catalogo_documentos"] = f"error: {type(error).__name__}"
    try:
        conexion_memoria.execute("SELECT 1").fetchone()
        chequeos["memoria_sqlite"] = "ok"
    except Exception as error:  # noqa: BLE001
        chequeos["memoria_sqlite"] = f"error: {type(error).__name__}"
    chequeos["openai_api_key_configurada"] = bool(os.getenv("OPENAI_API_KEY"))
    chequeos["langsmith_tracing"] = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    sano = (isinstance(chequeos["catalogo_documentos"], int) and chequeos["catalogo_documentos"] > 0
            and chequeos["memoria_sqlite"] == "ok" and chequeos["openai_api_key_configurada"])
    cuerpo = {"status": "ok" if sano else "degradado", "version": VERSION, "sistema_activo": sano,
                "chequeos": chequeos}
    return JSONResponse(status_code=200 if sano else 503, content=cuerpo)


@app.post("/consultar", response_model=ConsultaResponse)
async def consultar(consulta: ConsultaRequest, request: Request):
    request_id = request.state.request_id
    inicio = time.perf_counter()
    contexto_log = {"request_id": request_id, "thread_id": consulta.thread_id, "user_id": consulta.user_id,
                    "pregunta_chars": len(consulta.pregunta)}

    # CAPA 1 — guardrails de entrada: lo bloqueado nunca llega al agente (ni a la memoria, ni cuesta tokens).
    # Se responde 200 con exito=false y un mensaje amable, sin revelar qué regla lo frenó
    validacion = validar_entrada(consulta.pregunta)
    if not validacion.valido:
        metricas.registrar_bloqueo(validacion.motivo)
        logger.warning("consulta_bloqueada", extra={**contexto_log, "motivo": validacion.motivo})
        return ConsultaResponse(
            respuesta=validacion.mensaje, thread_id=consulta.thread_id, herramientas_usadas=[], exito=False,
            bloqueado=True, motivo_bloqueo=validacion.motivo,
            latencia_ms=round((time.perf_counter() - inicio) * 1000), request_id=request_id)

    logger.info("consulta_iniciada", extra={
        **contexto_log, "pregunta_enmascarada": enmascarar_pii(consulta.pregunta)[0][:300]})

    # CAPA 2 — el agente (síncrono) corre en un hilo aparte para no bloquear el servidor, con timeout
    try:
        resultado = await asyncio.wait_for(
            asyncio.to_thread(responder, consulta.pregunta, consulta.thread_id, consulta.user_id),
            timeout=TIMEOUT_CONSULTA_SEGUNDOS)
    except asyncio.TimeoutError:
        duracion = (time.perf_counter() - inicio) * 1000
        metricas.registrar_error(duracion, "Timeout")
        logger.error("consulta_timeout", extra={**contexto_log, "duracion_ms": round(duracion),
                                                "timeout_s": TIMEOUT_CONSULTA_SEGUNDOS})
        raise HTTPException(status_code=504, detail=(
            f"El agente no respondió en {TIMEOUT_CONSULTA_SEGUNDOS:.0f} segundos. Intentá de nuevo en un momento."))
    except Exception as error:
        duracion = (time.perf_counter() - inicio) * 1000
        metricas.registrar_error(duracion, type(error).__name__)
        logger.error("consulta_fallida", exc_info=error, extra={
            **contexto_log, "duracion_ms": round(duracion), "error_type": type(error).__name__})
        raise HTTPException(status_code=500, detail="Error interno al procesar la consulta.") from None

    # CAPA 3 — el guardrail de salida ya se aplicó dentro del grafo (nodo sintetizar); acá solo se mide
    duracion = (time.perf_counter() - inicio) * 1000
    metricas.registrar_consulta(duracion, resultado)
    logger.info("consulta_completada", extra={
        **contexto_log, "duracion_ms": round(duracion), "camino": resultado["camino"],
        "intencion": resultado["intencion"], "herramientas_usadas": resultado["herramientas_usadas"],
        "iteraciones_agente": resultado["iteraciones"], "limite_alcanzado": resultado["limite_alcanzado"],
        "pii_enmascarada": resultado["pii_enmascarada"], "tokens_entrada": resultado["tokens_entrada"],
        "tokens_salida": resultado["tokens_salida"], "costo_usd": round(resultado["costo_usd"], 6),
        "respuesta_chars": len(resultado["respuesta"]),
    })
    return ConsultaResponse(
        respuesta=resultado["respuesta"], thread_id=consulta.thread_id,
        herramientas_usadas=resultado["herramientas_usadas"], intencion=resultado["intencion"],
        camino=resultado["camino"], latencia_ms=round(duracion), request_id=request_id)


@app.get("/estado/{thread_id}", response_model=EstadoResponse)
async def estado(thread_id: str):
    """Historial de la conversación guardado en la memoria SQLite (sobrevive a reinicios del servidor)."""
    historial = await asyncio.to_thread(obtener_historial, thread_id)
    if historial is None:
        raise HTTPException(status_code=404, detail=f"No hay una conversación con thread_id '{thread_id}'.")
    return historial


@app.get("/metricas")
async def obtener_metricas():
    """Métricas operativas desde que arrancó el proceso (se reinician al reiniciar el servidor)."""
    resumen = metricas.resumen()
    if resumen["alertas_activas"]:
        logger.warning("alertas_activas", extra={"alertas": resumen["alertas_activas"]})
    return resumen


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
