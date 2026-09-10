"""Lógica de extracción: llama a Gemini, valida la respuesta y resuelve el año de las fechas."""
from datetime import date, datetime
from config import client, MODELO_LLM, SYSTEM_INSTRUCTION_COT, SYSTEM_INSTRUCTION_ZERO_SHOT
from google.genai import types
from schemas import OutputResponse, SolicitudEntrada, MESES_ES
from typing import Literal

# Arma la solicitud, llama a Gemini con salida estructurada, valida y resuelve el año de las fechas


def extraer_intencion(texto: str, canal: str, tecnica: Literal["cot", "zero-shot"] = "cot"):
    if tecnica not in ("cot", "zero-shot"):
        raise ValueError(
            f"tecnica debe ser 'cot' o 'zero-shot', se recibió: {tecnica!r}")
    system_instruction = SYSTEM_INSTRUCTION_COT if tecnica == "cot" else SYSTEM_INSTRUCTION_ZERO_SHOT
    solicitud = SolicitudEntrada(
        canal=canal,
        texto_libre=texto,
        timestamp=datetime.now().astimezone(),
    )
    response = client.models.generate_content(
        model=MODELO_LLM,
        contents=solicitud.texto_libre,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=OutputResponse,
            temperature=0.0,
        ),
    )
    respuesta = OutputResponse.model_validate_json(response.text)
    return resolver_anio_de_fechas(respuesta, solicitud.timestamp.date())

# Calcula el año real de un mes: si ya pasó este año, asume el año que viene


def _resolver_anio(mes: int, hoy: date):
    return hoy.year if mes >= hoy.month else hoy.year + 1

# Parsea un campo de fecha (mes en texto o formato MM/MM-DD) y le agrega el año real


def _resolver_fecha(valor: str, hoy: date):
    if valor is not None:
        if valor in MESES_ES:
            mes = MESES_ES[valor]
            dia = None
        else:

            mes_str, _, dia = valor.partition("-")
            mes = int(mes_str)
            dia = dia or None
        anio = _resolver_anio(mes, hoy)
        valor = f"{anio}-{mes:02d}-{int(dia):02d}" if dia else f"{anio}-{mes:02d}"
    return valor

# Devuelve una copia de la respuesta con fecha_desde/fecha_hasta ya resueltas (no muta el original)


def resolver_anio_de_fechas(respuesta: OutputResponse, hoy: date):
    fecha_desde = _resolver_fecha(respuesta.fecha_desde, hoy)
    fecha_hasta = _resolver_fecha(respuesta.fecha_hasta, hoy)
    return respuesta.model_copy(update={"fecha_desde": fecha_desde, "fecha_hasta": fecha_hasta})
