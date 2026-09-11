"""Lógica de extracción: llama a Gemini, valida la respuesta y resuelve el año de las fechas."""
from config import client, MODELO_LLM
from extraccion_comun import construir_solicitud, resolver_anio_de_fechas
from google.genai import types
from prompts import SYSTEM_INSTRUCTION_COT, SYSTEM_INSTRUCTION_ZERO_SHOT
from schemas import OutputResponse
from typing import Literal

# Arma la solicitud, llama a Gemini con salida estructurada, valida y resuelve el año de las fechas


def extraer_intencion(texto: str, canal: str, tecnica: Literal["cot", "zero-shot"] = "cot"):
    solicitud = construir_solicitud(texto, canal, tecnica)
    system_instruction = SYSTEM_INSTRUCTION_COT if tecnica == "cot" else SYSTEM_INSTRUCTION_ZERO_SHOT
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
