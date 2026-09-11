"""Lógica de extracción: llama a Claude (Anthropic), valida la respuesta y resuelve el año de las fechas."""
from config_claude import client, MODELO_LLM
from extraccion_comun import construir_solicitud, resolver_anio_de_fechas
from prompts import SYSTEM_INSTRUCTION_COT, SYSTEM_INSTRUCTION_ZERO_SHOT
from schemas import OutputResponse
from typing import Literal

NOMBRE_TOOL = "extraer_datos"

# Arma la solicitud, llama a Claude forzando salida estructurada vía tool-use, valida y resuelve el año de las fechas


def extraer_intencion(texto: str, canal: str, tecnica: Literal["cot", "zero-shot"] = "cot"):
    solicitud = construir_solicitud(texto, canal, tecnica)
    system_instruction = SYSTEM_INSTRUCTION_COT if tecnica == "cot" else SYSTEM_INSTRUCTION_ZERO_SHOT
    response = client.messages.create(
        model=MODELO_LLM,
        max_tokens=1024,
        system=system_instruction,
        messages=[{"role": "user", "content": solicitud.texto_libre}],
        tools=[{
            "name": NOMBRE_TOOL,
            "description": "Registra la intención y los parámetros de viaje extraídos del mensaje del usuario.",
            "input_schema": OutputResponse.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": NOMBRE_TOOL},
    )
    tool_use = next(
        bloque for bloque in response.content if bloque.type == "tool_use")
    respuesta = OutputResponse.model_validate(tool_use.input)
    return resolver_anio_de_fechas(respuesta, solicitud.timestamp.date())
