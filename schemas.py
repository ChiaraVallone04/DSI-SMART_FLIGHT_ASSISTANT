"""Contratos de datos (Pydantic): entrada de la API y salida estructurada del LLM."""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal
from datetime import datetime
import re

# Mapea nombres de meses en español a su número (1-12)
MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


# Payload de entrada según el contrato de la API


class SolicitudEntrada(BaseModel):
    canal: str = Field(
        description="Canal de origen del mensaje, ej. 'whatsapp', 'cli'")
    texto_libre: str = Field(
        description="Texto crudo en lenguaje natural ingresado por el usuario")
    adjuntos: list[str] = Field(
        default_factory=list, description="Adjuntos del mensaje (vacío por ahora)")
    timestamp: datetime = Field(
        description="Momento real de la interacción del usuario")


# Define la respuesta del LLM en formato JSON


class OutputResponse(BaseModel):
    razonamiento: str = Field(
        description="Razonamiento paso a paso del modelo antes de fijar la salida")
    # La intencion esta limitada en cuatro valores posibles
    intencion: Literal["buscar_vuelos", "recomendar_compra", "comparar_opciones", "fuera_de_alcance"] = Field(
        description="Tipo de intención detectada")
    origen: str | None = Field(
        default=None, description="Código IATA de origen, 3 letras mayúsculas, ej. MAD")
    destino: str | None = Field(
        default=None, description="Código IATA de destino, 3 letras mayúsculas, ej. BER")
    fecha_desde: str | None = Field(
        default=None, description="Inicio de la ventana de viaje")
    fecha_hasta: str | None = Field(
        default=None, description="Fin de la ventana de viaje")
    escalas_max: int | None = Field(
        default=None, description="Escalas máximas toleradas (0-5)")
    presupuesto_max: float | None = Field(
        default=None, description="Presupuesto máximo que el usuario está dispuesto a pagar")

    # Valida que los códigos IATA de origen y destino sean correctos (3 letras alfabeticas mayúsculas)
    # El anotador @field_validator ejecuta automáticamente la función de validación para los campos indicados
    # El anotador @classmethod indica que la función es un método de clase y recibe la clase como primer argumento
    @field_validator('origen', 'destino')
    @classmethod
    def validar_formato_iata(cls, v: str):
        if v is not None:
            v = v.strip().upper()
            if len(v) != 3 or not v.isalpha():
                raise ValueError(
                    f"Código IATA inválido: '{v}'. Debe ser exactamente 3 letras (ej. MAD, BER).")
        return v

    # Valida que escalas_max esté en el rango permitido (0-5)
    @field_validator('escalas_max')
    @classmethod
    def validar_rango_escalas(cls, v: int):
        if v is not None and not (0 <= v <= 5):
            raise ValueError(
                f"escalas_max fuera de rango: {v}. Debe estar entre 0 y 5.")
        return v

    # Valida que presupuesto_max sea positivo si se proporciona
    @field_validator('presupuesto_max')
    @classmethod
    def validar_presupuesto_positivo(cls, v: float):
        if v is not None and v <= 0:
            raise ValueError(
                f"presupuesto_max debe ser positivo, se recibió: {v}")
        return v

    # Valida que fecha_desde y fecha_hasta tengan un formato válido (mes en español o MM/MM-DD)
    @field_validator('fecha_desde', 'fecha_hasta')
    @classmethod
    def validar_formato_fecha(cls, v: str):
        if v is not None:
            v = v.strip().lower()
            es_mes_en_texto = v in MESES_ES
            es_formato_numerico = bool(re.fullmatch(r"\d{1,2}(-\d{1,2})?", v))
            if not (es_mes_en_texto or es_formato_numerico):
                raise ValueError(
                    f"Fecha inválida: '{v}'. Debe ser un mes en español (ej. 'octubre') o formato MM/MM-DD (ej. '10' o '10-15').")
        return v

    # Regla de oro: Valida que los campos estén vacíos si la intención es "fuera_de_alcance"

    @model_validator(mode="after")
    def validar_campos_vacios_si_fuera_de_alcance(self):
        if self.intencion == "fuera_de_alcance":
            if any([
                    self.origen, self.destino, self.fecha_desde,
                    self.fecha_hasta, self.escalas_max, self.presupuesto_max,]):
                raise ValueError(
                    "Intención 'fuera_de_alcance' no puede traer parámetros cargados")
        return self
