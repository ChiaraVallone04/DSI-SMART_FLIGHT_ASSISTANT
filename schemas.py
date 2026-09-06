"""
Contrato de datos del Smart Flight Assistant.

Traduce a Pydantic V2 el System Prompt y el contrato de entrada de la API:
una sola consulta en lenguaje natural (texto_libre) se convierte en un objeto
ExtraccionVuelo que cubre cuatro intenciones posibles:

    buscar_vuelos | recomendar_compra | comparar_opciones | fuera_de_alcance

El LLM solo puede devolver estos cuatro valores exactos para `intencion` (Literal).
Cada intención usa un subconjunto distinto de los campos de parámetros; los que
no aplican quedan en None (tal como exige el System Prompt: "si un dato no
existe, asigná null").
"""

import re
from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

Intencion = Literal[
    "buscar_vuelos",
    "recomendar_compra",
    "comparar_opciones",
    "fuera_de_alcance",
]

_IATA_RE = re.compile(r"^[A-Z]{3}$")

# El System Prompt le pide al modelo que nunca invente un año: solo extrae
# mes (y día, si lo hay). El año es una cuenta determinista que hace el
# código con la fecha real de la consulta — el LLM no tiene una noción
# confiable de "qué año es hoy" ni de si un mes ya pasó, así que esa
# decisión no se le delega.
_FECHA_NUMERICA_RE = re.compile(r"^(?:\d{4}-)?(\d{1,2})(?:-(\d{1,2}))?$")

_MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def _resolver_anio(mes: int, hoy: date) -> int:
    """Año correcto para un mes sin año explícito: el año en curso si el mes
    todavía no pasó, o el próximo si ya pasó."""
    return hoy.year if mes >= hoy.month else hoy.year + 1


def _normalizar_fecha_sin_anio(texto: str, hoy: date) -> str:
    """Extrae mes (y día, si está) de texto en formato MM, MM-DD o nombre de mes
    en español, ignora cualquier año que el modelo haya podido inventar, y
    devuelve la fecha con el año recalculado en código respecto de `hoy`.

    `hoy` es la fecha real de la consulta (el `timestamp` de SolicitudEntrada),
    no la fecha del reloj del servidor en el momento de validar: así una
    interacción registrada el 2026-09-05 siempre resuelve igual, corra
    cuando corra.
    """
    v = texto.strip().lower()

    match = _FECHA_NUMERICA_RE.fullmatch(v)
    if match:
        mes = int(match.group(1))
        dia = match.group(2)
    else:
        mes = next((numero for nombre, numero in _MESES_ES.items() if nombre in v), None)
        dia_match = re.search(r"\b(\d{1,2})\b", v) if mes else None
        dia = dia_match.group(1) if dia_match else None

    if mes is None or not (1 <= mes <= 12):
        raise ValueError(
            f"No se pudo interpretar el mes en '{texto}'. Usá un mes válido (1-12 o nombre en español)."
        )

    anio = _resolver_anio(mes, hoy)
    if dia:
        return f"{anio}-{mes:02d}-{int(dia):02d}"
    return f"{anio}-{mes:02d}"

# Campos de parámetros que deben quedar en None cuando la intención
# detectada es "fuera_de_alcance": el LLM interpreta el lenguaje, pero nunca
# decide una regla de negocio por su cuenta.
_CAMPOS_DE_EXTRACCION = (
    "origen",
    "destino",
    "fecha_desde",
    "fecha_hasta",
    "fecha_viaje_aprox",
    "escalas_max",
    "presupuesto_max",
    "opcion_a",
    "opcion_b",
)


class SolicitudEntrada(BaseModel):
    """Contrato de entrada de la API: el payload crudo que llega al sistema
    por POST /api/v1/flights, antes de que el LLM lo procese.

    `texto_libre` es lo único que se le manda al LLM. `timestamp` no se le
    manda al modelo — se usa como ancla determinista para resolver el año de
    las fechas relativas en ExtraccionVuelo (ver `normalizar_anio_de_fecha`
    más abajo) y, en una implementación completa del sistema, para calcular
    cuántos días faltan hasta la fecha de viaje.
    """

    canal: str = Field(description="Canal de origen del mensaje, ej. 'whatsapp', 'cli'")
    texto_libre: str = Field(description="Texto crudo en lenguaje natural ingresado por el usuario")
    adjuntos: list[str] = Field(default_factory=list, description="Adjuntos del mensaje (vacío por ahora)")
    timestamp: datetime = Field(description="Momento real de la interacción del usuario")


class ExtraccionVuelo(BaseModel):
    """Salida estructurada del extractor de intenciones del Smart Flight Assistant."""

    # Campo de razonamiento pedido por el System Prompt: el modelo lo completa
    # ANTES de fijar la intención y los parámetros (técnica Chain-of-Thought).
    razonamiento: str = Field(
        description="Razonamiento paso a paso del modelo antes de fijar la salida."
    )

    intencion: Intencion

    # --- Parámetros de buscar_vuelos / recomendar_compra ---
    origen: Optional[str] = Field(default=None, description="Código IATA de origen, ej. MAD")
    destino: Optional[str] = Field(default=None, description="Código IATA de destino, ej. BER")
    fecha_desde: Optional[str] = Field(default=None, description="Inicio de la ventana de viaje")
    fecha_hasta: Optional[str] = Field(default=None, description="Fin de la ventana de viaje")
    fecha_viaje_aprox: Optional[str] = Field(
        default=None, description="Fecha aproximada de viaje (usada en recomendar_compra)"
    )
    escalas_max: Optional[int] = Field(default=None, description="Escalas máximas toleradas (0-5)")
    # Nota: el rango (gt=0) se aplica en un @field_validator, no como constraint
    # declarativa (Field(gt=0)), porque el SDK de Gemini traduce el schema de
    # Pydantic a su propio tipo `Schema` y ese tipo no soporta la keyword
    # "exclusiveMinimum" que genera `gt=0` — rompe response_schema en tiempo real.
    presupuesto_max: Optional[float] = Field(default=None, description="Presupuesto máximo en euros")

    # --- Parámetros de comparar_opciones ---
    opcion_a: Optional[str] = Field(default=None, description="Descripción/filtros del vuelo A")
    opcion_b: Optional[str] = Field(default=None, description="Descripción/filtros del vuelo B")

    # --- Motivo interno cuando la intención es fuera_de_alcance ---
    motivo_rechazo: Optional[str] = Field(
        default=None,
        description="Motivo interno de rechazo (prompt injection, fuera de dominio, etc.)",
    )

    @field_validator("origen", "destino")
    @classmethod
    def normalizar_y_validar_iata(cls, v: Optional[str]) -> Optional[str]:
        """Limpia y valida que el código sea un IATA de 3 letras (ej. 'madrid' -> error, 'mad' -> 'MAD')."""
        if v is None:
            return v
        v = v.strip().upper()
        if not _IATA_RE.fullmatch(v):
            raise ValueError(
                f"Código IATA inválido: '{v}'. Debe ser exactamente 3 letras (ej. MAD, BER, CDG)."
            )
        return v

    @field_validator("fecha_desde", "fecha_hasta", "fecha_viaje_aprox")
    @classmethod
    def normalizar_anio_de_fecha(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Recalcula el año de la fecha en código: usa el año actual, o el
        próximo si el mes mencionado ya pasó. Nunca confía en un año que el
        modelo haya podido inventar, porque un LLM no tiene una noción
        confiable de qué día es "hoy". Toma "hoy" del `timestamp` de
        SolicitudEntrada si se pasó por contexto; si no, usa la fecha real
        del sistema."""
        if v is None:
            return v
        hoy = (info.context or {}).get("hoy", date.today())
        return _normalizar_fecha_sin_anio(v, hoy)

    @field_validator("escalas_max")
    @classmethod
    def validar_rango_escalas(cls, v: Optional[int]) -> Optional[int]:
        """Rechaza un número de escalas fuera del rango real del dataset (0 a 5)."""
        if v is None:
            return v
        if not (0 <= v <= 5):
            raise ValueError(f"escalas_max fuera de rango: {v}. Debe estar entre 0 y 5.")
        return v

    @field_validator("presupuesto_max")
    @classmethod
    def validar_presupuesto_positivo(cls, v: Optional[float]) -> Optional[float]:
        """Rechaza un presupuesto máximo que no sea un número positivo."""
        if v is None:
            return v
        if v <= 0:
            raise ValueError(f"presupuesto_max debe ser positivo, se recibió: {v}")
        return v

    @model_validator(mode="after")
    def aplicar_regla_de_oro_fuera_de_alcance(self) -> "ExtraccionVuelo":
        """Si la intención es fuera_de_alcance, ningún parámetro de extracción puede venir cargado.

        Esto blinda en código la misma regla que ya le pedimos al LLM en el
        System Prompt ("si la intención es fuera_de_alcance, colocá todos los
        parámetros de extracción en null"): el LLM propone, el código dispone.
        """
        if self.intencion == "fuera_de_alcance":
            cargados = [c for c in _CAMPOS_DE_EXTRACCION if getattr(self, c) is not None]
            if cargados:
                raise ValueError(
                    "Intención 'fuera_de_alcance' no puede traer parámetros de extracción "
                    f"cargados: {cargados}"
                )
        return self
