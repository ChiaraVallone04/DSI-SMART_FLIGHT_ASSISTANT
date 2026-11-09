"""Lógica de extracción compartida entre proveedores: construcción de la solicitud y resolución de fechas."""
from datetime import date, datetime
from typing import Literal

from schemas import MESES_ES, OutputResponse, SolicitudEntrada

TECNICAS = ("cot", "zero-shot")

# Valida la técnica y arma la solicitud de entrada, común a cualquier proveedor


def construir_solicitud(texto: str, canal: str, tecnica: Literal["cot", "zero-shot"]):
    if tecnica not in TECNICAS:
        raise ValueError(
            f"tecnica debe ser 'cot' o 'zero-shot', se recibió: {tecnica!r}")
    return SolicitudEntrada(
        canal=canal,
        texto_libre=texto,
        timestamp=datetime.now().astimezone(),
    )

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
