"""Parte C — observabilidad: logs JSON estructurados y métricas operativas en memoria.

Los tres pilares:
- Logs (qué pasó): un JSON por línea, en consola y en data/runtime/api.log, filtrables por thread_id, evento
  o nivel. Nunca se loguea el texto crudo del usuario: solo su largo y una versión enmascarada.
- Métricas (cuánto pasó): contador global thread-safe, expuesto en GET /metricas.
- Trazas (cómo pasó): LangSmith, activado por las variables LANGSMITH_* del .env (no requiere código acá).
"""
import json
import logging
import math
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from src.config import RUTA_LOGS

# ── Logs JSON ────────────────────────────────────────────────────────────────────────────────────────────────────

_CAMPOS_ESTANDAR = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class FormateadorJSON(logging.Formatter):
    """Cada registro sale como un JSON de una línea: timestamp, nivel, evento y los campos extra del contexto."""

    def format(self, registro: logging.LogRecord) -> str:
        datos = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": registro.levelname,
            "event": registro.getMessage(),
            "logger": registro.name,
        }
        datos.update({k: v for k, v in vars(registro).items() if k not in _CAMPOS_ESTANDAR})
        if registro.exc_info:
            datos["exception"] = self.formatException(registro.exc_info)
        return json.dumps(datos, ensure_ascii=False, default=str)


def configurar_logger(nombre: str = "smart_flight_api") -> logging.Logger:
    logger = logging.getLogger(nombre)
    if logger.handlers:  # ya configurado (uvicorn --reload o imports repetidos)
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formateador = FormateadorJSON()
    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(formateador)
    archivo = RotatingFileHandler(RUTA_LOGS, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    archivo.setFormatter(formateador)
    logger.addHandler(consola)
    logger.addHandler(archivo)
    return logger


logger = configurar_logger()


# ── Métricas ─────────────────────────────────────────────────────────────────────────────────────────────────────

# umbrales de alerta (Clase 12): si se superan, /metricas los lista en "alertas_activas" y se loguea un WARNING
UMBRAL_LATENCIA_P95_MS = 15_000
UMBRAL_TASA_ERROR_PCT = 5.0


def percentil(valores: list[float], p: float) -> float:
    """Percentil por rango más cercano: el valor por debajo del cual queda el p% de las mediciones."""
    if not valores:
        return 0
    ordenados = sorted(valores)
    posicion = max(1, math.ceil(p / 100 * len(ordenados)))
    return ordenados[posicion - 1]


class Metricas:
    """Contador global de la API. Una sola instancia para todo el proceso (si se creara por request, /metricas
    devolvería siempre cero) y con lock, porque FastAPI atiende requests concurrentes en varios hilos."""

    def __init__(self):
        self._lock = threading.Lock()
        self.inicio = datetime.now(timezone.utc)
        self.total_consultas = 0
        self.exitosas = 0
        self.fallidas = 0
        self.bloqueadas = 0
        self.latencias_ms: list[float] = []
        self.herramientas = Counter()
        self.caminos = Counter()
        self.motivos_bloqueo = Counter()
        self.tipos_error = Counter()
        self.pii_enmascarada = Counter()
        self.tokens_entrada = 0
        self.tokens_salida = 0
        self.costo_usd = 0.0
        self.limites_alcanzados = 0

    def registrar_consulta(self, duracion_ms: float, resultado: dict):
        with self._lock:
            self.total_consultas += 1
            self.exitosas += 1
            self.latencias_ms.append(duracion_ms)
            self.herramientas.update(resultado.get("herramientas_usadas", []))
            self.caminos[resultado.get("camino") or "desconocido"] += 1
            self.pii_enmascarada.update(resultado.get("pii_enmascarada", []))
            self.tokens_entrada += resultado.get("tokens_entrada", 0)
            self.tokens_salida += resultado.get("tokens_salida", 0)
            self.costo_usd += resultado.get("costo_usd", 0.0)
            self.limites_alcanzados += int(bool(resultado.get("limite_alcanzado")))

    def registrar_bloqueo(self, motivo: str):
        with self._lock:
            self.total_consultas += 1
            self.bloqueadas += 1
            self.motivos_bloqueo[motivo] += 1

    def registrar_error(self, duracion_ms: float, tipo_error: str):
        with self._lock:
            self.total_consultas += 1
            self.fallidas += 1
            self.latencias_ms.append(duracion_ms)
            self.tipos_error[tipo_error] += 1

    def resumen(self) -> dict:
        with self._lock:
            procesadas = self.exitosas + self.fallidas
            datos = {
                "total_consultas": self.total_consultas,
                "consultas_exitosas": self.exitosas,
                "consultas_fallidas": self.fallidas,
                "consultas_bloqueadas": self.bloqueadas,
                "tasa_error_pct": round(100 * self.fallidas / procesadas, 2) if procesadas else 0.0,
                # latencias de las consultas que llegaron al agente (un bloqueo tarda ~0 ms y escondería la cola)
                "latencia_promedio_ms": round(sum(self.latencias_ms) / len(self.latencias_ms)) if self.latencias_ms else 0,
                "latencia_p95_ms": round(percentil(self.latencias_ms, 95)),
                "latencia_p99_ms": round(percentil(self.latencias_ms, 99)),
                "latencia_max_ms": round(max(self.latencias_ms)) if self.latencias_ms else 0,
                "herramientas_mas_usadas": dict(self.herramientas.most_common()),
                "caminos_del_grafo": dict(self.caminos.most_common()),
                "bloqueos": {"total": self.bloqueadas, "por_motivo": dict(self.motivos_bloqueo)},
                "errores_por_tipo": dict(self.tipos_error),
                "pii_enmascarada": dict(self.pii_enmascarada),
                "limite_iteraciones_alcanzado": self.limites_alcanzados,
                "tokens": {"entrada": self.tokens_entrada, "salida": self.tokens_salida,
                           "total": self.tokens_entrada + self.tokens_salida},
                "costo_estimado_usd": round(self.costo_usd, 6),
                "costo_promedio_por_consulta_usd": round(self.costo_usd / self.exitosas, 6) if self.exitosas else 0.0,
                "desde": self.inicio.isoformat(timespec="seconds"),
                "uptime_s": int((datetime.now(timezone.utc) - self.inicio).total_seconds()),
            }
        alertas = []
        if datos["latencia_p95_ms"] > UMBRAL_LATENCIA_P95_MS:
            alertas.append(f"latencia p95 {datos['latencia_p95_ms']} ms > umbral {UMBRAL_LATENCIA_P95_MS} ms")
        if datos["tasa_error_pct"] > UMBRAL_TASA_ERROR_PCT:
            alertas.append(f"tasa de error {datos['tasa_error_pct']}% > umbral {UMBRAL_TASA_ERROR_PCT}%")
        datos["alertas_activas"] = alertas
        return datos


metricas = Metricas()
