"""Configuración del TP Final: lee config.toml (valores no secretos) y resuelve las rutas del proyecto.

Los secretos (OPENAI_API_KEY, LANGSMITH_*) van en .env y los carga vector_db.py (Entrega 2) al importarse.
Algunas rutas se pueden pisar con variables de entorno (útil para los tests o para Docker):
DATA_DIR (carpeta runtime), MEMORY_DB, RESERVAS_DB, LOG_FILE y TIMEOUT_CONSULTA_SEGUNDOS.
"""
import os
import sys
import tomllib
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parent.parent
RUTA_CONFIG = RAIZ_REPO / "config.toml"

with open(RUTA_CONFIG, "rb") as archivo:
    CONFIG = tomllib.load(archivo)

# el agente reutiliza el código de las entregas anteriores: vector_db.py (entrega_2/) y rag_pipeline.py
# (entrega_3/). Se agregan esas dos carpetas al path; la Entrega 1 se carga por ruta (ver graph.py)
for carpeta in (RAIZ_REPO / "entrega_2", RAIZ_REPO / "entrega_3"):
    if str(carpeta) not in sys.path:
        sys.path.insert(0, str(carpeta))

# esquema de la Entrega 1: de ahí salen las intenciones que usa el router
RUTA_SCHEMAS_E1 = RAIZ_REPO / "entrega_1" / "schemas.py"

# ── Rutas ────────────────────────────────────────────────────────────────────────────────────────────────────────
CARPETA_DATOS = RAIZ_REPO / CONFIG["rutas"]["datos"]
CARPETA_RESULTADOS = RAIZ_REPO / CONFIG["rutas"]["resultados"]
# lo que se genera al usar el sistema (no se versiona): memoria, reservas y logs
CARPETA_RUNTIME = Path(os.getenv("DATA_DIR", RAIZ_REPO / CONFIG["rutas"]["runtime"]))
CARPETA_RUNTIME.mkdir(parents=True, exist_ok=True)

# memoria conversacional del agente (checkpointer de LangGraph), una conversación por thread_id
RUTA_MEMORIA = Path(os.getenv("MEMORY_DB", CARPETA_RUNTIME / "memoria.db"))
# reservas creadas por el actuador
RUTA_RESERVAS = Path(os.getenv("RESERVAS_DB", CARPETA_RUNTIME / "reservas.db"))
# logs JSON (además de la consola)
RUTA_LOGS = Path(os.getenv("LOG_FILE", CARPETA_RUNTIME / "api.log"))

# ── LLM ──────────────────────────────────────────────────────────────────────────────────────────────────────────
MODELO_LLM = CONFIG["llm"]["modelo"]
TEMPERATURA_LLM = CONFIG["llm"]["temperatura"]
TIMEOUT_LLM_SEGUNDOS = CONFIG["llm"]["timeout_segundos"]
REINTENTOS_LLM = CONFIG["llm"]["reintentos"]

# precios en USD por millón de tokens, para estimar el costo de cada consulta
PRECIO_INPUT_POR_MILLON = CONFIG["precios"]["input_por_millon"]
PRECIO_INPUT_CACHE_POR_MILLON = CONFIG["precios"]["input_cache_por_millon"]
PRECIO_OUTPUT_POR_MILLON = CONFIG["precios"]["output_por_millon"]

# ── Agente ───────────────────────────────────────────────────────────────────────────────────────────────────────
# límite de vueltas razonar -> herramienta del nodo ReAct: al llegar, el agente responde con lo que tiene
MAX_ITERACIONES_AGENTE = CONFIG["agente"]["max_iteraciones"]
# red de seguridad de LangGraph: cantidad máxima de pasos (nodos) por invocación del grafo
RECURSION_LIMIT = CONFIG["agente"]["recursion_limit"]
MENSAJES_EN_CONTEXTO = CONFIG["agente"]["mensajes_en_contexto"]

# ── API y guardrails ─────────────────────────────────────────────────────────────────────────────────────────────
API_HOST = CONFIG["api"]["host"]
API_PORT = int(os.getenv("PORT", CONFIG["api"]["port"]))
# tiempo máximo que la API espera al agente antes de devolver una respuesta controlada
TIMEOUT_CONSULTA_SEGUNDOS = float(os.getenv("TIMEOUT_CONSULTA_SEGUNDOS", CONFIG["api"]["timeout_consulta_segundos"]))
MAX_CARACTERES = CONFIG["guardrails"]["max_caracteres"]
