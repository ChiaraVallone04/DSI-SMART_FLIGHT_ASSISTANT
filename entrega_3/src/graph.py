"""Parte A — el agente: grafo de LangGraph con router, nodo ReAct, memoria SQLite y guardrail de salida.

    START -> inicio -> router --(fuera de alcance)--------> fuera_de_alcance --+
                          |----(saludo / conversación)----> conversacion -------+
                          |----(consulta de catálogo)-----> rag_directo --------+--> sintetizar -> END
                          +----(fecha, precio del día,      agente_react <-+    |
                                reserva, varios pasos)----> |   (ReAct)    |    |
                                                            +-> herramientas    |
                                                            +-------------------+

- inicio: reinicia los contadores del turno (la memoria de la conversación se conserva).
- router: clasifica la intención con las categorías de la Entrega 1 (entrega_1/schemas.py) y reescribe la consulta para
  que se entienda sola ("¿y la vuelta?" -> "¿Cómo es la ruta Roma–Madrid?"). El CAMINO lo decide el código.
- rag_directo: camino determinista y barato para preguntas de catálogo: llama una vez al RAG de la Entrega 3.
- agente_react: el loop razonar -> herramienta -> observar, con las 3 herramientas y un límite de iteraciones.
- conversacion / fuera_de_alcance: saludos y preguntas sobre la charla / rechazo amable sin llamar al LLM.
- sintetizar: punto único de salida: enmascara datos personales en la respuesta antes de guardarla y devolverla.
"""
import importlib.util
import sqlite3
import threading
from datetime import date
from typing import Annotated, Literal, TypedDict, get_args

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.config import (
    MAX_ITERACIONES_AGENTE,
    MENSAJES_EN_CONTEXTO,
    MODELO_LLM,
    PRECIO_INPUT_CACHE_POR_MILLON,
    PRECIO_INPUT_POR_MILLON,
    PRECIO_OUTPUT_POR_MILLON,
    RECURSION_LIMIT,
    REINTENTOS_LLM,
    RUTA_MEMORIA,
    RUTA_SCHEMAS_E1,
    TEMPERATURA_LLM,
    TIMEOUT_LLM_SEGUNDOS,
)
from src.guardrails import enmascarar_pii
from src.prompts import (
    AVISO_LIMITE,
    AVISO_RESPUESTA_LIMITE,
    MENSAJE_FUERA_DE_ALCANCE,
    PROMPT_AGENTE,
    PROMPT_CONVERSACION,
    PROMPT_ROUTER,
)
from src.tools import HERRAMIENTAS, buscar_info_rutas


def _cargar_esquema_entrega_1():
    """El esquema de la Entrega 1 (entrega_1/schemas.py) se carga por ruta y no por sys.path: la carpeta de la E1
    tiene módulos con nombres genéricos (config, prompts, schemas) que chocarían con otros."""
    spec = importlib.util.spec_from_file_location("schemas_entrega_1", RUTA_SCHEMAS_E1)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo.OutputResponse


OutputResponse = _cargar_esquema_entrega_1()

# ── Estado ───────────────────────────────────────────────────────────────────────────────────────────────────────

# las 5 intenciones de la Entrega 1 + "conversacion" (saludos y preguntas sobre la propia charla)
INTENCIONES = (*get_args(OutputResponse.model_fields["intencion"].annotation), "conversacion")
Camino = Literal["rag_directo", "agente_react", "conversacion", "fuera_de_alcance"]


class EstadoAgente(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # historial: lo persiste el checkpointer por thread_id
    intencion: str
    camino: str
    consulta_autocontenida: str
    herramientas_usadas: list[str]  # del turno actual
    iteraciones: int                # vueltas del nodo ReAct en el turno actual
    limite_alcanzado: bool
    pii_detectada: list[str]


# ── Modelos ──────────────────────────────────────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model=MODELO_LLM, temperature=TEMPERATURA_LLM, timeout=TIMEOUT_LLM_SEGUNDOS,
                 max_retries=REINTENTOS_LLM)
llm_con_herramientas = llm.bind_tools(HERRAMIENTAS)
HERRAMIENTAS_POR_NOMBRE = {h.name: h for h in HERRAMIENTAS}

DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def _hoy():
    hoy = date.today()
    return f"{hoy.isoformat()} ({DIAS_SEMANA[hoy.weekday()]})"


def _historial_texto(mensajes: list[AnyMessage], ultimos: int = 6) -> str:
    """Últimos mensajes de usuario y asistente (sin llamadas a herramientas), para el router."""
    lineas = []
    for m in mensajes:
        if isinstance(m, HumanMessage):
            lineas.append(f"Usuario: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lineas.append(f"Asistente: {m.content}")
    return "\n".join(lineas[-ultimos:])


def _historial_reciente(mensajes: list[AnyMessage]) -> list[AnyMessage]:
    """Recorta la memoria a los últimos mensajes (config.toml: mensajes_en_contexto), empezando siempre en un mensaje
    del usuario (así nunca queda una respuesta de herramienta sin la llamada que la originó, algo que OpenAI
    rechaza)."""
    return trim_messages(mensajes, max_tokens=MENSAJES_EN_CONTEXTO, token_counter=len, strategy="last", start_on="human",
                         include_system=False, allow_partial=False)


# ── Nodo: inicio ─────────────────────────────────────────────────────────────────────────────────────────────────

def inicio(estado: EstadoAgente):
    return {"herramientas_usadas": [], "iteraciones": 0, "limite_alcanzado": False, "pii_detectada": [],
            "camino": "", "intencion": "", "consulta_autocontenida": ""}


# ── Nodo: router ─────────────────────────────────────────────────────────────────────────────────────────────────

class DecisionRouter(BaseModel):
    intencion: Literal[INTENCIONES] = Field(description="Intención del ÚLTIMO mensaje del usuario")
    pide_datos_de_vuelos: bool = Field(
        description="True si el último mensaje pide CUALQUIER dato sobre rutas, vuelos, aerolíneas, precios, "
                    "horarios, vuelos directos o reservas, aunque además salude o pregunte por la conversación.")
    necesita_datos_en_vivo: bool = Field(
        description="True si para responder hacen falta vuelos de una FECHA concreta (disponibilidad, precio de "
                    "ese día, asientos) o hay que crear o continuar una reserva. False si alcanza con información "
                    "general del catálogo.")
    consulta_autocontenida: str = Field(
        description="El último mensaje reescrito para que se entienda solo, reemplazando referencias a la "
                    "conversación ('esa ruta', '¿y la vuelta?') por las ciudades o datos concretos.")


router_llm = llm.with_structured_output(DecisionRouter)


def router(estado: EstadoAgente):
    mensajes = estado["messages"]
    decision: DecisionRouter = router_llm.invoke(PROMPT_ROUTER.format(
        hoy=_hoy(), historial=_historial_texto(mensajes[:-1]) or "(sin mensajes previos)",
        mensaje=mensajes[-1].content))
    return {"intencion": decision.intencion, "consulta_autocontenida": decision.consulta_autocontenida,
            "camino": _elegir_camino(decision)}


def _elegir_camino(decision: DecisionRouter) -> Camino:
    """La decisión de qué rama usar la toma el código a partir de la clasificación, no el LLM."""
    if decision.intencion == "fuera_de_alcance" and not decision.pide_datos_de_vuelos:
        return "fuera_de_alcance"
    # la rama de conversación no tiene herramientas: si el mensaje pide algún dato de vuelos, no puede ir ahí
    # (respondería de memoria). Cubre mensajes mixtos como "¿qué te pregunté? ¿y tiene vuelo directo?"
    if decision.intencion == "conversacion" and not decision.pide_datos_de_vuelos:
        return "conversacion"
    if decision.intencion == "crear_reserva" or decision.necesita_datos_en_vivo:
        return "agente_react"
    return "rag_directo"


# ── Nodo: rag_directo (camino determinista) ──────────────────────────────────────────────────────────────────────

def rag_directo(estado: EstadoAgente, config: RunnableConfig):
    consulta = estado.get("consulta_autocontenida") or estado["messages"][-1].content
    respuesta = buscar_info_rutas.invoke({"consulta": consulta}, config=config)
    return {"messages": [AIMessage(content=respuesta)], "herramientas_usadas": ["buscar_info_rutas"]}


# ── Nodos: agente_react + herramientas (el loop ReAct) ───────────────────────────────────────────────────────────

def agente_react(estado: EstadoAgente, config: RunnableConfig):
    iteraciones = estado.get("iteraciones", 0) + 1
    mensajes = [SystemMessage(PROMPT_AGENTE.format(hoy=_hoy())), *_historial_reciente(estado["messages"])]
    if iteraciones >= MAX_ITERACIONES_AGENTE:
        # límite de iteraciones: la última vuelta se hace SIN herramientas, así el agente está obligado a cerrar
        respuesta = llm.invoke([*mensajes, SystemMessage(AVISO_LIMITE)], config=config)
        return {"messages": [respuesta], "iteraciones": iteraciones, "limite_alcanzado": True}
    respuesta = llm_con_herramientas.invoke(mensajes, config=config)
    return {"messages": [respuesta], "iteraciones": iteraciones}


def herramientas(estado: EstadoAgente, config: RunnableConfig):
    """Ejecuta las herramientas que pidió el agente y registra cuáles se usaron en el turno."""
    llamadas = estado["messages"][-1].tool_calls
    resultados = []
    for llamada in llamadas:
        herramienta = HERRAMIENTAS_POR_NOMBRE.get(llamada["name"])
        if herramienta is None:
            resultados.append(ToolMessage(f"Error: la herramienta {llamada['name']} no existe.",
                                          tool_call_id=llamada["id"], name=llamada["name"], status="error"))
            continue
        try:
            resultados.append(herramienta.invoke(llamada, config=config))
        except Exception as error:  # una herramienta que falla no tira abajo la conversación
            resultados.append(ToolMessage(f"Error: la herramienta falló ({type(error).__name__}).",
                                          tool_call_id=llamada["id"], name=llamada["name"], status="error"))
    usadas = estado.get("herramientas_usadas", []) + [llamada["name"] for llamada in llamadas]
    return {"messages": resultados, "herramientas_usadas": usadas}


def despues_del_agente(estado: EstadoAgente) -> Literal["herramientas", "sintetizar"]:
    ultimo = estado["messages"][-1]
    return "herramientas" if getattr(ultimo, "tool_calls", None) else "sintetizar"


# ── Nodos: conversacion y fuera_de_alcance ───────────────────────────────────────────────────────────────────────

def conversacion(estado: EstadoAgente, config: RunnableConfig):
    mensajes = [SystemMessage(PROMPT_CONVERSACION), *_historial_reciente(estado["messages"])]
    # sin las llamadas a herramientas del historial: este nodo no las tiene, y OpenAI exige que vengan de a pares
    mensajes = [m for m in mensajes if not isinstance(m, ToolMessage) and not getattr(m, "tool_calls", None)]
    return {"messages": [llm.invoke(mensajes, config=config)]}


def fuera_de_alcance(estado: EstadoAgente):
    # escape decidido por código: no hace falta gastar una llamada al LLM para decir que no
    return {"messages": [AIMessage(content=MENSAJE_FUERA_DE_ALCANCE)]}


# ── Nodo: sintetizar (guardrail de salida, punto único) ──────────────────────────────────────────────────────────

def valores_sensibles(mensajes: list[AnyMessage]) -> dict[str, list[str]]:
    """Datos personales que el agente le pasó al actuador en este thread (estado enriquecido): el backend los
    necesita para reservar, pero no pueden salir en las respuestas, en /estado ni en los logs."""
    valores = {"NOMBRE": [], "DOCUMENTO": [], "EMAIL": []}
    for m in mensajes:
        for llamada in getattr(m, "tool_calls", None) or []:
            if llamada["name"] != "crear_reserva":
                continue
            args = llamada["args"]
            nombre = args.get("pasajero_nombre") or ""
            # el nombre completo y también cada parte ("¡Gracias, Lucía!")
            valores["NOMBRE"] += [nombre, *[p for p in nombre.split() if len(p) >= 4]]
            documento = args.get("pasajero_documento") or ""
            valores["DOCUMENTO"] += [documento, documento.replace(".", "").replace(" ", "")]
            valores["EMAIL"].append(args.get("email") or "")
    return valores


def sintetizar(estado: EstadoAgente):
    ultimo = estado["messages"][-1]
    texto = ultimo.content if isinstance(ultimo.content, str) else str(ultimo.content)
    if not texto.strip():
        texto = "No pude completar la consulta. ¿Podés reformularla?"
    enmascarado, tipos = enmascarar_pii(texto, valores_sensibles(estado["messages"]))
    # mismo id -> add_messages REEMPLAZA el mensaje: en la memoria también queda la versión enmascarada
    return {"messages": [AIMessage(content=enmascarado, id=ultimo.id)], "pii_detectada": tipos}


# ── Grafo ────────────────────────────────────────────────────────────────────────────────────────────────────────

def construir_grafo():
    grafo = StateGraph(EstadoAgente)
    grafo.add_node("inicio", inicio)
    grafo.add_node("router", router)
    grafo.add_node("rag_directo", rag_directo)
    grafo.add_node("agente_react", agente_react)
    grafo.add_node("herramientas", herramientas)
    grafo.add_node("conversacion", conversacion)
    grafo.add_node("fuera_de_alcance", fuera_de_alcance)
    grafo.add_node("sintetizar", sintetizar)

    grafo.add_edge(START, "inicio")
    grafo.add_edge("inicio", "router")
    grafo.add_conditional_edges("router", lambda estado: estado["camino"],
                                ["rag_directo", "agente_react", "conversacion", "fuera_de_alcance"])
    grafo.add_conditional_edges("agente_react", despues_del_agente, ["herramientas", "sintetizar"])
    grafo.add_edge("herramientas", "agente_react")
    for nodo in ("rag_directo", "conversacion", "fuera_de_alcance"):
        grafo.add_edge(nodo, "sintetizar")
    grafo.add_edge("sintetizar", END)
    return grafo


# memoria persistente: UNA conexión y UN checkpointer para todo el proceso (si se crearan por request, cada
# consulta empezaría sin historia). check_same_thread=False porque la API atiende requests desde varios hilos;
# SqliteSaver serializa los accesos con su propio lock
conexion_memoria = sqlite3.connect(RUTA_MEMORIA, check_same_thread=False)
memoria = SqliteSaver(conexion_memoria)
agente = construir_grafo().compile(checkpointer=memoria)


# ── Interfaz pública (el "contrato" que usa la API) ──────────────────────────────────────────────────────────────

def _costo_usd(uso: dict) -> tuple[int, int, float]:
    """Tokens y costo de todas las llamadas al LLM del turno. Los tokens de entrada que OpenAI sirve desde su
    caché de prompts (el system prompt repetido, el historial) se cobran a mitad de precio."""
    entrada = sum(u.get("input_tokens", 0) for u in uso.values())
    salida = sum(u.get("output_tokens", 0) for u in uso.values())
    en_cache = sum((u.get("input_token_details") or {}).get("cache_read", 0) for u in uso.values())
    costo = ((entrada - en_cache) * PRECIO_INPUT_POR_MILLON + en_cache * PRECIO_INPUT_CACHE_POR_MILLON
             + salida * PRECIO_OUTPUT_POR_MILLON) / 1_000_000
    return entrada, salida, costo


def _reparar_hilo(config: dict):
    """Si el grafo se cortó a mitad de un loop (GraphRecursionError), cierra las llamadas a herramientas que
    quedaron sin respuesta y agrega un mensaje final, para que el próximo turno del thread no falle."""
    mensajes = agente.get_state(config).values.get("messages", [])
    ultimo = mensajes[-1] if mensajes else None
    reparacion = []
    if isinstance(ultimo, AIMessage) and ultimo.tool_calls:
        reparacion = [ToolMessage("Cancelado por límite de pasos.", tool_call_id=c["id"], name=c["name"])
                      for c in ultimo.tool_calls]
    reparacion.append(AIMessage(content=AVISO_RESPUESTA_LIMITE))
    agente.update_state(config, {"messages": reparacion}, as_node="sintetizar")


# un lock por conversación: dos requests simultáneos con el MISMO thread_id leerían el mismo estado y el último en
# guardar pisaría el turno del otro. Conversaciones distintas siguen corriendo en paralelo
_locks_por_hilo: dict[str, threading.Lock] = {}
_lock_registro = threading.Lock()


def _lock_del_hilo(thread_id: str) -> threading.Lock:
    with _lock_registro:
        return _locks_por_hilo.setdefault(thread_id, threading.Lock())


def responder(pregunta: str, thread_id: str, user_id: str | None = None) -> dict:
    """Corre un turno de la conversación `thread_id` y devuelve la respuesta (ya enmascarada) con sus métricas.

    Los turnos de una misma conversación se atienden de a uno (en orden de llegada)."""
    with _lock_del_hilo(thread_id):
        return _responder(pregunta, thread_id, user_id)


def _responder(pregunta: str, thread_id: str, user_id: str | None) -> dict:
    medidor = UsageMetadataCallbackHandler()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
        "callbacks": [medidor],
        "run_name": "consulta_smart_flight",
        "tags": ["tp_final", "entrega_4"],
        "metadata": {"thread_id": thread_id, "user_id": user_id},
    }
    try:
        estado = agente.invoke({"messages": [HumanMessage(content=pregunta)]}, config=config)
        respuesta = estado["messages"][-1].content
    except GraphRecursionError:
        _reparar_hilo(config)
        estado = agente.get_state(config).values
        respuesta = AVISO_RESPUESTA_LIMITE
        estado["limite_alcanzado"] = True

    tokens_entrada, tokens_salida, costo = _costo_usd(medidor.usage_metadata)
    return {
        "respuesta": respuesta,
        "herramientas_usadas": estado.get("herramientas_usadas", []),
        "intencion": estado.get("intencion"),
        "camino": estado.get("camino"),
        "iteraciones": estado.get("iteraciones", 0),
        "limite_alcanzado": estado.get("limite_alcanzado", False),
        "pii_enmascarada": estado.get("pii_detectada", []),
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "costo_usd": costo,
    }


ROLES = {"human": "usuario", "ai": "asistente", "tool": "herramienta"}


def obtener_historial(thread_id: str) -> dict | None:
    """Historial completo del thread desde la memoria SQLite, con TODO el contenido enmascarado (el mensaje del
    usuario se guarda tal cual lo escribió, y las herramientas pueden haber recibido su documento)."""
    estado = agente.get_state({"configurable": {"thread_id": thread_id}})
    if not estado.values or not estado.values.get("messages"):
        return None
    mensajes = []
    sensibles = valores_sensibles(estado.values["messages"])
    for m in estado.values["messages"]:
        contenido = m.content if isinstance(m.content, str) else str(m.content)
        contenido, _ = enmascarar_pii(contenido, sensibles)
        item = {"rol": ROLES.get(m.type, m.type), "contenido": contenido}
        if isinstance(m, AIMessage) and m.tool_calls:
            item["herramientas_llamadas"] = [c["name"] for c in m.tool_calls]
        if isinstance(m, ToolMessage):
            item["herramienta"] = m.name
        mensajes.append(item)
    return {
        "thread_id": thread_id,
        "mensajes": mensajes,
        "cantidad_mensajes": len(mensajes),
        "ultima_intencion": estado.values.get("intencion"),
        "ultimo_camino": estado.values.get("camino"),
    }


if __name__ == "__main__":
    import uuid

    hilo = f"prueba-{uuid.uuid4().hex[:6]}"
    for pregunta in ["Hola, ¿qué podés hacer?",
                     "¿Qué aerolíneas operan la ruta Madrid–Roma?",
                     "¿Y cuánto sale volar esa ruta el 5 de diciembre de 2026?",
                     "¿Cuál es la capital de Australia?"]:
        r = responder(pregunta, hilo)
        print(f"\n> {pregunta}\n[{r['camino']} | {r['intencion']} | herramientas={r['herramientas_usadas']} | "
              f"iter={r['iteraciones']} | {r['tokens_entrada']}+{r['tokens_salida']} tokens | "
              f"US$ {r['costo_usd']:.5f}]\n{r['respuesta']}")
