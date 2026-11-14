"""Parte A — matriz de validación del agente: 4 escenarios + memoria + límite de iteraciones.

Cada escenario corre contra el agente real (LLM + herramientas) y se verifica automáticamente: qué camino del
grafo tomó, qué herramientas usó y qué tiene (o no tiene) la respuesta. Los resultados se guardan en
resultados/resultados_matriz.md.

Uso (desde la raíz del repo):
    python main.py matriz
"""
import time
import uuid
from datetime import date, timedelta

from langchain_core.tracers.langchain import wait_for_all_tracers

from src import graph as modulo_agente
from src.config import CARPETA_RESULTADOS
from src.graph import obtener_historial, responder
from src.tools import listar_reservas

RUTA_RESULTADOS = CARPETA_RESULTADOS / "resultados_matriz.md"
FECHA = (date.today() + timedelta(days=30)).isoformat()


def nuevo_hilo(nombre: str) -> str:
    return f"matriz-{nombre}-{uuid.uuid4().hex[:6]}"


def chequear(condicion: bool, descripcion: str, chequeos: list):
    chequeos.append((bool(condicion), descripcion))


def turno(hilo: str, pregunta: str, registro: list) -> dict:
    inicio = time.perf_counter()
    r = responder(pregunta, hilo)
    r["latencia_s"] = time.perf_counter() - inicio
    registro.append((pregunta, r))
    return r


# ── Escenarios ───────────────────────────────────────────────────────────────────────────────────────────────────

def escenario_1_catalogo():
    """Pregunta general de catálogo -> camino determinista rag_directo con el retriever de la Entrega 3."""
    registro, chequeos = [], []
    r = turno(nuevo_hilo("catalogo"),
              "¿La ruta Barcelona–Roma suele tener vuelos directos y qué tipo de aerolíneas la operan?", registro)
    chequear(r["camino"] == "rag_directo", "camino = rag_directo", chequeos)
    chequear(r["herramientas_usadas"] == ["buscar_info_rutas"], "herramienta = buscar_info_rutas", chequeos)
    chequear("RUTA-BCN-FCO" in r["respuesta"], "cita la fuente [RUTA-BCN-FCO]", chequeos)
    return "1. Consulta de catálogo (RAG)", registro, chequeos


def escenario_2_sensor():
    """Disponibilidad de una fecha -> el agente ReAct elige solo el sensor."""
    registro, chequeos = [], []
    r = turno(nuevo_hilo("sensor"), f"¿Qué vuelos hay de Copenhague a Oslo el {FECHA} y cuánto salen?", registro)
    chequear(r["camino"] == "agente_react", "camino = agente_react", chequeos)
    chequear("consultar_disponibilidad" in r["herramientas_usadas"], "usó consultar_disponibilidad", chequeos)
    chequear("crear_reserva" not in r["herramientas_usadas"], "NO reservó (nadie lo pidió)", chequeos)
    chequear("EUR" in r["respuesta"] or "€" in r["respuesta"], "informa precios del día", chequeos)
    return "2. Disponibilidad de una fecha (sensor)", registro, chequeos


def escenario_3_reserva():
    """Reserva en varios turnos: consulta -> elección con datos del pasajero -> confirmación -> actuador."""
    registro, chequeos = [], []
    hilo = nuevo_hilo("reserva")
    reservas_antes = len(listar_reservas())

    r1 = turno(hilo, f"Quiero volar de Madrid a Roma el {FECHA}, ¿qué opciones tengo?", registro)
    chequear("consultar_disponibilidad" in r1["herramientas_usadas"], "turno 1: consultó disponibilidad", chequeos)

    r2 = turno(hilo, "Quiero el más barato que tenga lugar. Soy Lucía Fernández, pasaporte XK1234567.", registro)
    reservo_sin_confirmar = "crear_reserva" in r2["herramientas_usadas"]
    chequear(not reservo_sin_confirmar, "turno 2: NO reserva sin confirmación explícita", chequeos)

    r3 = turno(hilo, "Sí, confirmo, reservalo.", registro) if not reservo_sin_confirmar else r2
    chequear("crear_reserva" in r3["herramientas_usadas"], "turno 3: usó crear_reserva (actuador)", chequeos)
    chequear("SFA-" in r3["respuesta"], "informa el código de reserva SFA-XXXXXX", chequeos)
    chequear(len(listar_reservas()) == reservas_antes + 1, "la reserva quedó guardada en reservas.db", chequeos)
    texto_total = " ".join(r["respuesta"] for _, r in registro)
    chequear("XK1234567" not in texto_total and "Lucía Fernández" not in texto_total,
             "las respuestas NO exponen el pasaporte ni el nombre (guardrail de salida)", chequeos)
    historial = obtener_historial(hilo)
    chequear(historial and all("XK1234567" not in m["contenido"] for m in historial["mensajes"]),
             "/estado tampoco expone el pasaporte (historial enmascarado)", chequeos)
    return "3. Reserva en varios pasos (sensor + actuador)", registro, chequeos


def escenario_4_fuera_de_alcance():
    """Fuera de dominio -> rechazo sin herramientas; ruta inexistente -> frase de escape del RAG."""
    registro, chequeos = [], []
    r1 = turno(nuevo_hilo("fuera"), "¿Me pasás una receta de ñoquis caseros?", registro)
    chequear(r1["camino"] == "fuera_de_alcance", "receta: camino = fuera_de_alcance", chequeos)
    chequear(r1["herramientas_usadas"] == [], "receta: no usó herramientas", chequeos)
    r2 = turno(nuevo_hilo("escape"), "¿Qué vuelos hay entre Madrid y Berlín?", registro)
    chequear("No poseo información" in r2["respuesta"] or "no está en el catálogo" in r2["respuesta"].lower()
             or "no tengo" in r2["respuesta"].lower(), "ruta fuera del catálogo: no inventa (escape)", chequeos)
    return "4. Fuera de alcance y fuera de catálogo (escape)", registro, chequeos


def extra_memoria():
    """La memoria por thread_id resuelve referencias a turnos anteriores, y otro thread no la ve."""
    registro, chequeos = [], []
    hilo = nuevo_hilo("memoria")
    turno(hilo, "¿Qué aerolíneas vuelan entre Atenas y Londres?", registro)
    r2 = turno(hilo, "¿Y suele haber vuelo directo en esa ruta?", registro)
    chequear("RUTA-ATH-LHR" in r2["respuesta"], "resuelve 'esa ruta' = Atenas–Londres", chequeos)
    r3 = turno(nuevo_hilo("otro"), "¿Y suele haber vuelo directo en esa ruta?", registro)
    chequear("RUTA-ATH-LHR" not in r3["respuesta"], "otro thread_id NO ve esa conversación", chequeos)
    return "Extra. Memoria por thread_id", registro, chequeos


def extra_limite_iteraciones():
    """Con el límite bajado a 2, una consulta que pide muchas herramientas se corta y el agente igual responde."""
    registro, chequeos = [], []
    original = modulo_agente.MAX_ITERACIONES_AGENTE
    modulo_agente.MAX_ITERACIONES_AGENTE = 2
    try:
        r = turno(nuevo_hilo("limite"),
                  f"Para el {FECHA} decime el vuelo más barato de Madrid a Roma, después el de Barcelona a Roma "
                  "y después el de Copenhague a Oslo. Consultalos de a uno.", registro)
    finally:
        modulo_agente.MAX_ITERACIONES_AGENTE = original
    chequear(r["iteraciones"] <= 2, "no supera 2 iteraciones", chequeos)
    chequear(r["limite_alcanzado"], "marca limite_alcanzado = True", chequeos)
    chequear(bool(r["respuesta"].strip()), "igual devuelve una respuesta (no se cuelga ni tira error)", chequeos)
    return "Extra. Límite de iteraciones (MAX = 2)", registro, chequeos


ESCENARIOS = [escenario_1_catalogo, escenario_2_sensor, escenario_3_reserva, escenario_4_fuera_de_alcance,
              extra_memoria, extra_limite_iteraciones]


def main():
    secciones, total_ok, total = [], 0, 0
    for escenario in ESCENARIOS:
        titulo, registro, chequeos = escenario()
        ok = sum(c for c, _ in chequeos)
        total_ok, total = total_ok + ok, total + len(chequeos)
        print(f"\n=== {titulo}: {ok}/{len(chequeos)} ===")
        for c, d in chequeos:
            print(f"  {'PASS' if c else 'FAIL'}  {d}")

        lineas = [f"## {titulo} — {ok}/{len(chequeos)}", ""]
        for pregunta, r in registro:
            lineas += [f"**Usuario:** {pregunta}", "",
                       f"*camino `{r['camino']}` · intención `{r['intencion']}` · herramientas "
                       f"`{r['herramientas_usadas']}` · iteraciones {r['iteraciones']} · "
                       f"{r['tokens_entrada']}+{r['tokens_salida']} tokens · US$ {r['costo_usd']:.5f} · "
                       f"{r['latencia_s']:.1f} s*", "",
                       "> " + r["respuesta"].replace("\n", "\n> "), ""]
        lineas += ["| Chequeo | Resultado |", "|---|---|"]
        lineas += [f"| {d} | {'✅' if c else '❌'} |" for c, d in chequeos]
        secciones.append("\n".join(lineas))

    encabezado = (f"# Matriz de validación del agente\n\nCorrida del {date.today().isoformat()} "
                  f"(fecha de vuelo usada: {FECHA}). Resultado: **{total_ok}/{total} chequeos**.\n")
    RUTA_RESULTADOS.write_text(encabezado + "\n\n".join(secciones) + "\n", encoding="utf-8")
    print(f"\nTOTAL: {total_ok}/{total} — guardado en resultados/{RUTA_RESULTADOS.name}")
    wait_for_all_tracers()  # que las trazas lleguen a LangSmith antes de que termine el proceso


if __name__ == "__main__":
    main()
