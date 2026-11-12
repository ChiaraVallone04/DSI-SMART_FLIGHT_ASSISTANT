"""Parte C — análisis de costos y latencia a partir de las trazas de LangSmith.

Cada consulta al agente queda trazada en LangSmith como un run raíz "consulta_smart_flight" (tags "tp_final" y
"entrega_4"; se filtra por "entrega_4", que tienen todos los runs, incluidos los de antes de la reorganización), con
todos sus pasos adentro (router, juez, generación, herramientas). LangSmith suma los tokens de todas las llamadas
al LLM del run y calcula su costo con la tarifa del modelo (incluido el descuento de la caché de prompts).

Este script agrupa esos runs por el camino que tomó el grafo y compara cuánto cuesta y cuánto tarda cada uno.
Los inputs bloqueados por los guardrails no aparecen: nunca llegan al LLM, así que su costo es cero.

Uso (desde la raíz del repo, con LANGSMITH_API_KEY y LANGSMITH_PROJECT en el .env):
    python main.py costos              # runs de las últimas 24 horas
    python main.py costos --horas 72
"""
import argparse
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from langsmith import Client

from src.config import CARPETA_RESULTADOS, PRECIO_INPUT_POR_MILLON, PRECIO_OUTPUT_POR_MILLON, RAIZ_REPO
from src.observability import percentil

RUTA_RESULTADOS = CARPETA_RESULTADOS / "resultados_costos.md"
ORDEN_CAMINOS = ["fuera_de_alcance", "conversacion", "rag_directo", "agente_react"]


def obtener_runs(horas: int):
    load_dotenv(RAIZ_REPO / ".env", override=True)
    proyecto = os.getenv("LANGSMITH_PROJECT")
    runs = Client().list_runs(
        project_name=proyecto, is_root=True,
        filter='and(eq(name, "consulta_smart_flight"), has(tags, "entrega_4"))',
        start_time=datetime.now(timezone.utc) - timedelta(hours=horas))
    return proyecto, [r for r in runs if r.status == "success" and r.end_time and r.total_tokens]


def main():
    parser = argparse.ArgumentParser(description="Costos y latencia por camino del grafo, desde LangSmith")
    parser.add_argument("--horas", type=int, default=24)
    analizar(parser.parse_args().horas)


def analizar(horas: int = 24):
    """Agrupa por camino del grafo los runs de las últimas `horas` y escribe resultados/resultados_costos.md."""
    proyecto, runs = obtener_runs(horas)
    if not runs:
        raise SystemExit("No hay runs del agente en LangSmith en esa ventana. Correr antes la matriz o usar la API.")

    grupos = defaultdict(list)
    for r in runs:
        salida = r.outputs or {}
        camino = salida.get("camino") or "desconocido"
        grupos[camino].append({
            "tokens_in": r.prompt_tokens or 0,
            "tokens_out": r.completion_tokens or 0,
            "costo": float(r.total_cost or 0),
            # lo que costaría a precio de lista, sin el descuento de la caché de prompts
            "costo_lista": ((r.prompt_tokens or 0) * PRECIO_INPUT_POR_MILLON
                            + (r.completion_tokens or 0) * PRECIO_OUTPUT_POR_MILLON) / 1_000_000,
            "latencia": (r.end_time - r.start_time).total_seconds(),
            "iteraciones": salida.get("iteraciones") or 0,
            "herramientas": len(salida.get("herramientas_usadas") or []),
        })

    filas = []
    for camino in sorted(grupos, key=lambda c: ORDEN_CAMINOS.index(c) if c in ORDEN_CAMINOS else 99):
        g = grupos[camino]
        filas.append({
            "camino": camino, "n": len(g),
            "tokens": statistics.mean(x["tokens_in"] + x["tokens_out"] for x in g),
            "tokens_in": statistics.mean(x["tokens_in"] for x in g),
            "tokens_out": statistics.mean(x["tokens_out"] for x in g),
            "costo": statistics.mean(x["costo"] for x in g),
            "costo_lista": statistics.mean(x["costo_lista"] for x in g),
            "latencia": statistics.mean(x["latencia"] for x in g),
            "p95": percentil([x["latencia"] for x in g], 95),
            "herramientas": statistics.mean(x["herramientas"] for x in g),
        })
    todos = [x for g in grupos.values() for x in g]
    costo_total = sum(x["costo"] for x in todos)
    costo_lista_total = sum(x["costo_lista"] for x in todos)

    lineas = [
        "# Análisis de costos y latencia por camino del grafo (LangSmith)", "",
        f"Proyecto `{proyecto}`, runs raíz `consulta_smart_flight` con tag `entrega_4` de las últimas "
        f"{horas} h: **{len(todos)} consultas**, costo total **US$ {costo_total:.4f}** "
        f"(US$ {costo_total / len(todos):.5f} por consulta en promedio). Generado el "
        f"{datetime.now():%Y-%m-%d %H:%M}.", "",
        "| Camino | Consultas | Tokens prom. (in + out) | Herram. prom. | Costo prom. (US$) | "
        "Costo por 1.000 consultas | Latencia prom. | Latencia p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in filas:
        lineas.append(
            f"| `{f['camino']}` | {f['n']} | {f['tokens']:,.0f} ({f['tokens_in']:,.0f} + {f['tokens_out']:,.0f}) | "
            f"{f['herramientas']:.1f} | {f['costo']:.5f} | US$ {f['costo'] * 1000:.2f} | {f['latencia']:.1f} s | "
            f"{f['p95']:.1f} s |")
    ahorro = 100 * (1 - costo_total / costo_lista_total) if costo_lista_total else 0
    lineas += [
        "", f"Caché de prompts de OpenAI: a precio de lista estas consultas habrían costado US$ "
        f"{costo_lista_total:.4f}; con los tokens servidos desde la caché costaron US$ {costo_total:.4f} "
        f"({ahorro:.0f}% menos).",
        "", "Los inputs bloqueados por los guardrails no figuran porque no llaman al LLM (costo cero).",
    ]
    texto = "\n".join(lineas) + "\n"
    RUTA_RESULTADOS.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Guardado en resultados/{RUTA_RESULTADOS.name}")


if __name__ == "__main__":
    main()
