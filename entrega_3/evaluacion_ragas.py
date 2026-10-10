"""Entrega 3 — Parte C: evaluación con RAGAS sobre el golden dataset.

Corre las preguntas de golden_dataset.json contra el RAG básico (Parte A) y el avanzado
(Parte B: chunking + reranking) y mide Faithfulness, Answer Relevancy, Context Precision y
Context Recall con RAGAS (API de ragas.metrics.collections, versión 0.4.3).

Uso (desde la raíz del repo, con la base de la Entrega 2 y la colección de chunks ya construidas):
    python entrega_3/evaluacion_ragas.py                     # los dos pipelines (C.2 y C.3)
    python entrega_3/evaluacion_ragas.py --pipeline basico   # solo el baseline (C.2)
    python entrega_3/evaluacion_ragas.py --pipeline avanzado
"""
import argparse
import asyncio
import json
import math
import sys
import types
from pathlib import Path

# RAGAS importa la librería `datasets` (y con ella pandas) aunque las métricas de ragas.metrics.collections no la
# usan. En Windows con Smart App Control activado, la DLL de pandas puede quedar bloqueada: en ese caso se registra
# un módulo `datasets` vacío para que RAGAS cargue igual. Donde pandas carga normalmente, se usa la librería real.
try:
    import datasets  # noqa: F401
except ImportError:
    sustituto = types.ModuleType("datasets")
    sustituto.Dataset = type("Dataset", (), {})
    sys.modules["datasets"] = sustituto

from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms.base import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

# rag_pipeline.py también carga el .env (vía vector_db.py) y deja configurado LangSmith
from rag_pipeline import rag_chain, rag_chain_avanzado

CARPETA = Path(__file__).resolve().parent
RUTA_DATASET = CARPETA / "golden_dataset.json"
RUTA_RESULTADOS = CARPETA / "resultados_ragas.json"

PIPELINES = {
    "basico": rag_chain,             # fichas completas, k = 4, umbral 0.50
    "avanzado": rag_chain_avanzado,  # chunks 500/100, k = 8, juez LLM, top 3
}

METRICAS = ["faithfulness", "answer_relevancy",
            "context_precision", "context_recall"]
# las que RAGAS no puede calcular sin contexto recuperado
METRICAS_CON_CONTEXTO = ["faithfulness", "context_precision", "context_recall"]

# el evaluador usa el mismo modelo que el generador y los mismos embeddings que la base
cliente = AsyncOpenAI()
llm_evaluador = llm_factory("gpt-4o-mini", client=cliente)
embeddings_evaluador = embedding_factory(
    "openai", model="text-embedding-3-small", client=cliente)

faithfulness = Faithfulness(llm=llm_evaluador)
answer_relevancy = AnswerRelevancy(
    llm=llm_evaluador, embeddings=embeddings_evaluador)
context_precision = ContextPrecision(llm=llm_evaluador)
context_recall = ContextRecall(llm=llm_evaluador)


def cargar_dataset():
    with open(RUTA_DATASET, encoding="utf-8") as f:
        return json.load(f)


async def evaluar_pregunta(item: dict, respuesta: str, contextos: list[str]):
    """Las 4 métricas de una pregunta.

    Faithfulness, Context Precision y Context Recall necesitan contexto: si el retriever no devolvió
    nada (como en las preguntas fuera de catálogo), RAGAS tira error, así que quedan en NaN
    ("no aplica") en vez de inventar un puntaje.
    """
    pregunta, referencia = item["pregunta"], item["ground_truth"]
    puntajes = {metrica: math.nan for metrica in METRICAS_CON_CONTEXTO}
    puntajes["answer_relevancy"] = (await answer_relevancy.ascore(
        user_input=pregunta, response=respuesta)).value
    if contextos:
        puntajes["faithfulness"] = (await faithfulness.ascore(
            user_input=pregunta, response=respuesta, retrieved_contexts=contextos)).value
        puntajes["context_precision"] = (await context_precision.ascore(
            user_input=pregunta, reference=referencia, retrieved_contexts=contextos)).value
        puntajes["context_recall"] = (await context_recall.ascore(
            user_input=pregunta, retrieved_contexts=contextos, reference=referencia)).value
    return puntajes


def promedio(valores: list[float]):
    validos = [v for v in valores if not math.isnan(v)]
    return {"promedio": sum(validos) / len(validos) if validos else math.nan, "n": len(validos)}


async def evaluar_pipeline(nombre: str, dataset: list[dict]):
    chain = PIPELINES[nombre]
    filas = []
    for item in dataset:
        # cada invocación queda trazada en LangSmith si LANGSMITH_TRACING=true
        resultado = await chain.ainvoke(
            item["pregunta"],
            config={"run_name": f"C ragas {nombre} {item['id']}", "tags": [
                "entrega_3", "C", nombre]},
        )
        contextos = [d.page_content for d in resultado["context"]]
        puntajes = await evaluar_pregunta(item, resultado["answer"], contextos)
        filas.append({
            "id": item["id"],
            "tipo": item["tipo"],
            "pregunta": item["pregunta"],
            "respuesta": resultado["answer"],
            "fuentes": [d.metadata.get("id_chunk", d.metadata["id"]) for d in resultado["context"]],
            **puntajes,
        })
        print(f"  [{item['id']}] " + " | ".join(
            f"{m} {puntajes[m]:.3f}" if not math.isnan(puntajes[m]) else f"{m} n/a" for m in METRICAS))

    # dos promedios: con todas las preguntas y sin las de escape. Answer Relevancy le pone 0 a cualquier
    # respuesta evasiva ("no sé"), así que la frase de escape correcta igual baja su promedio
    con_respuesta = [f for f in filas if f["tipo"] != "escape"]
    resumen = {
        alcance: {m: promedio([f[m] for f in conjunto]) for m in METRICAS}
        for alcance, conjunto in (("todas", filas), ("sin_escape", con_respuesta))
    }
    return {"filas": filas, "resumen": resumen}


def imprimir_resumen(resultados: dict, dataset: list[dict]):
    cantidad_escape = sum(1 for item in dataset if item["tipo"] == "escape")
    titulos = {
        "todas": f"Promedio sobre las {len(dataset)} preguntas",
        "sin_escape": f"Promedio sin las {cantidad_escape} de escape ({len(dataset) - cantidad_escape} preguntas)",
    }
    for alcance, titulo in titulos.items():
        print(f"\n{titulo}")
        print(f"  {'métrica':<20}" + "".join(f"{p:>14}" for p in resultados))
        for m in METRICAS:
            celdas = []
            for p in resultados:
                r = resultados[p]["resumen"][alcance][m]
                celdas.append(
                    f"{r['promedio']:.3f} (n={r['n']})" if r["n"] else "n/a")
            print(f"  {m:<20}" + "".join(f"{c:>14}" for c in celdas))


def reemplazar_nan(valor):
    """NaN no es JSON válido: las métricas que no aplican se guardan como null."""
    if isinstance(valor, dict):
        limpio = {clave: reemplazar_nan(v) for clave, v in valor.items()}
    elif isinstance(valor, list):
        limpio = [reemplazar_nan(v) for v in valor]
    elif isinstance(valor, float) and math.isnan(valor):
        limpio = None
    else:
        limpio = valor
    return limpio


async def main(pipelines: list[str]):
    dataset = cargar_dataset()
    print(f"Golden dataset: {len(dataset)} preguntas ({RUTA_DATASET.name})")

    resultados = {}
    for nombre in pipelines:
        print(f"\nEvaluando pipeline '{nombre}'...")
        resultados[nombre] = await evaluar_pipeline(nombre, dataset)

    imprimir_resumen(resultados, dataset)

    # se guardan junto al dataset: los resultados también son parte del entregable (C.2 y C.3)
    anteriores = json.loads(RUTA_RESULTADOS.read_text(
        encoding="utf-8")) if RUTA_RESULTADOS.exists() else {}
    anteriores.update(resultados)
    RUTA_RESULTADOS.write_text(
        json.dumps(reemplazar_nan(anteriores),
                   ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nResultados guardados en {RUTA_RESULTADOS.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación RAGAS de la Entrega 3 (Parte C)")
    parser.add_argument(
        "--pipeline", choices=["basico", "avanzado", "ambos"], default="ambos")
    argumentos = parser.parse_args()
    elegidos = list(PIPELINES) if argumentos.pipeline == "ambos" else [
        argumentos.pipeline]
    asyncio.run(main(elegidos))
