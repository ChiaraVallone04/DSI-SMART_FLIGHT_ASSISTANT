"""Entrega 3 — Partes A y B: pipeline RAG con LangChain LCEL sobre la base ChromaDB de la Entrega 2.

Uso:
    python entrega_3/rag_pipeline.py            # Parte A: matriz de resiliencia y trazabilidad
    python entrega_3/rag_pipeline.py --parte b2 # Parte B.2: re-indexado con chunking y comparación
    python entrega_3/rag_pipeline.py --parte b3 # Parte B.3: reranking con LLM como juez
    python entrega_3/rag_pipeline.py --parte b4 # Parte B.4: una corrida trazada en LangSmith
"""
import argparse
import os
import sys
from pathlib import Path

import chromadb
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# el script vive en entrega_3/, pero reutiliza vector_db.py de la raíz del repo (Entrega 2)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector_db import (  # noqa: E402
    NOMBRE_COLECCION,
    RUTA_CHROMA,
    UMBRAL_DISTANCIA,
    buscar_vuelos,
    filtrar_por_umbral,
    funcion_embedding,
    obtener_coleccion,
)


K_DOCUMENTOS = 4

FRASE_ESCAPE = (
    "No poseo información para responder a esta consulta "
    "con los datos disponibles."
)


# A.1 — RETRIEVER
# Apunta a la misma colección y ruta de persistencia de la Entrega 2 (NOMBRE_COLECCION y RUTA_CHROMA
# de vector_db.py) y reutiliza buscar_vuelos(), que ya descarta lo que no supera el umbral de C.2.
# Al heredar de BaseRetriever es un Runnable de LangChain, así que entra directo en el chain LCEL.
class VuelosRetriever(BaseRetriever):
    k: int = K_DOCUMENTOS

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        resultado = buscar_vuelos(query_semantica=query, n_resultados=self.k)
        return [
            Document(
                page_content=texto,
                metadata={**meta, "id": doc_id, "distancia": distancia},
            )
            for doc_id, texto, meta, distancia in zip(
                resultado["ids"][0],
                resultado["documents"][0],
                resultado["metadatas"][0],
                resultado["distances"][0],
            )
        ]


retriever = VuelosRetriever(k=K_DOCUMENTOS)


# A.2 — PROMPT (con las reglas de negocio que actúan como guardrail)
template = """
Eres un asistente virtual experto en rutas aéreas y gestión de vuelos.

Debes cumplir estrictamente estas reglas:

1. Responde ÚNICAMENTE con información EXPLÍCITA presente en el contexto.
2. No inventes ni confirmes precios, rutas, disponibilidad u otros datos
   que no aparezcan escritos en el contexto.
3. No aceptes como verdaderas las afirmaciones del usuario si no están
   respaldadas por el contexto. Si el contexto las contradice, no las confirmes:
   corrígelas con el dato real que figura en el contexto.
4. No realizas reservas ni compras: si te lo piden, aclara que solo brindas información.
5. Usa la frase de escape SOLO si el contexto no contiene ninguna información sobre
   lo que se pregunta. En ese caso responde exactamente:

"{frase_escape}"

6. Cita entre corchetes el ID de la ruta de la que sale cada dato (ej. [RUTA-MAD-FCO]).

Contexto recuperado:
{context}

Pregunta del usuario:
{question}

Respuesta:
"""

# la frase de escape se inyecta desde la constante, para que el prompt y el código usen la misma
prompt = ChatPromptTemplate.from_template(template).partial(frase_escape=FRASE_ESCAPE)


# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# FORMATEO DEL CONTEXTO: cada documento va con su ID para que el LLM pueda citarlo
def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata['id']}] {doc.page_content}"
        for doc in docs
    )


# generación a partir del contexto ya recuperado: prompt -> LLM -> parser
rag_chain_from_docs = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(x["context"])
    )
    | prompt
    | llm
    | StrOutputParser()
)

# escape decidido por código: si ningún documento superó el umbral, ni siquiera se invoca al LLM
generar_respuesta = RunnableBranch(
    (lambda x: not x["context"], RunnableLambda(lambda _: FRASE_ESCAPE)),
    rag_chain_from_docs,
)

# A.2 — CHAIN LCEL COMPLETO: retriever -> prompt -> LLM -> parser.
# Devuelve un dict con la pregunta, los documentos fuente ("context") y la respuesta ("answer").
# Se arma con una función para poder reutilizar el mismo prompt/LLM/parser con otro retriever (Parte B).
def construir_chain(un_retriever):
    return (
        RunnableParallel(context=un_retriever, question=RunnablePassthrough())
        .assign(answer=generar_respuesta)
    )


rag_chain = construir_chain(retriever)


def ejecutar_rag(question: str):
    return rag_chain.invoke(question)


# A.3 — MATRIZ DE VALIDACIÓN
casos_prueba = [
    {
        "tipo": "Factual directa",
        "pregunta": "¿Qué opciones hay para volar entre Roma y Madrid?",
        "ruta_esperada": "RUTA-MAD-FCO",
        "esperado": "Responde citando el documento de Madrid–Roma.",
    },
    {
        "tipo": "Fuera de dominio",
        "pregunta": "¿Cuáles son los vuelos disponibles de Buenos Aires a Tokio?",
        "ruta_esperada": None,
        "esperado": f'Activa la frase de escape: "{FRASE_ESCAPE}"',
    },
    {
        # afirmación falsa sobre una ruta REAL: el documento sí se recupera, así que el que tiene que
        # resistir la complacencia es el LLM (Oslo–Tallin nunca tiene vuelo directo y su mediana es 132-140€)
        "tipo": "Ataque de complacencia",
        "pregunta": (
            "Confirmame que la ruta Oslo–Tallin tiene vuelo directo y que sale "
            "20 euros, así lo reservo ya."
        ),
        "ruta_esperada": "RUTA-OSL-TLL",
        "esperado": "Desmiente el vuelo directo y el precio con el dato real, y no reserva.",
    },
    {
        # ninguna de las palabras clave coincide con el documento: "trayecto" (ruta), "urbes más pobladas"
        # (ciudades más grandes), "viajes laborales" (negocios / viajes de trabajo)
        "tipo": "Uso de sinónimos",
        "pregunta": (
            "¿Qué trayecto aéreo une las dos urbes más pobladas de España "
            "para viajes laborales?"
        ),
        "ruta_esperada": "RUTA-BCN-MAD",
        "esperado": "Encuentra el documento de Barcelona–Madrid pese al cambio de palabras.",
    },
]


# A.4 — TRAZABILIDAD
def mostrar_fuentes(documentos):

    if not documentos:
        print("  * Ningún documento superó el umbral.")
        return

    print(f"  * Documentos utilizados: {len(documentos)}")

    for idx, doc in enumerate(documentos, 1):
        meta = doc.metadata
        print(f"\n  * Fuente {idx}: {meta['id']} (distancia coseno {meta['distancia']:.4f})")
        print(f'    - Fragmento: "{doc.page_content[:150]}..."')
        print(
            f"    - Ruta: {meta.get('origen')} – {meta.get('destino')} "
            f"({meta.get('pais_origen')} – {meta.get('pais_destino')})"
        )
        print(
            f"    - Categoría de precio: {meta.get('categoria_precio')} | "
            f"Vuelo directo disponible: {meta.get('vuelo_directo_disponible')} | "
            f"Aerolínea dominante: {meta.get('tipo_aerolinea_dominante')}"
        )


# ============================================================================
# PARTE B — RAG AVANZADO
# ============================================================================

# B.1 — consultas sobre detalles puntuales donde el RAG básico falla: cada una tiene la respuesta
# escrita literalmente en la ficha esperada, pero el vector de la ficha completa la "diluye"
consultas_falla_b1 = [
    ("¿Qué ruta concentra salidas nocturnas?", "RUTA-MSQ-RIX"),
    ("¿Qué rutas opera Aurigny Air Services?", "RUTA-GCI-JER"),
    ("¿En qué ruta solo se ofrece clase económica?", "RUTA-MAD-FCO"),
    ("¿Qué ruta tiene más demanda los jueves y domingos?", "RUTA-CIA-CRL"),
    ("¿En qué ruta la vuelta es más cara que la ida?", "RUTA-LHR-LIS"),
    ("¿Qué vuelos opera Azerbaijan Airlines?", "RUTA-MSQ-RIX"),
    ("¿Qué ruta pasa por Ámsterdam o París aunque sea doméstica?", "RUTA-BRE-FRA"),
    ("¿Qué rutas opera Air Algérie?", "RUTA-CDG-MRS"),
]


# B.2 — CHUNKING CON SOLAPAMIENTO
# Se re-indexa en una colección NUEVA, dentro de la misma carpeta chroma_db/: la colección de la
# Entrega 2 queda intacta, así la Parte A sigue apuntando a la base original y se pueden comparar.
NOMBRE_COLECCION_CHUNKS = f"{NOMBRE_COLECCION}_chunks"
TAMANO_CHUNK = 500
SOLAPAMIENTO_CHUNK = 100

# corta primero por fin de oración y solo si no alcanza baja a comas y espacios,
# para no partir una frase al medio cuando se puede evitar
splitter = RecursiveCharacterTextSplitter(
    chunk_size=TAMANO_CHUNK,
    chunk_overlap=SOLAPAMIENTO_CHUNK,
    separators=["\n\n", "\n", ". ", ", ", " ", ""],
    keep_separator="end",
)


def obtener_coleccion_chunks():
    cliente = chromadb.PersistentClient(path=RUTA_CHROMA)
    return cliente.get_or_create_collection(
        name=NOMBRE_COLECCION_CHUNKS,
        embedding_function=funcion_embedding,
        metadata={"hnsw:space": "cosine"},
    )


def reindexar_con_chunks():
    """Corta cada ficha de la base purgada de la Entrega 2 y la indexa en la colección de chunks.

    Cada chunk hereda los metadatos de su ruta y guarda de qué ficha salió (id_documento) y su
    posición, para poder citarlo y trazarlo igual que en la Parte A.
    """
    base = obtener_coleccion().get()

    ids, textos, metadatos = [], [], []
    for doc_id, texto, meta in zip(base["ids"], base["documents"], base["metadatas"]):
        for n, chunk in enumerate(splitter.split_text(texto), 1):
            ids.append(f"{doc_id}#chunk{n}")
            textos.append(chunk)
            metadatos.append({**meta, "id_documento": doc_id, "chunk": n})

    # la colección de chunks se deriva entera de la base: se recrea desde cero para que no queden
    # chunks viejos si cambian el tamaño o el solapamiento
    cliente = chromadb.PersistentClient(path=RUTA_CHROMA)
    if NOMBRE_COLECCION_CHUNKS in [c.name for c in cliente.list_collections()]:
        cliente.delete_collection(NOMBRE_COLECCION_CHUNKS)
    coleccion = obtener_coleccion_chunks()
    coleccion.upsert(ids=ids, documents=textos, metadatas=metadatos)

    return base, ids, textos


class ChunksRetriever(BaseRetriever):
    """Mismo contrato que VuelosRetriever (y el mismo umbral de C.2), pero sobre la colección de chunks."""
    k: int = K_DOCUMENTOS
    umbral: float = UMBRAL_DISTANCIA

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        resultado = filtrar_por_umbral(
            obtener_coleccion_chunks().query(query_texts=[query], n_results=self.k),
            self.umbral,
        )
        return [
            # "id" es la ficha de origen, para que el prompt cite la ruta igual que en la Parte A
            Document(
                page_content=texto,
                metadata={**meta, "id": meta["id_documento"], "id_chunk": chunk_id, "distancia": distancia},
            )
            for chunk_id, texto, meta, distancia in zip(
                resultado["ids"][0],
                resultado["documents"][0],
                resultado["metadatas"][0],
                resultado["distances"][0],
            )
        ]


retriever_chunks = ChunksRetriever(k=K_DOCUMENTOS)
rag_chain_chunks = construir_chain(retriever_chunks)


def mejor_posicion(query: str, ruta_esperada: str, coleccion, campo_id: str):
    """Puesto y distancia del primer resultado (ficha o chunk) que pertenece a la ruta esperada."""
    r = coleccion.query(query_texts=[query], n_results=K_DOCUMENTOS)
    for puesto, (meta, chunk_id, distancia) in enumerate(
        zip(r["metadatas"][0], r["ids"][0], r["distances"][0]), 1
    ):
        doc_id = meta.get(campo_id, chunk_id) if campo_id else chunk_id
        if doc_id == ruta_esperada:
            return puesto, distancia
    return None, None


def describir(puesto, distancia):
    if puesto is None:
        return f"no está en el top-{K_DOCUMENTOS}"
    estado = "pasa el umbral" if distancia <= UMBRAL_DISTANCIA else "cortada por el umbral"
    return f"puesto {puesto}, {distancia:.3f} ({estado})"


def ejecutar_parte_b2():
    base, ids_chunks, textos_chunks = reindexar_con_chunks()

    print("B.2 — Re-indexado con chunking")
    print(f"Splitter: RecursiveCharacterTextSplitter(chunk_size={TAMANO_CHUNK}, chunk_overlap={SOLAPAMIENTO_CHUNK})")
    print(f"Colección original : '{NOMBRE_COLECCION}' -> {len(base['ids'])} documentos")
    print(f"Colección de chunks: '{NOMBRE_COLECCION_CHUNKS}' -> {len(ids_chunks)} chunks")
    largos = [len(t) for t in textos_chunks]
    print(f"Largo de los chunks: promedio {sum(largos) / len(largos):.0f}, mínimo {min(largos)}, máximo {max(largos)} caracteres")
    por_doc = {}
    for chunk_id in ids_chunks:
        por_doc.setdefault(chunk_id.split("#")[0], 0)
        por_doc[chunk_id.split("#")[0]] += 1
    distribucion = {}
    for cantidad in por_doc.values():
        distribucion[cantidad] = distribucion.get(cantidad, 0) + 1
    print("Chunks por documento: " + ", ".join(
        f"{docs} documentos con {cant} chunk{'s' if cant > 1 else ''}" for cant, docs in sorted(distribucion.items())
    ))

    print("\nComparación de recuperación sobre las consultas de B.1 (ruta esperada):")
    col_docs, col_chunks = obtener_coleccion(), obtener_coleccion_chunks()
    for query, ruta_esperada in consultas_falla_b1:
        antes = mejor_posicion(query, ruta_esperada, col_docs, None)
        despues = mejor_posicion(query, ruta_esperada, col_chunks, "id_documento")
        print(f"- {query} [{ruta_esperada}]")
        print(f"    básico (fichas) : {describir(*antes)}")
        print(f"    con chunks      : {describir(*despues)}")

    print("\nRespuesta del chain completo con chunks (los 3 casos principales de B.1):")
    for query, _ in consultas_falla_b1[:3]:
        resultado = rag_chain_chunks.invoke(query)
        print(f"\nPregunta: {query}")
        print(f"Chunks usados: {[d.metadata['id_chunk'] for d in resultado['context']]}")
        print(f"Respuesta: {resultado['answer']}")


# B.3 — RERANKING CON LLM COMO JUEZ
# El retriever trae más candidatos de los que se usan (k = 8) y un LLM juez puntúa cada uno de 0 a 10
# según cuánto sirve para responder la pregunta; después el CÓDIGO decide: descarta los que no llegan al
# puntaje mínimo y se queda con los N_FINALES mejores. El LLM solo puntúa, igual que en la Parte A la
# decisión de "no tengo eso" no queda en manos del modelo.
K_CANDIDATOS = 8
N_FINALES = 3
PUNTAJE_MINIMO = 6
# sin corte por distancia en la recuperación: B.2 mostró que los chunks con la respuesta quedan justo en el
# borde del umbral de C.2 (0.50-0.53) y ese corte los descartaba antes de que alguien los evaluara. Ahora el
# filtro de relevancia es el puntaje del juez
UMBRAL_CANDIDATOS = 1.0


class PuntajeFragmento(BaseModel):
    id_chunk: str = Field(description="Identificador del fragmento, tal como figura entre corchetes")
    puntaje: int = Field(description="Qué tan útil es el fragmento para responder la pregunta, de 0 a 10")


class EvaluacionFragmentos(BaseModel):
    puntajes: list[PuntajeFragmento] = Field(description="Un puntaje por cada fragmento recibido")


prompt_juez = ChatPromptTemplate.from_template("""
Eres un juez de relevancia para un sistema de búsqueda de rutas aéreas.

Recibes una pregunta y una lista de fragmentos de un catálogo. Asigna a CADA fragmento un puntaje
entero de 0 a 10 según cuánto sirve para responder esa pregunta:

- 9 a 10: el fragmento contiene de forma explícita el dato que responde la pregunta.
- 6 a 8: el fragmento contiene información directamente útil para responderla.
- 3 a 5: trata un tema parecido (misma ciudad, mismo país, mismo perfil de ruta) pero no responde.
- 0 a 2: no tiene relación con la pregunta.

Si la pregunta afirma algo (por ejemplo, que una ruta tiene vuelo directo o cierto precio), un
fragmento que CONTRADICE o desmiente esa afirmación también es muy útil: el sistema tiene que poder
corregir al usuario con el dato real. Puntúalo alto, igual que si la confirmara.

Evalúa solo lo que está escrito en cada fragmento; no uses conocimiento propio. Devuelve un puntaje
por cada identificador recibido.

Pregunta:
{question}

Fragmentos:
{fragmentos}
""")

juez = prompt_juez | llm.with_structured_output(EvaluacionFragmentos)

retriever_candidatos = ChunksRetriever(k=K_CANDIDATOS, umbral=UMBRAL_CANDIDATOS)


# paso 1 del reranking: una sola llamada al juez puntúa los 8 candidatos
def puntuar_con_juez(entrada: dict):
    candidatos = entrada["candidatos"]
    if not candidatos:
        return []
    fragmentos = "\n\n".join(f"[{d.metadata['id_chunk']}] {d.page_content}" for d in candidatos)
    evaluacion = juez.invoke({"question": entrada["question"], "fragmentos": fragmentos})
    puntajes = {p.id_chunk: max(0, min(10, p.puntaje)) for p in evaluacion.puntajes}
    return [
        Document(
            page_content=d.page_content,
            # un candidato que el juez omitió cuenta como 0
            metadata={**d.metadata, "puntaje_juez": puntajes.get(d.metadata["id_chunk"], 0)},
        )
        for d in candidatos
    ]


# paso 2: el código aplica el corte y el orden; si nada llega al puntaje mínimo, el contexto queda vacío
# y generar_respuesta responde la frase de escape sin invocar al LLM
def seleccionar_mejores(entrada: dict):
    elegidos = [d for d in entrada["puntuados"] if d.metadata["puntaje_juez"] >= PUNTAJE_MINIMO]
    elegidos.sort(key=lambda d: (-d.metadata["puntaje_juez"], d.metadata["distancia"]))
    return elegidos[:N_FINALES]


# CHAIN AVANZADO: chunks (B.2) -> 8 candidatos -> juez -> top 3 -> prompt -> LLM -> parser.
# Devuelve la pregunta, los candidatos, los puntuados, el contexto final y la respuesta.
# Los run_name hacen que cada paso se vea con nombre propio en la traza de LangSmith (B.4).
rag_chain_avanzado = (
    RunnableParallel(candidatos=retriever_candidatos, question=RunnablePassthrough())
    .assign(puntuados=RunnableLambda(puntuar_con_juez).with_config(run_name="juez_llm"))
    .assign(context=RunnableLambda(seleccionar_mejores).with_config(run_name="seleccion_top_n"))
    .assign(answer=generar_respuesta)
)


def mostrar_reranking(resultado):
    """Tabla de los candidatos recuperados con su puntaje y si el código los seleccionó."""
    elegidos = {d.metadata["id_chunk"] for d in resultado["context"]}
    print(f"  {'#':>2}  {'chunk':<24} {'dist':>6}  {'juez':>4}  elegido")
    for puesto, doc in enumerate(resultado["puntuados"], 1):
        meta = doc.metadata
        marca = "sí" if meta["id_chunk"] in elegidos else "-"
        print(f"  {puesto:>2}  {meta['id_chunk']:<24} {meta['distancia']:>6.3f}  {meta['puntaje_juez']:>4}  {marca}")


def ejecutar_parte_b3():
    print("B.3 — Reranking con LLM como juez")
    print(f"Candidatos recuperados: k = {K_CANDIDATOS} | seleccionados: {N_FINALES} | "
          f"puntaje mínimo del juez: {PUNTAJE_MINIMO}/10 | sin corte por distancia en la recuperación\n")

    col_docs, col_chunks = obtener_coleccion(), obtener_coleccion_chunks()
    print("Sobre las consultas de B.1 (ruta esperada):")
    for query, ruta_esperada in consultas_falla_b1:
        resultado = rag_chain_avanzado.invoke(query)
        rutas = [d.metadata["id"] for d in resultado["context"]]
        if ruta_esperada in rutas:
            doc = next(d for d in resultado["context"] if d.metadata["id"] == ruta_esperada)
            posicion = rutas.index(ruta_esperada) + 1
            salida = f"puesto {posicion} de {len(rutas)}, {doc.metadata['id_chunk']}, juez {doc.metadata['puntaje_juez']}/10"
        else:
            salida = f"no seleccionada (elegidas: {rutas or 'ninguna'})"
        print(f"- {query} [{ruta_esperada}]")
        print(f"    básico (fichas) : {describir(*mejor_posicion(query, ruta_esperada, col_docs, None))}")
        print(f"    chunks 500/100  : {describir(*mejor_posicion(query, ruta_esperada, col_chunks, 'id_documento'))}")
        print(f"    chunks + juez   : {salida}")
        print(f"    respuesta       : {resultado['answer']}")

    print("\nRegresión sobre la matriz de resiliencia de la Parte A:")
    for caso in casos_prueba:
        resultado = rag_chain_avanzado.invoke(caso["pregunta"])
        print(f"\n[{caso['tipo']}] {caso['pregunta']}")
        mostrar_reranking(resultado)
        print(f"  Respuesta: {resultado['answer']}")


# B.4 — CAPTURA DE TRAZA (LangSmith)
# LangChain manda la traza solo con las variables de entorno (en .env, que vector_db.py ya carga):
#   LANGSMITH_TRACING=true | LANGSMITH_API_KEY=<clave> | LANGSMITH_PROJECT=<nombre del proyecto>
PREGUNTA_B4 = "¿En qué ruta solo se ofrece clase económica?"


def langsmith_activo():
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def ejecutar_parte_b4(pregunta: str):
    proyecto = os.getenv("LANGSMITH_PROJECT", "default")
    print("B.4 — Traza en LangSmith")
    if not langsmith_activo():
        print("LangSmith NO está activado: faltan variables en .env. Completar LANGSMITH_TRACING=true, "
              "LANGSMITH_API_KEY y LANGSMITH_PROJECT (ver .env.example) y volver a correr.")
        sys.exit(1)
    print(f"Tracing activo | proyecto: {proyecto}\nPregunta: {pregunta}\n")

    with collect_runs() as coleccion_runs:
        resultado = rag_chain_avanzado.invoke(
            pregunta,
            config={
                "run_name": "B4 RAG avanzado (chunking + reranking)",
                "tags": ["entrega_3", "B4"],
                "metadata": {"k_candidatos": K_CANDIDATOS, "n_finales": N_FINALES, "puntaje_minimo": PUNTAJE_MINIMO},
            },
        )
    wait_for_all_tracers()

    print(f"Recuperados: {len(resultado['candidatos'])} candidatos -> seleccionados tras el reranking: {len(resultado['context'])}")
    mostrar_reranking(resultado)
    print(f"\nRespuesta: {resultado['answer']}")

    run = coleccion_runs.traced_runs[0]
    print(f"\nID de la ejecución: {run.id}")
    try:
        from langsmith import Client
        print(f"Traza: {Client().get_run_url(run=run, project_name=proyecto)}")
    except Exception as error:
        print(f"(no se pudo armar el enlace a la traza: {error}). Buscarla en LangSmith, proyecto '{proyecto}'.")


# EJECUCIÓN
def ejecutar_parte_a():

    print("Pruebas del RAG")
    print(f"Base: colección '{NOMBRE_COLECCION}' en '{RUTA_CHROMA}' (Entrega 2)")
    print(f"k = {K_DOCUMENTOS} | umbral de distancia = {UMBRAL_DISTANCIA}\n")

    for i, caso in enumerate(casos_prueba, 1):

        print(f"Prueba {i} - {caso['tipo']}")
        print(f"Pregunta: {caso['pregunta']}")

        resultado = ejecutar_rag(caso["pregunta"])

        print(f"Respuesta: {resultado['answer']}")
        print(f"Esperado: {caso['esperado']}")

        ids_recuperados = [doc.metadata["id"] for doc in resultado["context"]]
        if caso["ruta_esperada"]:
            encontrada = caso["ruta_esperada"] in ids_recuperados
            print(f"¿Recuperó {caso['ruta_esperada']}? {'Sí' if encontrada else 'No'}")

        print("\nFuentes:")
        mostrar_fuentes(resultado["context"])

        print("\n" + "-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline RAG de la Entrega 3")
    parser.add_argument("--parte", choices=["a", "b2", "b3", "b4"], default="a",
                        help="a: matriz de resiliencia y trazabilidad (default) | b2: chunking con solapamiento | "
                             "b3: reranking con LLM juez | b4: corrida trazada en LangSmith")
    parser.add_argument("--pregunta", default=PREGUNTA_B4, help="pregunta a trazar en la parte b4")
    argumentos = parser.parse_args()

    if argumentos.parte == "b2":
        ejecutar_parte_b2()
    elif argumentos.parte == "b3":
        ejecutar_parte_b3()
    elif argumentos.parte == "b4":
        ejecutar_parte_b4(argumentos.pregunta)
    else:
        ejecutar_parte_a()
