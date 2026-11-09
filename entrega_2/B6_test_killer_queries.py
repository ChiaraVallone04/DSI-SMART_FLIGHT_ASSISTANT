import os
from dotenv import load_dotenv
from openai import OpenAI

from vector_db import UMBRAL_DISTANCIA, filtrar_por_umbral, obtener_coleccion

load_dotenv(override=True)

# configuración de clientes
api_key = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=api_key)

# cargar colección de ChromaDB
col = obtener_coleccion()


def ejecutar_rag(query, where_filter=None, n_results=3):
    print(f"\n==========================================")
    print(f"QUERY: '{query}'")
    if where_filter:
        print(f"FILTRO METADATOS: {where_filter}")

    # recuperación semántica
    results = col.query(
        query_texts=[query], n_results=n_results, where=where_filter
    )

    # C.2 — umbral de aceptación: se descartan los vecinos que Chroma devuelve aunque estén lejos de la consulta
    print(
        f"\n--> Candidatos (id: distancia): {dict(zip(results['ids'][0], [round(d, 4) for d in results['distances'][0]]))}")
    results = filtrar_por_umbral(results, UMBRAL_DISTANCIA)

    documentos_recuperados = results["documents"][0]
    ids_recuperados = results["ids"][0]

    print(
        f"--> IDs Recuperados (distancia <= {UMBRAL_DISTANCIA}): {ids_recuperados}")

    # si nada supera el umbral no se llama al LLM: la respuesta "no tengo eso" la decide el código, no el prompt
    if not documentos_recuperados:
        print("\n--> Respuesta: No dispongo de esa información en el catálogo.")
        print("    (ningún resultado superó el umbral de similitud; no se invocó al LLM)")
        return

    contexto = "\n\n".join(documentos_recuperados)

    # generación con LLM
    system_prompt = (
        "Sos un asistente de vuelos. Respondé únicamente con la información proporcionada "
        "en el contexto. Si la respuesta no está en el contexto, indicá claramente que no "
        "disponés de esa información en el catálogo."
    )

    user_prompt = f"Contexto:\n{contexto}\n\nPregunta: {query}"

    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    respuesta_llm = response.choices[0].message.content
    print(f"\n--> Respuesta LLM:\n{respuesta_llm}")


# EJECUCIÓN DE LAS 3 KILLER QUERIES


# Test 1: Búsqueda puramente semántica (jerga / sin palabras clave)
ejecutar_rag(
    "Quiero pegarme una escapada barata en el puente aéreo para laburar en el día"
)

# Test 2a: la misma consulta SIN filtro duro (la semántica cruda: el "desastre")
ejecutar_rag(
    "ruta directa a Tallin",
    n_results=1,
)

# Test 2b: la misma consulta CON filtro duro por metadato (el filtro lo bloquea)
ejecutar_rag(
    "ruta directa a Tallin",
    where_filter={"vuelo_directo_disponible": True},
    n_results=1,
)


# Test 3: Consulta fuera de catálogo
ejecutar_rag(
    "¿Qué vuelos tienen disponibles para ir desde Buenos Aires a Tokio?")
