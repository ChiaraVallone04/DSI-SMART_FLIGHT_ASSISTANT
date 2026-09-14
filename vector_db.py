"""B.1–B.4 — ChromaDB persistente, evento en caliente y búsqueda híbrida sobre la base de A.3."""
import json
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY no encontrada en el entorno.")

MODELO_EMBEDDING = "text-embedding-3-small"
RUTA_BASE_CONOCIMIENTO = "base_conocimiento.json"
RUTA_CHROMA = "chroma_db"
NOMBRE_COLECCION = "vuelos_smart_flight_assistant"

funcion_embedding = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=MODELO_EMBEDDING,
)


# Lee los documentos de la Base de Conocimiento desde base_conocimiento.json
def cargar_base_conocimiento(ruta: str = RUTA_BASE_CONOCIMIENTO):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


# Abre (o crea) la colección persistente de ChromaDB con similitud coseno
def obtener_coleccion():
    cliente = chromadb.PersistentClient(path=RUTA_CHROMA)
    return cliente.get_or_create_collection(
        name=NOMBRE_COLECCION,
        embedding_function=funcion_embedding,
        metadata={"hnsw:space": "cosine"},
    )


# Carga los documentos en la colección con upsert, para poder re-ejecutar sin duplicar
def cargar_coleccion(coleccion: chromadb.Collection, documentos: list[dict]):
    # upsert en lugar de add: permite re-ejecutar el script sin duplicar registros
    coleccion.upsert(
        ids=[doc["id"] for doc in documentos],
        documents=[doc["descripcion_semantica"] for doc in documentos],
        metadatas=[doc["metadatos"] for doc in documentos],
    )
    print(
        f"Colección '{coleccion.name}' actualizada: {coleccion.count()} rutas indexadas.")


def ejecutar_evento_en_caliente():
    """
    Simula una actualización operativa en tiempo real para la ruta KEF-MAD
    utilizando el método idempotente 'upsert' de ChromaDB.
    """
    coleccion = obtener_coleccion()

    # define los datos del evento de negocio
    doc_id = "RUTA-KEF-MAD"
    nuevo_texto = (
        "Ruta directa inaugurada entre Reikiavik (KEF) y Madrid (MAD). "
        "Opción económica e ideal para turismo de auroras boreales y viajes nórdicos sin escalas."
    )
    nuevos_metadatos = {
        "origen": "KEF",
        "destino": "MAD",
        "pais_origen": "Islandia",
        "pais_destino": "España",
        "vuelo_directo_disponible": True,  # cambió de False a True
        "categoria_precio": "medio",       # cambió de premium a medio
        "tipo_aerolinea_dominante": "low-cost",
        "tags_regionales": ["auroras boreales", "islandia", "escapada nórdica", "directo"]
    }

    print(f"=== Ejecutando Evento en Caliente para ID: {doc_id} ===")

    # aplica actualización atómica con metodo upsert
    coleccion.upsert(
        ids=[doc_id],
        documents=[nuevo_texto],
        metadatas=[nuevos_metadatos]
    )
    print(" Actualización realizada con éxito en ChromaDB.\n")

    # verificación de lectura directa desde el disco (SQLite) -> metodo get
    resultado = coleccion.get(ids=[doc_id])

    print("=== Estado Verificado en ChromaDB ===")
    print("ID:", resultado["ids"][0])
    print("Documento:", resultado["documents"][0])
    print("Metadatos actualizados:", resultado["metadatas"][0])


def buscar_vuelos(query_semantica: str, categoria_precio: str = None, solo_directos: bool = False, n_resultados: int = 3):
    """
    Realiza una búsqueda híbrida en ChromaDB combinando similtud semántica
    con filtros nativos 'where' sin post-filtering manual.
    """
    coleccion = obtener_coleccion()

    # construcción nativa del filtro where
    condiciones = []

    if categoria_precio:
        condiciones.append({"categoria_precio": {"$eq": categoria_precio}})

    if solo_directos:
        condiciones.append({"vuelo_directo_disponible": {"$eq": True}})

    where_filter = None
    if len(condiciones) == 1:
        where_filter = condiciones[0]
    elif len(condiciones) > 1:
        where_filter = {"$and": condiciones}

    # query directa a la base vectorial
    resultados = coleccion.query(
        query_texts=[query_semantica],
        n_results=n_resultados,
        where=where_filter
    )

    return resultados


if __name__ == "__main__":
    # B.1 — carga inicial de la base de conocimiento
    documentos = cargar_base_conocimiento()
    coleccion = obtener_coleccion()
    cargar_coleccion(coleccion, documentos)

    # B.3 — evento de negocio en caliente
    ejecutar_evento_en_caliente()

    # B.4 — búsqueda híbrida
    print("TEST 1: Búsqueda Semántica + Vuelo Directo")
    res1 = buscar_vuelos(
        query_semantica="escapada para ver auroras boreales",
        solo_directos=True,
        n_resultados=2
    )
    for doc, meta in zip(res1["documents"][0], res1["metadatas"][0]):
        print(f"-> {doc}\n   Metadatos: {meta}\n")

    print("TEST 2: Búsqueda Semántica + Categoría de Precio")
    res2 = buscar_vuelos(
        query_semantica="vuelos económicos a Europa",
        categoria_precio="medio",
        n_resultados=2
    )
    for doc, meta in zip(res2["documents"][0], res2["metadatas"][0]):
        print(f"-> {doc}\n   Metadatos: {meta}\n")

    print("TEST 3: Ambos Filtros ($and NATIVO)")
    res3 = buscar_vuelos(
        query_semantica="turismo nórdico",
        categoria_precio="medio",
        solo_directos=True,
        n_resultados=2
    )
    for doc, meta in zip(res3["documents"][0], res3["metadatas"][0]):
        print(f"-> {doc}\n   Metadatos: {meta}\n")
