"""B.1 — Migración a ChromaDB: colección persistente con la misma base de A.3."""
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


if __name__ == "__main__":
    documentos = cargar_base_conocimiento()
    coleccion = obtener_coleccion()
    cargar_coleccion(coleccion, documentos)
