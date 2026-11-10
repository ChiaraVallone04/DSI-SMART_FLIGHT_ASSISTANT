"""Construye desde cero la base vectorial que usa el agente (data/chroma_db/ no se versiona: se regenera acá).

Reproduce en orden los pasos de las entregas anteriores:
  1. Entrega 2 (B.1): carga data/base_conocimiento.json en la colección ChromaDB con similitud coseno.
  2. Entrega 2 (B.5): ETL de purga (normaliza, resuelve la colisión de IDs, elimina casi-duplicados) -> 21 rutas.
  3. Entrega 3 (B.2): corta las fichas en chunks de 500/100 en la colección que usa el RAG avanzado -> 40 chunks.

Uso (con OPENAI_API_KEY en el .env):
    python main.py ingest
Se puede volver a correr sin duplicar nada (upsert y recreación de la colección de chunks).
"""
import runpy

from src.config import RAIZ_REPO


def construir_base():
    from vector_db import cargar_base_conocimiento, cargar_coleccion, obtener_coleccion  # Entrega 2

    print("== 1/3 Carga de la base de conocimiento (Entrega 2, B.1) ==")
    cargar_coleccion(obtener_coleccion(), cargar_base_conocimiento())

    print("\n== 2/3 ETL de purga (Entrega 2, B.5) ==")
    runpy.run_path(str(RAIZ_REPO / "entrega_2" / "etl_purga.py"), run_name="__main__")

    print("\n== 3/3 Colección de chunks para el RAG avanzado (Entrega 3, B.2) ==")
    # se importa recién ahora: rag_pipeline exige que exista la base del paso 2
    from rag_pipeline import reindexar_con_chunks  # Entrega 3

    base, ids_chunks, _ = reindexar_con_chunks()
    print(f"{len(base['ids'])} rutas -> {len(ids_chunks)} chunks indexados.")
    print("\nBase lista. Ya se puede levantar la API: python main.py serve")
