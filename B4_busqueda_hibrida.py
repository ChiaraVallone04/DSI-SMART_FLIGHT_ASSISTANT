import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv(override=True)

def buscar_vuelos(query_semantica: str, categoria_precio: str = None, solo_directos: bool = False, n_resultados: int = 3):
    """
    Realiza una búsqueda híbrida en ChromaDB combinando similtud semántica 
    con filtros nativos 'where' sin post-filtering manual.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY no encontrada en el archivo .env")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    
    client = chromadb.PersistentClient(path="chroma_db")
    coleccion = client.get_collection(
        name="vuelos_smart_flight_assistant",
        embedding_function=openai_ef
    )

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