import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# carga variables de entorno (OPENAI_API_KEY)
load_dotenv(override=True)

def ejecutar_evento_en_caliente():
    """
    Simula una actualización operativa en tiempo real para la ruta KEF-MAD
    utilizando el método idempotente 'upsert' de ChromaDB.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY no encontrada en el archivo .env")

    # configura función de embeddings y cliente de ChromaDB
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    
    client = chromadb.PersistentClient(path="chroma_db")
    
    # obtiene o crea la colección del Smart Flight Assistant con distancia coseno
    coleccion = client.get_or_create_collection(
        name="vuelos_smart_flight_assistant",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

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

if __name__ == "__main__":
    ejecutar_evento_en_caliente()