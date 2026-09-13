import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# configuración de clientes
api_key = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=api_key)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key, model_name="text-embedding-3-small"
)

# cargar colección de ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
col = chroma_client.get_collection(
    name="vuelos_smart_flight_assistant", embedding_function=openai_ef
)


def ejecutar_rag(query, where_filter=None, n_results=3):
    print(f"\n==========================================")
    print(f"QUERY: '{query}'")
    if where_filter:
        print(f"FILTRO METADATOS: {where_filter}")

    # recuperación semántica
    results = col.query(
        query_texts=[query], n_results=n_results, where=where_filter
    )

    documentos_recuperados = results["documents"][0]
    ids_recuperados = results["ids"][0]

    print(f"\n--> IDs Recuperados: {ids_recuperados}")

    if not documentos_recuperados:
        contexto = "No se encontraron rutas relevantes en el catálogo."
    else:
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

# Test 2: Búsqueda con filtro estricto por metadato (limitado a 1 resultado)
ejecutar_rag(
    "Busco un vuelo directo para ir de Madrid a Roma",
    where_filter={"vuelo_directo_disponible": True},
    n_results=1,
)

# Test 3: Consulta fuera de catálogo
ejecutar_rag("¿Qué vuelos tienen disponibles para ir desde Buenos Aires a Tokio?")