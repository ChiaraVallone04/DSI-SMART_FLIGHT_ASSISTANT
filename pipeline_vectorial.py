"""A.4 — Índice FAISS: embeddings de base_conocimiento.json, persistencia y búsqueda semántica top-K."""
import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY no encontrada en el entorno.")

MODELO_EMBEDDING = "text-embedding-3-small"
RUTA_BASE_CONOCIMIENTO = "base_conocimiento.json"
DIR_INDICE = "faiss_index"
RUTA_INDICE = os.path.join(DIR_INDICE, "index.faiss")
RUTA_IDS = os.path.join(DIR_INDICE, "ids.json")

client = OpenAI(api_key=OPENAI_API_KEY)

CONSULTAS_PRUEBA = [
    "quiero una escapada barata y directa a Italia",
    "busco un vuelo de lujo, sin escalas, no me importa el precio",
    "ruta entre Alemania y algún país báltico",
]
TOP_K = 3


def cargar_base_conocimiento(ruta: str = RUTA_BASE_CONOCIMIENTO) -> list[dict]:
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def generar_embeddings(textos: list[str]) -> np.ndarray:
    respuesta = client.embeddings.create(model=MODELO_EMBEDDING, input=textos)
    vectores = np.array([item.embedding for item in respuesta.data], dtype="float32")
    faiss.normalize_L2(vectores)
    return vectores


def construir_indice(documentos: list[dict]) -> tuple[faiss.Index, list[str]]:
    textos = [doc["descripcion_semantica"] for doc in documentos]
    vectores = generar_embeddings(textos)

    indice = faiss.IndexFlatIP(vectores.shape[1])
    indice.add(vectores)

    os.makedirs(DIR_INDICE, exist_ok=True)
    faiss.write_index(indice, RUTA_INDICE)
    ids = [doc["id"] for doc in documentos]
    with open(RUTA_IDS, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    return indice, ids


def cargar_o_construir_indice(documentos: list[dict]) -> tuple[faiss.Index, list[str]]:
    ids_actuales = [doc["id"] for doc in documentos]

    if os.path.exists(RUTA_INDICE) and os.path.exists(RUTA_IDS):
        with open(RUTA_IDS, encoding="utf-8") as f:
            ids_persistidos = json.load(f)
        if ids_persistidos == ids_actuales:
            print(f"[disco] Índice recargado desde '{RUTA_INDICE}' — sin llamadas a la API de embeddings.")
            return faiss.read_index(RUTA_INDICE), ids_persistidos
        print("[aviso] base_conocimiento.json cambió respecto al índice persistido — se regenera.")

    print(f"[api] No hay índice en disco — generando embeddings para {len(documentos)} documentos...")
    return construir_indice(documentos)


def buscar(indice: faiss.Index, ids: list[str], documentos: list[dict], consulta: str, k: int = TOP_K):
    vector_consulta = generar_embeddings([consulta])
    similitudes, posiciones = indice.search(vector_consulta, k)

    por_id = {doc["id"]: doc for doc in documentos}
    resultados = []
    for similitud, pos in zip(similitudes[0], posiciones[0]):
        if pos == -1:
            continue
        doc_id = ids[pos]
        resultados.append((doc_id, float(similitud), por_id[doc_id]["descripcion_semantica"][:100]))
    return resultados


if __name__ == "__main__":
    documentos = cargar_base_conocimiento()
    indice, ids = cargar_o_construir_indice(documentos)

    for consulta in CONSULTAS_PRUEBA:
        print(f"\nConsulta: \"{consulta}\"")
        for doc_id, similitud, extracto in buscar(indice, ids, documentos, consulta):
            print(f"  [{similitud:.4f}] {doc_id} — {extracto}...")
