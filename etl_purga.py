import copy
import json
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import numpy as np

load_dotenv(override=True)

# carga de datos sucios
with open("base_conocimiento.json", "r", encoding="utf-8") as f:
    datos_sucios = json.load(f)

print(f"Total registros cargados iniciales: {len(datos_sucios)}")

# etl tradicional (normalización de claves y tipos)
datos_normalizados = []
ids_vistos = set()

for item_raw in datos_sucios:
    # se usa deepcopy para no alterar el objeto original por referencia en memoria
    item = copy.deepcopy(item_raw)

    # acceder al sub-diccionario de metadatos
    meta = item.get("metadatos", {})

    # normalizar clave mal nombrada:
    # agarra exactamente el valor que venga en is_direct (como el booleano True), 
    # lo asigna a la clave correcta vuelo_directo_disponible 
    # y elimina la clave vieja para que el diccionario quede limpio.
    if "is_direct" in meta:
        meta["vuelo_directo_disponible"] = meta.pop("is_direct")
    if "vuelo_directo" in meta:
        meta["vuelo_directo_disponible"] = meta.pop("vuelo_directo")

    # normalizar booleano cargado como string
    val_bool = meta.get("vuelo_directo_disponible")
    if isinstance(val_bool, str):
        meta["vuelo_directo_disponible"] = val_bool.strip().lower() in [
            "true",
            "1",
            "si",
            "sí",
        ]

    # resolver colisión de ids (si hay ids duplicados con distinto contenido)
    item_id = item["id"]
    if item_id in ids_vistos:
        item_id = f"{item_id}_dup"
        item["id"] = item_id
    ids_vistos.add(item_id)

    item["metadatos"] = meta
    datos_normalizados.append(item)

print(f"Registros tras normalización ETL: {len(datos_normalizados)}")

# purga semántica mediante vectorización y distancia coseno
api_key = os.getenv("OPENAI_API_KEY")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key, model_name="text-embedding-3-small"
)

# genera embeddings y asegura formato numpy array
textos = [d["descripcion_semantica"] for d in datos_normalizados]
embeddings = np.array(openai_ef(textos), dtype=np.float32)

# umbral de distancia coseno (1 - similitud)
# distancias por debajo de 0.20 indican casi-duplicados semánticos
UMBRAL_DISTANCIA = 0.20

indices_a_eliminar = set()
pares_detectados = []

for i in range(len(embeddings)):
    if i in indices_a_eliminar:
        continue
    for j in range(i + 1, len(embeddings)):
        if j in indices_a_eliminar:
            continue

        # calcular distancia coseno entre vectores norma-1
        vec1 = embeddings[i]
        vec2 = embeddings[j]
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            continue

        similitud = np.dot(vec1, vec2) / (norm1 * norm2)
        distancia = 1.0 - similitud

        if distancia < UMBRAL_DISTANCIA:
            indices_a_eliminar.add(j)
            pares_detectados.append(
                {
                    "conservado": datos_normalizados[i]["id"],
                    "eliminado": datos_normalizados[j]["id"],
                    "distancia": round(float(distancia), 4),
                    "texto_conservado": datos_normalizados[i][
                        "descripcion_semantica"
                    ],
                    "texto_eliminado": datos_normalizados[j][
                        "descripcion_semantica"
                    ],
                }
            )

datos_purgados = [
    d
    for idx, d in enumerate(datos_normalizados)
    if idx not in indices_a_eliminar
]

print(f"\nSe eliminaron {len(indices_a_eliminar)} casi-duplicados semánticos.")
print(f"Total registros finales purgados: {len(datos_purgados)}")

for par in pares_detectados:
    print(f"\n[PURGA DETECTADA - Distancia: {par['distancia']}]")
    print(
        f"  - Mantener ({par['conservado']}): {par['texto_conservado'][:100]}..."
    )
    print(
        f"  - Eliminar ({par['eliminado']}): {par['texto_eliminado'][:100]}..."
    )

# indexación automática en ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

col = chroma_client.get_or_create_collection(
    name="vuelos_smart_flight_assistant",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},
)

# eliminar de la colección los IDs detectados como casi-duplicados
ids_eliminados = [datos_normalizados[i]["id"] for i in indices_a_eliminar]
if ids_eliminados:
    col.delete(ids=ids_eliminados)

# upsert de los registros purgados (permite re-ejecutar sin duplicar)
col.upsert(
    documents=[d["descripcion_semantica"] for d in datos_purgados],
    metadatas=[d["metadatos"] for d in datos_purgados],
    ids=[d["id"] for d in datos_purgados],
)

print(
    f"\n¡ChromaDB actualizada con éxito! Total indexados: {len(datos_purgados)}"
)