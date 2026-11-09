# Entrega 2 — Base de conocimiento vectorial

Embeddings, índice FAISS, base persistente en ChromaDB con búsqueda híbrida, umbral de aceptación y ETL de purga. El detalle narrativo (justificaciones, capturas, reflexiones) está en [`informe.md`](informe.md).

La fuente de verdad es [`../data/base_conocimiento.json`](../data/base_conocimiento.json). Los índices que se generan (`data/faiss_index/`, `data/chroma_db/`) no se versionan: se reconstruyen corriendo los pasos de abajo. La misma base la usan la Entrega 3 y el TP Final (`python main.py ingest` hace los pasos 2 y 3 de una vez).

Requiere `OPENAI_API_KEY` en el `.env` (embeddings `text-embedding-3-small` y LLM de las Killer Queries).

## Uso (desde la raíz del repo, en este orden)

1. **`python entrega_2/pipeline_vectorial.py`** (A.4) — genera los embeddings de la base y construye/persiste el índice FAISS en `data/faiss_index/`. Si el índice ya existe y coincide con el dataset, se recarga desde disco sin llamar a la API (A.5). Corre 3 consultas de prueba y muestra el top-3 con su similitud.

2. **`python entrega_2/vector_db.py`** (B.1–B.4) — carga la base en una colección ChromaDB persistente (`data/chroma_db/`) con `upsert`; simula un cambio de estado en caliente sobre `RUTA-KEF-MAD` verificado con `coleccion.get()` (B.3); y corre 4 pruebas de búsqueda híbrida (semántica + filtro `where` nativo por metadatos, B.4; la cuarta, fuera de catálogo). Define también el umbral de aceptación `UMBRAL_DISTANCIA = 0.50` (C.2): `buscar_vuelos()` descarta los resultados con distancia coseno mayor y no devuelve nada si ninguno lo supera.

3. **`python entrega_2/etl_purga.py`** (B.5) — normaliza claves y tipos inconsistentes, resuelve la colisión de IDs, detecta y elimina casi-duplicados semánticos por umbral de distancia coseno, y deja la colección purgada (21 rutas).

4. **`python entrega_2/B6_test_killer_queries.py`** (B.6) — corre las 3 Killer Queries contra la colección purgada y aplica el umbral de C.2: si algún resultado lo supera, genera el contexto y la respuesta del LLM; si no, responde "no dispongo de esa información" sin invocar al LLM. Resultados en [`resultados_killer_queries.md`](resultados_killer_queries.md).

## Archivos

| Archivo | Contenido |
|---|---|
| [`pipeline_vectorial.py`](pipeline_vectorial.py) | A.4 y A.5: índice FAISS persistido |
| [`vector_db.py`](vector_db.py) | B.1 a B.4 y C.2: ChromaDB, evento en caliente, búsqueda híbrida y umbral |
| [`etl_purga.py`](etl_purga.py) | B.5: ETL y purga semántica |
| [`B6_test_killer_queries.py`](B6_test_killer_queries.py) | B.6: Killer Queries |
| [`resultados_killer_queries.md`](resultados_killer_queries.md) | Resultados de B.6 |
| [`informe.md`](informe.md) | Informe de la entrega |
