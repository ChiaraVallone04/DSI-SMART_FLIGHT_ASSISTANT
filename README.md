# Smart Flight Assistant — TP Integrador (Entregas 1 y 2)

**Dominio:** Asistente Virtual Intuitivo de Planificación de Vuelos y Optimización de Itinerarios (*Smart Flight Assistant*).

**Integrantes:**
- Chiara Vallone
- Nicolas Diego Diddi
- Gino Frigoni
- Jorge Alonso
- Joaquin Darquier

## Instalación

Instalar las dependencias con `pip install -r requirements.txt` (la lista está en [`requirements.txt`](requirements.txt)). Después copiar [`.env.example`](.env.example) a `.env` y completar `GEMINI_API_KEY` con una clave real; `GEMINI_MODEL_NAME` ya viene con un valor por defecto. Para correr también el backend alternativo con Claude (Anthropic), completar además `ANTHROPIC_API_KEY` y `ANTHROPIC_MODEL_NAME`.

Para la Entrega 2 (base de conocimiento vectorial), completar también `OPENAI_API_KEY` en `.env` — se usa para generar los embeddings con `text-embedding-3-small` (FAISS y ChromaDB) y para el LLM generador de las Killer Queries (`gpt-4o-mini`).

## Uso

`python app.py "<tu input>"` corre una consulta individual contra Gemini, y `python lote_pruebas.py` (o `python lote_pruebas.py --zero-shot`) corre la serie de seis pruebas y regenera `resultados_lote_cot.md` / `resultados_lote_zero_shot.md` según la técnica.

Para correr lo mismo contra Claude (Anthropic): `python app_claude.py "<tu input>"` y `python lote_pruebas_claude.py` (o `--zero-shot`), que generan `resultados_lote_claude_cot.md` / `resultados_lote_claude_zero_shot.md`.

## Entrega 2 — Base de Conocimiento Vectorial

Los scripts de esta entrega leen y reconstruyen todo a partir de `base_conocimiento.json` (la fuente de verdad). Correr en este orden:

1. **`python pipeline_vectorial.py`** (A.4) — genera embeddings de `base_conocimiento.json` y construye/persiste el índice FAISS en `faiss_index/`. Si el índice ya existe y coincide con el dataset, se recarga desde disco sin llamar a la API. Corre 3 consultas de prueba y muestra el top-3 con su similitud.

2. **`python vector_db.py`** (B.1–B.4) — carga la base en una colección ChromaDB persistente (`chroma_db/`) con `upsert`; simula un cambio de estado en caliente sobre `RUTA-KEF-MAD` verificado con `coleccion.get()` (B.3); y corre 4 pruebas de búsqueda híbrida (la cuarta, fuera de catálogo) (semántica + filtro `where` nativo por metadatos, B.4). Define también el umbral de aceptación `UMBRAL_DISTANCIA = 0.50` (C.2): `buscar_vuelos()` descarta los resultados con distancia coseno mayor y no devuelve nada si ninguno lo supera.

3. **`python etl_purga.py`** (B.5) — normaliza claves/tipos inconsistentes, resuelve colisión de IDs, detecta y elimina casi-duplicados semánticos por umbral de distancia coseno, y reconstruye la colección de ChromaDB ya purgada.

4. **`python B6_test_killer_queries.py`** (B.6) — corre las 3 Killer Queries contra la colección purgada, aplica el umbral de C.2 y, si algún resultado lo supera, genera el contexto + respuesta del LLM (si no, responde "no dispongo de esa información" sin invocar al LLM) (ver resultados en [`resultados_killer_queries.md`](resultados_killer_queries.md)).

El detalle narrativo completo (justificaciones, capturas, reflexiones) está en [`informe_entrega2.md`](informe_entrega2.md). `faiss_index/` y `chroma_db/` no se versionan (`.gitignore`): se reconstruyen enteros desde `base_conocimiento.json` corriendo los pasos de arriba.

## Entrega 3 — Sistema RAG con LangChain

Usa la misma base de la Entrega 2, así que primero hay que haberla construido (pasos 2 y 3 de arriba: `vector_db.py` y `etl_purga.py`). Después:

**`python entrega_3/rag_pipeline.py`** (Parte A) — pipeline RAG con LangChain LCEL (retriever → prompt con guardrails → `gpt-4o-mini` → parser) sobre la colección ChromaDB de la Entrega 2. Corre la matriz de resiliencia de 4 pruebas (factual, fuera de dominio, ataque de complacencia y sinónimos) y muestra, para cada respuesta, los documentos fuente con su fragmento y sus metadatos. Usa la misma `OPENAI_API_KEY` del `.env`. Resultados y análisis en [`entrega_3/informe.md`](entrega_3/informe.md).

**`python entrega_3/rag_pipeline.py --parte b2`** (Parte B.2) — corta las fichas en chunks de 500 caracteres con 100 de solapamiento, los indexa en una colección nueva (`vuelos_smart_flight_assistant_chunks`, sin tocar la de la Entrega 2) y compara la recuperación contra la versión básica sobre las consultas de detalle de B.1.

**`python entrega_3/rag_pipeline.py --parte b3`** (Parte B.3) — reranking con LLM como juez: trae 8 candidatos de la colección de chunks, los puntúa con `gpt-4o-mini` y pasa los 3 mejores a la generación. Requiere haber corrido antes `--parte b2` (colección de chunks).

**`python entrega_3/rag_pipeline.py --parte b4`** (Parte B.4) — corre el chain avanzado con trazabilidad en LangSmith. Requiere `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` y `LANGSMITH_PROJECT` en el `.env` (ver `.env.example`) y haber corrido antes `--parte b2` (colección de chunks).

**`python entrega_3/evaluacion_ragas.py`** (Parte C) — evalúa el RAG básico y el avanzado con RAGAS (Faithfulness, Answer Relevancy, Context Precision y Context Recall) sobre el golden dataset de [`entrega_3/golden_dataset.json`](entrega_3/golden_dataset.json), y guarda los resultados en `entrega_3/resultados_ragas.json`. Requiere haber corrido antes `--parte b2` (colección de chunks). Con `--pipeline basico` o `--pipeline avanzado` evalúa uno solo.
