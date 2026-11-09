# Entrega 3 — Sistema RAG con LangChain

RAG básico y avanzado (chunking con solapamiento y reranking con un LLM juez) sobre la base de la Entrega 2, con trazas en LangSmith y evaluación con RAGAS. Resultados y análisis en [`informe.md`](informe.md).

El RAG avanzado de esta entrega es la herramienta `buscar_info_rutas` del agente del TP Final (`src/retrieval.py` lo reutiliza sin modificar este archivo).

**Antes de correr:** la base tiene que estar construida. Desde la raíz, `python main.py ingest` arma todo de una vez (carga y ETL de la Entrega 2, y la colección de chunks de B.2). Requiere `OPENAI_API_KEY` en el `.env`.

## Uso (desde la raíz del repo)

**`python entrega_3/rag_pipeline.py`** (Parte A) — pipeline RAG con LangChain LCEL (retriever → prompt con guardrails → `gpt-4o-mini` → parser) sobre la colección ChromaDB de la Entrega 2. Corre la matriz de resiliencia de 4 pruebas (factual, fuera de dominio, ataque de complacencia y sinónimos) y muestra, para cada respuesta, los documentos fuente con su fragmento y sus metadatos.

**`python entrega_3/rag_pipeline.py --parte b2`** (Parte B.2) — corta las fichas en chunks de 500 caracteres con 100 de solapamiento, los indexa en una colección nueva (`vuelos_smart_flight_assistant_chunks`, sin tocar la de la Entrega 2) y compara la recuperación contra la versión básica sobre las consultas de detalle de B.1.

**`python entrega_3/rag_pipeline.py --parte b1`** (Parte B.1, segunda falla) — reproduce la falla de "chunk cortado": compara el RAG básico, el de chunks y el avanzado sobre preguntas cuya respuesta quedó en un chunk que no nombra su ruta, y prueba en memoria (sin modificar la colección) el efecto de anteponerle el nombre de la ruta. Requiere la colección de chunks (`--parte b2`).

**`python entrega_3/rag_pipeline.py --parte b3`** (Parte B.3) — reranking con LLM como juez: trae 8 candidatos de la colección de chunks, los puntúa con `gpt-4o-mini` y pasa los 3 mejores a la generación.

**`python entrega_3/rag_pipeline.py --parte b4`** (Parte B.4) — corre el chain avanzado con trazabilidad en LangSmith. Requiere `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` y `LANGSMITH_PROJECT` en el `.env`.

**`python entrega_3/evaluacion_ragas.py`** (Parte C) — evalúa el RAG básico y el avanzado con RAGAS (Faithfulness, Answer Relevancy, Context Precision y Context Recall) sobre [`golden_dataset.json`](golden_dataset.json), y guarda los resultados en [`resultados_ragas.json`](resultados_ragas.json). Con `--pipeline basico` o `--pipeline avanzado` evalúa uno solo. También se corre con `python main.py eval`. Cada corrida reemplaza `resultados_ragas.json` y los puntajes varían entre corridas: si se vuelve a correr, hay que actualizar las tablas de la Parte C del informe.

## Archivos

| Archivo | Contenido |
|---|---|
| [`rag_pipeline.py`](rag_pipeline.py) | Partes A y B: RAG básico, chunking, juez LLM y trazas |
| [`evaluacion_ragas.py`](evaluacion_ragas.py) | Parte C: evaluación con RAGAS |
| [`golden_dataset.json`](golden_dataset.json) | Golden dataset (16 preguntas) |
| `resultados_ragas*.json` | Resultados de RAGAS (corrida final y antes/después de C.4) |
| [`langsmith_trace.png`](langsmith_trace.png) | Captura de la traza de B.4 |
| [`informe.md`](informe.md) | Informe de la entrega |
