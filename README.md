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

2. **`python vector_db.py`** (B.1) — carga la misma base en una colección ChromaDB persistente (`chroma_db/`), con `upsert` para poder re-ejecutar sin duplicar.

3. **`python B3_evento_caliente.py`** (B.3) — simula un cambio de estado en caliente sobre `RUTA-KEF-MAD` y verifica el resultado con `coleccion.get()`.

4. **`python B4_busqueda_hibrida.py`** (B.4) — corre 3 pruebas de búsqueda híbrida (semántica + filtro `where` nativo por metadatos).

5. **`python etl_purga.py`** (B.5) — normaliza claves/tipos inconsistentes, resuelve colisión de IDs, detecta y elimina casi-duplicados semánticos por umbral de distancia coseno, y reconstruye la colección de ChromaDB ya purgada.

6. **`python B6_test_killer_queries.py`** (B.6) — corre las 3 Killer Queries contra la colección purgada y genera el contexto + respuesta del LLM (ver resultados en [`resultados_killer_queries.md`](resultados_killer_queries.md)).

El detalle narrativo completo (justificaciones, capturas, reflexiones) está en [`informe_entrega2.md`](informe_entrega2.md). `faiss_index/` y `chroma_db/` no se versionan (`.gitignore`): se reconstruyen enteros desde `base_conocimiento.json` corriendo los pasos de arriba.
