# Smart Flight Assistant — TP Integrador

**Dominio:** Asistente Virtual Intuitivo de Planificación de Vuelos y Optimización de Itinerarios (*Smart Flight Assistant*).

**Integrantes:**
- Chiara Vallone
- Nicolas Diego Diddi
- Gino Frigoni
- Jorge Alonso
- Joaquin Darquier

El repositorio sigue la organización de la app de referencia de la cátedra (`ortelana_app`): la **raíz es el sistema final** (TP Final: agente + API), con un único punto de entrada (`main.py`), el código en `src/`, los datos en `data/`, las pruebas en `tests/` y la configuración no secreta en `config.toml`. Las entregas anteriores, sobre las que se construyó, quedan en su propia carpeta.

- **TP Final:** [`informe.md`](informe.md) · [`ficha_tecnica.md`](ficha_tecnica.md)
- **Entregas anteriores:** [Entrega 1](entrega_1/README.md) (extracción estructurada) · [Entrega 2](entrega_2/README.md) (base vectorial) · [Entrega 3](entrega_3/README.md) (RAG con LangChain)

## Estructura

```
DSI-SMART_FLIGHT_ASSISTANT/
├── main.py              Punto de entrada (CLI): ingest | ask | serve | matriz | verificar | costos | eval
├── config.toml          Configuración no secreta: modelo, límites del agente, API, guardrails, rutas
├── .env.example         Variables de entorno (solo secretos) — copiar a .env
├── requirements.txt     Dependencias con versión fija
├── pytest.ini           Configuración de la suite de seguridad
├── informe.md           Informe del TP Final (partes A a E)
├── ficha_tecnica.md     Ficha técnica del sistema
├── src/                 Código del agente y de la API
│   ├── config.py          lee config.toml y resuelve las rutas
│   ├── tools.py           las 3 herramientas: RAG, sensor de disponibilidad, actuador de reservas
│   ├── retrieval.py       RAG avanzado de la Entrega 3 adaptado al agente
│   ├── prompts.py         prompts del router, del agente ReAct y de la conversación
│   ├── graph.py           grafo LangGraph: router, rama RAG, nodo ReAct, memoria SqliteSaver
│   ├── guardrails.py      validadores de entrada (inyección, longitud) y salida (datos personales)
│   ├── observability.py   logs JSON y métricas (p95, bloqueos, costo)
│   ├── schemas.py         contratos Pydantic de la API
│   ├── api.py             API REST (FastAPI)
│   ├── ingest.py          construye la base vectorial (carga + ETL + chunks)
│   ├── validation.py      matriz de validación del agente
│   ├── verify.py          verificación end-to-end de la API y prueba de reinicio
│   └── costs.py           análisis de costos con las trazas de LangSmith
├── tests/
│   └── security_tests.py  suite automática de seguridad (pytest)
├── data/
│   ├── base_conocimiento.json   fuente de verdad del catálogo (21 rutas tras la purga)
│   ├── chroma_db/               base vectorial (se genera con: python main.py ingest)
│   └── runtime/                 memoria, reservas y logs (se generan al usar el sistema)
├── resultados/          Evidencia: matriz de validación, costos y prueba de reinicio
├── entrega_1/           Extracción estructurada de intenciones (Pydantic + LLM)
├── entrega_2/           Embeddings, FAISS, ChromaDB, búsqueda híbrida y ETL de purga
└── entrega_3/           RAG básico y avanzado, LangSmith y RAGAS
```

`data/chroma_db/`, `data/faiss_index/`, `data/runtime/` y el `.env` no se versionan (`.gitignore`).

## TP Final — Agente inteligente en producción

El RAG de la Entrega 3 convertido en un **agente con LangGraph** que decide qué herramientas usar (catálogo RAG, sensor de disponibilidad y actuador de reservas), recuerda cada conversación en **SQLite** y queda expuesto como **API REST** (FastAPI) con guardrails, logs JSON, métricas y trazas en LangSmith.

### Levantarlo desde cero (otra máquina)

Requisitos: **Python 3.11, 3.12 o 3.13** (en 3.14 `ragas`, que usa la Entrega 3, no instala porque una de sus dependencias todavía no publica versión para 3.14) y una clave de OpenAI. Todos los comandos se corren **desde la raíz del repo**.

```bash
# 1. Entorno e instalación
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Configuración: copiar el ejemplo y completar OPENAI_API_KEY (LangSmith es opcional)
copy .env.example .env            # Windows   (Linux/Mac: cp .env.example .env)

# 3. Base vectorial (data/chroma_db/ no se versiona): carga + purga ETL + chunks, en un solo paso (~20 s)
python main.py ingest

# 4. Levantar la API
python main.py serve
```

Con el servidor arriba, abrir **http://localhost:8000/docs** (Swagger) para probar cada endpoint desde el navegador, o en otra terminal correr la verificación automática:

```bash
python main.py verificar
```

Para una consulta suelta sin levantar la API: `python main.py ask --session demo "¿Qué aerolíneas operan Madrid–Roma?"`.

### Comandos

| Comando | Qué hace |
|---|---|
| `python main.py ingest` | Construye la base vectorial desde `data/base_conocimiento.json` (carga y ETL de la E2, chunks de la E3) |
| `python main.py ask --session <id> "<mensaje>"` | Un turno de conversación con el agente (misma sesión = misma memoria) |
| `python main.py serve` | Levanta la API REST (host y puerto en `config.toml`; también `uvicorn src.api:app --port 8000`) |
| `python main.py matriz` | Matriz de validación del agente → `resultados/resultados_matriz.md` |
| `python main.py verificar [--paso1 \| --paso2]` | Verificación end-to-end contra la API levantada, y prueba de reinicio |
| `python main.py costos [--horas 24]` | Costos y latencia por camino del grafo, desde LangSmith → `resultados/resultados_costos.md` |
| `python main.py eval [--pipeline ambos]` | Evaluación RAGAS del RAG de la Entrega 3 sobre su golden dataset |
| `pytest` | Suite de seguridad completa (54 tests; 3 llaman a OpenAI). `pytest -m "not llm"`: solo los 51 deterministas |

`ask`, `serve`, `matriz`, `verificar`, `eval` y `pytest` necesitan la base vectorial: correr `python main.py ingest` antes, una sola vez (si no, fallan con "la colección no existe"). Y todos los comandos que llaman a OpenAI (`ingest` incluido) requieren una `OPENAI_API_KEY` válida en `.env`.

### Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| `GET` | `/health` | Estado del servicio y sus dependencias (catálogo, memoria SQLite, clave de OpenAI). 503 si algo falla |
| `POST` | `/consultar` | Body `{"pregunta": "...", "thread_id": "..."}`. Guardrails de entrada → agente → guardrail de salida. Devuelve `respuesta`, `herramientas_usadas`, `camino`, `exito`, `bloqueado` |
| `GET` | `/estado/{thread_id}` | Historial de la conversación guardado en SQLite (con datos personales enmascarados). 404 si no existe |
| `GET` | `/metricas` | Consultas, errores, bloqueos, latencia promedio / p95 / p99, herramientas, caminos del grafo, tokens y costo |

Ejemplo con curl (Linux/Mac; en Windows conviene usar `/docs` o `python main.py verificar`, porque la consola puede romper los acentos del JSON):

```bash
curl -X POST http://localhost:8000/consultar -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Qué vuelos hay de Madrid a Roma el 2026-12-05?", "thread_id": "demo_001"}'
curl http://localhost:8000/estado/demo_001
```

### Memoria después de reiniciar

La memoria vive en `data/runtime/memoria.db` (no se versiona), así que sobrevive a reinicios del servidor:

```bash
python main.py verificar --paso1   # crea una conversación nueva
# cortar el servidor con Ctrl+C y volver a levantarlo
python main.py verificar --paso2   # el agente recuerda la ruta de la conversación anterior
```

Los dos pasos dejan la evidencia en [`resultados/resultados_reinicio.md`](resultados/resultados_reinicio.md).

### Evidencia

| Archivo | Contenido |
|---|---|
| [`resultados/resultados_matriz.md`](resultados/resultados_matriz.md) | Matriz de validación: 4 escenarios + memoria + límite de iteraciones |
| [`resultados/resultados_costos.md`](resultados/resultados_costos.md) | Costo y latencia por camino del grafo (LangSmith) |
| [`resultados/resultados_reinicio.md`](resultados/resultados_reinicio.md) | La memoria sobrevive a un reinicio del servidor |
