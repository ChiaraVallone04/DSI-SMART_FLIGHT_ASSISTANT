# Entrega 1 — Extracción estructurada de intenciones

Script que recibe una consulta de vuelos en lenguaje natural y la convierte en un JSON validado con Pydantic (intención, origen, destino, fechas, presupuesto…), contra Gemini y, como alternativa, contra Claude. El análisis completo está en [`informe.md`](informe.md).

Las intenciones que define [`schemas.py`](schemas.py) son las mismas que hoy usa el router del agente del TP Final (`src/graph.py`).

## Configuración

Desde la raíz del repo: `pip install -r requirements.txt`, copiar [`../.env.example`](../.env.example) a `.env` y completar `GEMINI_API_KEY` (`GEMINI_MODEL_NAME` ya trae un valor por defecto). Para el backend alternativo con Claude, completar además `ANTHROPIC_API_KEY` y `ANTHROPIC_MODEL_NAME`.

## Uso (desde la raíz del repo)

```bash
python entrega_1/app.py "<tu consulta>"           # una consulta contra Gemini
python entrega_1/lote_pruebas.py                  # lote de 6 casos con Chain of Thought -> resultados_lote_cot.md
python entrega_1/lote_pruebas.py --zero-shot      # el mismo lote en zero-shot -> resultados_lote_zero_shot.md
python entrega_1/app_claude.py "<tu consulta>"    # lo mismo contra Claude (Anthropic)
python entrega_1/lote_pruebas_claude.py           # -> resultados_lote_claude_cot.md (o --zero-shot)
python entrega_1/A4_tokentest.py                  # A.4: medición de tokens del prompt
```

Los resultados del lote se guardan en esta carpeta, se corra desde donde se corra.

## Archivos

| Archivo | Contenido |
|---|---|
| [`schemas.py`](schemas.py) | Contrato Pydantic de la entrada y de la salida del LLM (C.1) |
| [`app.py`](app.py) · [`app_claude.py`](app_claude.py) | Script de C.2 (Gemini / Claude) |
| [`config.py`](config.py) · [`config_claude.py`](config_claude.py) | Credenciales desde `.env` y cliente de cada proveedor |
| [`extraccion.py`](extraccion.py) · [`extraccion_claude.py`](extraccion_claude.py) · [`extraccion_comun.py`](extraccion_comun.py) | Llamada al LLM con salida estructurada y normalización de fechas |
| [`prompts.py`](prompts.py) | System prompts (zero-shot y Chain of Thought) |
| [`casos_test.py`](casos_test.py) · [`lote_pruebas.py`](lote_pruebas.py) · [`lote_pruebas_claude.py`](lote_pruebas_claude.py) | Lote de prueba de C.3 |
| `resultados_lote_*.md` | Tablas de resultados del lote (C.3) |
| [`A4_tokentest.py`](A4_tokentest.py) | Medición de tokens (A.4) |
| [`informe.md`](informe.md) | Informe de la entrega |
