# Smart Flight Assistant — TP Integrador (Entrega 1)

**Dominio:** Asistente Virtual Intuitivo de Planificación de Vuelos y Optimización de Itinerarios (*Smart Flight Assistant*).

**Integrantes:**
- Chiara Vallone
- Nicolas Diego Diddi
- Gino Frigoni
- Jorge Alonso
- Joaquin Darquier

## Instalación

Instalar las dependencias con `pip install -r requirements.txt` (la lista está en [`requirements.txt`](requirements.txt)). Después copiar [`.env.example`](.env.example) a `.env` y completar `GEMINI_API_KEY` con una clave real; `GEMINI_MODEL_NAME` ya viene con un valor por defecto. Para correr también el backend alternativo con Claude (Anthropic), completar además `ANTHROPIC_API_KEY` y `ANTHROPIC_MODEL_NAME`.

## Uso

`python app.py "<tu input>"` corre una consulta individual contra Gemini, y `python lote_pruebas.py` (o `python lote_pruebas.py --zero-shot`) corre la serie de seis pruebas y regenera `resultados_lote_cot.md` / `resultados_lote_zero_shot.md` según la técnica.

Para correr lo mismo contra Claude (Anthropic): `python app_claude.py "<tu input>"` y `python lote_pruebas_claude.py` (o `--zero-shot`), que generan `resultados_lote_claude_cot.md` / `resultados_lote_claude_zero_shot.md`.
