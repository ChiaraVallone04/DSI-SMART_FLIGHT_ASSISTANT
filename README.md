# Smart Flight Assistant — TP Integrador (Entrega 1)

**Dominio:** Asistente Virtual Intuitivo de Planificación de Vuelos y Optimización de Itinerarios (*Smart Flight Assistant*).

**Integrantes:**
- Chiara Vallone
- Nicolas Diego Diddi
- Gino Frigoni
- Jorge Alonso
- Joaquin Darquier

- Rodolfo Messina
- Franco Leonel Cristillo

## Instalación

Instalar las dependencias con `pip install -r requirements.txt` (la lista está en [`requirements.txt`](requirements.txt)). Después copiar [`.env.example`](.env.example) a `.env` y completar `GEMINI_API_KEY` con una clave real; `GEMINI_MODEL_NAME` ya viene con un valor por defecto.

## Uso

`python app.py "<tu input>"` corre una consulta individual, y `python lote_pruebas.py` corre la serie de seis pruebas consumiendo la API de Gemini y regenera `resultados_lote.md`.
