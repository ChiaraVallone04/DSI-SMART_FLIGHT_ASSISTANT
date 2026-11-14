# Evidencia: la memoria sobrevive a un reinicio del servidor

Generado por `python main.py verificar --paso1` y `--paso2`, con un reinicio del servidor (Ctrl+C y volver a levantarlo) entre los dos pasos.

thread_id: `memoria_reinicio_20261109_133105`

## Paso 1 — antes del reinicio (2026-11-09 13:31:05)

Proceso del servidor en marcha desde `2026-11-09T16:30:22+00:00` (43 s, 5 consultas atendidas).

**Usuario:** Hola, me interesa viajar de Copenhague a Oslo, ¿qué aerolíneas operan esa ruta?

*camino `rag_directo` · herramientas `['buscar_info_rutas']`*

> Las aerolíneas que operan la ruta de Copenhague a Oslo son principalmente Scandinavian Airlines (SAS, 74%) y la low-cost noruega Norwegian (23%) [RUTA-CPH-OSL].
> (Fuentes: RUTA-CPH-OSL)

## Paso 2 — después del reinicio (2026-11-09 13:31:40)

`GET /metricas`: proceso nuevo, en marcha desde `2026-11-09T16:31:18+00:00`, **0 consultas** atendidas (las métricas viven en memoria y se reinician con el proceso).

`GET /estado/memoria_reinicio_20261109_133105`: HTTP 200, **2 mensajes** leídos de `memoria.db`.

**Usuario:** ¿Qué ruta te había preguntado? ¿Y suele tener vuelo directo?

*camino `rag_directo` · herramientas `['buscar_info_rutas']`*

> Sí, la ruta de Copenhague a Oslo suele tener vuelo directo, con un 96% de vuelos directos en esta conexión [RUTA-CPH-OSL].
> (Fuentes: RUTA-CPH-OSL)

| Chequeo | Resultado |
|---|---|
| El servidor es un proceso nuevo (0 consultas) | ✅ |
| `/estado` conserva la conversación | ✅ |
| El agente recuerda la ruta (cita `RUTA-CPH-OSL`) | ✅ |
