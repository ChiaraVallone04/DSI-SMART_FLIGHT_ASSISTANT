# TP Integrador DSI — Entrega 2: Del Prompt Saturado a la Base de Conocimiento Vectorial

**Dominio:** Asistente Virtual Inteligente de Planificación de Vuelos y Optimización de Itinerarios (*Smart Flight Assistant*). Se mantiene el mismo dominio definido en la Entrega 1; esta entrega construye la capa que faltaba — la Base de Conocimiento — migrando de un prompt con contexto estático a una arquitectura de recuperación vectorial.

---

## Parte A — Embeddings y Búsqueda Semántica (Clase 4)

### A.1 — Autopsia del contexto estático

Se documentan a continuación los tres problemas de meter toda la Base de Conocimiento directamente en el System Prompt, aplicados al dominio del Smart Flight Assistant.

| Problema | Aplicado al dominio |
|---|---|
| **Desangre de tokens** | La base construida en A.3 tiene **18 documentos**. |
| **Lost in the Middle** | Con 18 documentos ya es un riesgo real; a escala de catálogo, sería directamente inutilizable. |
| **Inconsistencia de estado concurrente** | El precio y la disponibilidad de un vuelo cambian constantemente, incluso dentro de la misma sesión de scraping. |

**Desangre de tokens.** Para medir el costo real (no una estimación) de mandar la base completa en cada consulta, se midieron con `tiktoken` (mismo método usado en A.4 de la Entrega 1, sobre el tokenizador de `gpt-4o`) los 18 documentos completos de `base_conocimiento.json` (texto + metadatos). El resultado: **5.345 tokens** solo por la base. Sumado al System Prompt actual de la Entrega 1 (211 tokens), se estarían mandando **5.556 tokens de contexto fijo en cada consulta**, antes de que el usuario escriba una sola palabra de su pregunta. Y esto es con apenas 18 documentos: si se quisiera cubrir las 4.147 rutas reales que existen en el dataset completo (`europe_flights_final.csv`), a un promedio de ~297 tokens por documento, serían **~1,23 millones de tokens por consulta** — inviable tanto en costo como en el límite de contexto del modelo. Lo más grave es que el 94% de esos 18 documentos (17 de 18) resultan irrelevantes para cualquier pregunta puntual del usuario, y se pagarían igual, en cada consulta, todo el tiempo.

**Lost in the Middle.** Si los 18 documentos se insertaran en el orden en que están en `base_conocimiento.json`, uno como `RUTA-BRE-FRA` (posición 9 de 18) queda enterrado justo en el medio de un bloque de ~5.300 tokens. Es exactamente el fenómeno descripto en la literatura de "lost in the middle": los LLM prestan más atención a lo que está al principio y al final del contexto, y degradan su capacidad de usar correctamente la información que cae en el medio de un bloque largo. Con 18 documentos el riesgo ya es real; con cientos o miles sería directamente inutilizable — el modelo podría "ver" la ruta Bremen-Fráncfort en su contexto y aun así no usarla para responder, simplemente por dónde quedó ubicada.

**Inconsistencia de estado concurrente.** Se buscó evidencia directa de esto en el propio CSV, en vez de asumirlo en abstracto. En una muestra de 400.000 filas se encontraron **673 casos donde el mismo vuelo** (misma aerolínea, mismo número de vuelo, mismo horario de salida — la clave primaria definida para la tabla `vuelos` en la Entrega 1) **aparece con dos precios distintos** registrados casi al mismo instante de scraping. Hay un caso todavía más simple y determinista: la columna `days_left` baja exactamente en 1 cada día que pasa, para cada vuelo futuro, sin ninguna excepción — cualquier dato que se "congele" hoy sobre "faltan 14 días para este vuelo" queda mal mañana. Si esa información viviera embebida en un párrafo estático de la Base de Conocimiento, el vector nunca se entera de que cambió — queda desactualizado desde el minuto uno, el mismo problema de fondo ya diagnosticado en A.2 de la Entrega 1 con la alucinación de precios del LLM genérico.

**Cierre — ¿por qué un `SELECT ... WHERE descripcion LIKE '%...%'` tampoco alcanza?**
Porque `LIKE` compara texto literal, no significado: si el usuario escribe "quiero una escapada barata a Italia" y ningún documento contiene exactamente esas palabras (dicen "Roma", "low-cost", "mediterráneo"), el `LIKE` no encuentra nada aunque el documento correcto exista. Tampoco resuelve el desangre de tokens ni el lost in the middle — seguiría siendo necesario decidir qué mandar y en qué orden. Y a diferencia de un embedding, `LIKE` no tiene noción de "cuán parecido" es un resultado a otro para priorizar: es una coincidencia binaria (sí/no), no un ranking por relevancia semántica.
