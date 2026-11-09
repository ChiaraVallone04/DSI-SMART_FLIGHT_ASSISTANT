### B.6 — Killer Queries (con umbral de distancia C.2 = 0.50)
| # | Consulta | Qué pone a prueba | Resultado esperado | Resultado real | ¿Pasó? |
|---|---|---|---|---|---|
| **1** | *"Quiero pegarme una escapada barata en el puente aéreo para laburar en el día"* | Poder semántico: jerga sin palabras exactas del documento | Recuperar `RUTA-BCN-MAD` reconociendo "puente aéreo" y "viaje de negocios" sin mencionar las ciudades. | Los 3 candidatos (`RUTA-BCN-FCO` 0.623, `RUTA-BCN-MAD` 0.633, `RUTA-MAD-FCO` 0.647) superan el umbral de 0.50 y se descartan: responde "no dispongo de esa información" sin llamar al LLM. | **No** (falso negativo conocido, ver B.6 del informe) |
| **2a** | *"ruta directa a Tallin"* — **sin filtro** | El metadato salva el día: qué recupera la semántica cruda | La semántica sola no distingue "directo" de "no directo". | Recupera `RUTA-OSL-TLL` (0.436, bajo el umbral), una ruta **sin vuelo directo**, y la entrega al LLM como contexto. El LLM, por su system prompt, aclara que "la ruta Oslo–Tallin no tiene vuelos directos". | **Parcial**: la restricción dura se violó en la recuperación y la salvó el prompt del LLM, no el sistema. |
| **2b** | *"ruta directa a Tallin"* — **con** `where={"vuelo_directo_disponible": True}` | El metadato salva el día: el filtro lo bloquea | Que `RUTA-OSL-TLL` ni se recupere. | El filtro la excluye de la búsqueda. La ruta directa más cercana (`RUTA-LHR-LIS`) está a 0.629, supera el umbral y se descarta: responde "no dispongo" sin llamar al LLM. | **Sí** |
| **3** | *"¿Qué vuelos tienen disponibles para ir desde Buenos Aires a Tokio?"* | Prueba de estrés: consulta fuera del catálogo — debe responder "no tengo eso" | Reconocer la falta de contexto relevante y no alucinar. | Los 3 vecinos (`RUTA-OSL-TLL` 0.561, `RUTA-ATH-LHR` 0.574, `RUTA-MAD-FCO` 0.605) superan el umbral y se descartan en el retriever; el LLM ni se invoca. | **Sí** |


Resultados:
```text

==========================================
QUERY: 'Quiero pegarme una escapada barata en el puente aéreo para laburar en el día'

--> Candidatos (id: distancia): {'RUTA-BCN-FCO': 0.6232, 'RUTA-BCN-MAD': 0.6328, 'RUTA-MAD-FCO': 0.647}
--> IDs Recuperados (distancia <= 0.5): []

--> Respuesta: No dispongo de esa información en el catálogo.
    (ningún resultado superó el umbral de similitud; no se invocó al LLM)

==========================================
QUERY: 'ruta directa a Tallin'

--> Candidatos (id: distancia): {'RUTA-OSL-TLL': 0.4362}
--> IDs Recuperados (distancia <= 0.5): ['RUTA-OSL-TLL']

--> Respuesta LLM:
No disponés de esa información en el catálogo. La ruta Oslo–Tallin no tiene vuelos directos.

==========================================
QUERY: 'ruta directa a Tallin'
FILTRO METADATOS: {'vuelo_directo_disponible': True}

--> Candidatos (id: distancia): {'RUTA-LHR-LIS': 0.629}
--> IDs Recuperados (distancia <= 0.5): []

--> Respuesta: No dispongo de esa información en el catálogo.
    (ningún resultado superó el umbral de similitud; no se invocó al LLM)

==========================================
QUERY: '¿Qué vuelos tienen disponibles para ir desde Buenos Aires a Tokio?'

--> Candidatos (id: distancia): {'RUTA-OSL-TLL': 0.561, 'RUTA-ATH-LHR': 0.574, 'RUTA-MAD-FCO': 0.6051}
--> IDs Recuperados (distancia <= 0.5): []

--> Respuesta: No dispongo de esa información en el catálogo.
    (ningún resultado superó el umbral de similitud; no se invocó al LLM)
```
