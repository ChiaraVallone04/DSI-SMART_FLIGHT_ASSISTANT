### B.6 — Killer Queries (con umbral de distancia C.2 = 0.50)
| # | Consulta | Qué pone a prueba | Resultado esperado | Resultado real | ¿Pasó? |
|---|---|---|---|---|---|
| **1** | *"Quiero pegarme una escapada barata en el puente aéreo para laburar en el día"* | Poder semántico: jerga sin palabras exactas del documento | Recuperar `RUTA-BCN-MAD` reconociendo "puente aéreo" y "viaje de negocios" sin mencionar las ciudades. | Los 3 candidatos (`RUTA-BCN-FCO` 0.623, `RUTA-BCN-MAD` 0.633, `RUTA-BRE-FRA` 0.647) superan el umbral de 0.50 y se descartan: responde "no dispongo de esa información" sin llamar al LLM. | **No** (falso negativo conocido, ver B.6 del informe) |
| **2** | *"Busco un vuelo directo para ir de Madrid a Roma"* | El metadato salva el día: la semántica cruda traería un desastre, el filtro lo bloquea | Filtrar por `vuelo_directo_disponible == true` excluyendo rutas indirectas o similares. | Retornó únicamente `RUTA-MAD-FCO` (distancia 0.279, bajo el umbral) y el LLM detalló características y precios. | **Sí** |
| **3** | *"¿Qué vuelos tienen disponibles para ir desde Buenos Aires a Tokio?"* | Prueba de estrés: consulta fuera del catálogo — debe responder "no tengo eso" | Reconocer la falta de contexto relevante y no alucinar. | Los 3 vecinos (`RUTA-OSL-TLL` 0.561, `RUTA-ATH-LHR` 0.574, `RUTA-MAD-FCO` 0.605) superan el umbral y se descartan en el retriever; el LLM ni se invoca. | **Sí** |


Resultados:
```text

==========================================
QUERY: 'Quiero pegarme una escapada barata en el puente aéreo para laburar en el día'

--> Candidatos (id: distancia): {'RUTA-BCN-FCO': 0.6231, 'RUTA-BCN-MAD': 0.6326, 'RUTA-BRE-FRA': 0.647}
--> IDs Recuperados (distancia <= 0.5): []

--> Respuesta: No dispongo de esa información en el catálogo.
    (ningún resultado superó el umbral de similitud; no se invocó al LLM)

==========================================
QUERY: 'Busco un vuelo directo para ir de Madrid a Roma'
FILTRO METADATOS: {'vuelo_directo_disponible': True}

--> Candidatos (id: distancia): {'RUTA-MAD-FCO': 0.2787}
--> IDs Recuperados (distancia <= 0.5): ['RUTA-MAD-FCO']

--> Respuesta LLM:
La ruta Madrid–Roma (MAD–FCO) ofrece vuelos directos, con más del 97% de los vuelos sin escalas. La duración típica del vuelo es de entre 2h20 y 2h40. Las aerolíneas que cubren esta ruta incluyen Iberia, Air Europa, ITA Airways, Ryanair y Wizz Air. Los precios son aproximadamente 150€ de media. Te recomiendo verificar la disponibilidad y horarios específicos para tu fecha de viaje.

==========================================
QUERY: '¿Qué vuelos tienen disponibles para ir desde Buenos Aires a Tokio?'

--> Candidatos (id: distancia): {'RUTA-OSL-TLL': 0.561, 'RUTA-ATH-LHR': 0.574, 'RUTA-MAD-FCO': 0.605}
--> IDs Recuperados (distancia <= 0.5): []

--> Respuesta: No dispongo de esa información en el catálogo.
    (ningún resultado superó el umbral de similitud; no se invocó al LLM)
```
