### B.6 — Killer Queries
| # | Consulta | Qué pone a prueba | Resultado esperado | Resultado real | ¿Pasó? |
|---|---|---|---|---|---|
| **1** | *"Quiero pegarme una escapada barata en el puente aéreo para laburar en el día"* | Poder semántico: jerga sin palabras exactas del documento | Recuperar `RUTA-BCN-MAD` reconociendo "puente aéreo" y "viaje de negocios" sin mencionar las ciudades. | Recuperó un conjunto de 3 rutas (`RUTA-BCN-MAD-DUP-TEXTO`, `RUTA-BCN-FCO`, `RUTA-BCN-MAD`). El LLM detectó el contexto corporativo pero adoptó una postura conservadora con la palabra "barata" por no encontrarla textualmente. | **Sí** |
| **2** | *"Busco un vuelo directo para ir de Madrid a Roma"* | El metadato salva el día: la semántica cruda traería un desastre, el filtro lo bloquea | Filtrar por `vuelo_directo_disponible == true` excluyendo rutas indirectas o similares. | Retornó únicamente `RUTA-MAD-FCO` aplicando el filtro estricto por metadatos y detalló características y precios con éxito. | **Sí** |
| **3** | *"¿Qué vuelos tienen disponibles para ir desde Buenos Aires a Tokio?"* | Prueba de estrés: consulta fuera del catálogo — debe responder "no tengo eso" | Reconocer la falta de contexto relevante y no alucinar. | Recuperó vecinos aleatorios del catálogo (`RUTA-OSL-TLL`, etc.), pero el LLM reconoció la falta de coincidencia y respondió correctamente que no dispone de esa ruta. | **Sí** |


Resultados:
```text

==========================================
QUERY: 'Quiero pegarme una escapada barata en el puente aéreo para laburar en el día'

--> IDs Recuperados: ['RUTA-BCN-MAD-DUP-TEXTO', 'RUTA-BCN-FCO', 'RUTA-BCN-MAD']

--> Respuesta LLM:
No disponés de esa información en el catálogo. La ruta Barcelona–Madrid (BCN–MAD) es predominantemente para viajeros de negocios, pero no se menciona que sea una escapada barata. La mediana de precios es de 117€ saliendo de Barcelona y 100€ saliendo de Madrid.

==========================================
QUERY: 'Busco un vuelo directo para ir de Madrid a Roma'
FILTRO METADATOS: {'vuelo_directo_disponible': True}

--> IDs Recuperados: ['RUTA-MAD-FCO']

--> Respuesta LLM:
Puedes encontrar vuelos directos de Madrid a Roma (MAD–FCO) operados principalmente por Iberia, Air Europa, ITA Airways, así como opciones low-cost como Ryanair y Wizz Air. La duración típica del vuelo es de entre 2h20 y 2h40. Los precios son en promedio de 150€, con un rango que va de 25€ a 675€ según la antelación de compra. Los vuelos son más frecuentes los martes y los fines de semana.

==========================================
QUERY: '¿Qué vuelos tienen disponibles para ir desde Buenos Aires a Tokio?'

--> IDs Recuperados: ['RUTA-OSL-TLL', 'RUTA-ATH-LHR', 'RUTA-BGY-BVA']

--> Respuesta LLM:
No dispongo de esa información en el catálogo.
```