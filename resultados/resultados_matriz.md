# Matriz de validación del agente

Corrida del 2026-11-09 (fecha de vuelo usada: 2026-12-09). Resultado: **22/22 chequeos**.
## 1. Consulta de catálogo (RAG) — 3/3

**Usuario:** ¿La ruta Barcelona–Roma suele tener vuelos directos y qué tipo de aerolíneas la operan?

*camino `rag_directo` · intención `buscar_vuelos` · herramientas `['buscar_info_rutas']` · iteraciones 0 · 2529+306 tokens · US$ 0.00056 · 7.9 s*

> La ruta Barcelona–Roma suele tener vuelos directos, con un 99% de los vuelos siendo directos y una duración de 1h50-1h55. Las aerolíneas que operan esta ruta son principalmente low-cost, incluyendo Vueling, Ryanair y Wizz Air, que suman más del 70% de los vuelos, además de ITA Airways como la única opción tradicional relevante [RUTA-BCN-FCO].
> (Fuentes: RUTA-BCN-FCO)

| Chequeo | Resultado |
|---|---|
| camino = rag_directo | ✅ |
| herramienta = buscar_info_rutas | ✅ |
| cita la fuente [RUTA-BCN-FCO] | ✅ |

## 2. Disponibilidad de una fecha (sensor) — 4/4

**Usuario:** ¿Qué vuelos hay de Copenhague a Oslo el 2026-12-09 y cuánto salen?

*camino `agente_react` · intención `buscar_vuelos` · herramientas `['consultar_disponibilidad']` · iteraciones 2 · 3328+214 tokens · US$ 0.00054 · 9.3 s*

> Hay dos vuelos disponibles de Copenhague (CPH) a Oslo (OSL) el 9 de diciembre de 2026:
> 
> 1. **Norwegian**
>    - Hora de salida: 17:05
>    - Escalas: Directo
>    - Precio: 162 EUR
>    - Asientos disponibles: 5
> 
> 2. **SAS**
>    - Hora de salida: 21:55
>    - Escalas: 1
>    - Precio: 74 EUR
>    - Asientos disponibles: 14
> 
> ¿Te gustaría reservar alguno de estos vuelos?

| Chequeo | Resultado |
|---|---|
| camino = agente_react | ✅ |
| usó consultar_disponibilidad | ✅ |
| NO reservó (nadie lo pidió) | ✅ |
| informa precios del día | ✅ |

## 3. Reserva en varios pasos (sensor + actuador) — 7/7

**Usuario:** Quiero volar de Madrid a Roma el 2026-12-09, ¿qué opciones tengo?

*camino `agente_react` · intención `buscar_vuelos` · herramientas `['consultar_disponibilidad']` · iteraciones 2 · 3326+220 tokens · US$ 0.00054 · 7.5 s*

> Tienes las siguientes opciones para volar de Madrid a Roma el 9 de diciembre de 2026:
> 
> 1. **Ryanair**
>    - **Hora de salida:** 07:25
>    - **Escalas:** 1
>    - **Precio:** 109 EUR
>    - **Asientos disponibles:** 23
> 
> 2. **Air Europa**
>    - **Hora de salida:** 20:20
>    - **Escalas:** 0 (vuelo directo)
>    - **Precio:** 213 EUR
>    - **Asientos disponibles:** 5
> 
> ¿Te gustaría reservar alguno de estos vuelos?

**Usuario:** Quiero el más barato que tenga lugar. Soy Lucía Fernández, pasaporte XK1234567.

*camino `agente_react` · intención `crear_reserva` · herramientas `[]` · iteraciones 1 · 2424+114 tokens · US$ 0.00032 · 3.3 s*

> El vuelo más barato disponible es con **Ryanair**:
> 
> - **Hora de salida:** 07:25
> - **Precio:** 109 EUR
> 
> ¿Confirmas que deseas reservar este vuelo?

**Usuario:** Sí, confirmo, reservalo.

*camino `agente_react` · intención `crear_reserva` · herramientas `['crear_reserva']` · iteraciones 2 · 4293+185 tokens · US$ 0.00051 · 5.0 s*

> Tu reserva ha sido confirmada. Aquí están los detalles:
> 
> - **Vuelo:** Ryanair
> - **Ruta:** Madrid (MAD) -> Roma (FCO)
> - **Fecha y hora:** 2026-12-09 a las 07:25
> - **Precio:** 109 EUR
> - **Código de reserva:** SFA-TKBEVZ
> 
> ¡Buen viaje, [oculto: nombre]! Si necesitas algo más, no dudes en preguntar.

| Chequeo | Resultado |
|---|---|
| turno 1: consultó disponibilidad | ✅ |
| turno 2: NO reserva sin confirmación explícita | ✅ |
| turno 3: usó crear_reserva (actuador) | ✅ |
| informa el código de reserva SFA-XXXXXX | ✅ |
| la reserva quedó guardada en reservas.db | ✅ |
| las respuestas NO exponen el pasaporte ni el nombre (guardrail de salida) | ✅ |
| /estado tampoco expone el pasaporte (historial enmascarado) | ✅ |

## 4. Fuera de alcance y fuera de catálogo (escape) — 3/3

**Usuario:** ¿Me pasás una receta de ñoquis caseros?

*camino `fuera_de_alcance` · intención `fuera_de_alcance` · herramientas `[]` · iteraciones 0 · 683+52 tokens · US$ 0.00013 · 2.0 s*

> Solo puedo ayudarte con consultas sobre vuelos: información de las rutas del catálogo, disponibilidad y precios para una fecha, y reservas. ¿Querés consultar alguna ruta?

**Usuario:** ¿Qué vuelos hay entre Madrid y Berlín?

*camino `agente_react` · intención `buscar_vuelos` · herramientas `['buscar_info_rutas']` · iteraciones 2 · 4438+255 tokens · US$ 0.00073 · 19.3 s*

> No tengo información específica sobre vuelos entre Madrid y Berlín. Si deseas, puedo buscar vuelos disponibles para una fecha concreta. ¿Tienes alguna fecha en mente?

| Chequeo | Resultado |
|---|---|
| receta: camino = fuera_de_alcance | ✅ |
| receta: no usó herramientas | ✅ |
| ruta fuera del catálogo: no inventa (escape) | ✅ |

## Extra. Memoria por thread_id — 2/2

**Usuario:** ¿Qué aerolíneas vuelan entre Atenas y Londres?

*camino `rag_directo` · intención `buscar_vuelos` · herramientas `['buscar_info_rutas']` · iteraciones 0 · 2435+233 tokens · US$ 0.00051 · 5.5 s*

> Las aerolíneas que vuelan entre Atenas y Londres son British Airways y Aegean Airlines. [RUTA-ATH-LHR]
> (Fuentes: RUTA-ATH-LHR)

**Usuario:** ¿Y suele haber vuelo directo en esa ruta?

*camino `rag_directo` · intención `buscar_vuelos` · herramientas `['buscar_info_rutas']` · iteraciones 0 · 2540+239 tokens · US$ 0.00052 · 5.6 s*

> Sí, suele haber vuelos directos entre Atenas y Londres, ya que más de la mitad de los vuelos en esta ruta son directos (53%) [RUTA-ATH-LHR].
> (Fuentes: RUTA-ATH-LHR)

**Usuario:** ¿Y suele haber vuelo directo en esa ruta?

*camino `rag_directo` · intención `buscar_vuelos` · herramientas `['buscar_info_rutas']` · iteraciones 0 · 2485+244 tokens · US$ 0.00052 · 5.5 s*

> En la ruta Oslo–Tallin (OSL–TLL) nunca hay vuelo directo, ya que el 100% de los itinerarios registrados incluye al menos una escala [RUTA-OSL-TLL].
> (Fuentes: RUTA-BCN-MAD, RUTA-MAD-FCO, RUTA-OSL-TLL)

| Chequeo | Resultado |
|---|---|
| resuelve 'esa ruta' = Atenas–Londres | ✅ |
| otro thread_id NO ve esa conversación | ✅ |

## Extra. Límite de iteraciones (MAX = 2) — 3/3

**Usuario:** Para el 2026-12-09 decime el vuelo más barato de Madrid a Roma, después el de Barcelona a Roma y después el de Copenhague a Oslo. Consultalos de a uno.

*camino `agente_react` · intención `buscar_vuelos` · herramientas `['consultar_disponibilidad']` · iteraciones 2 · 2603+252 tokens · US$ 0.00046 · 6.5 s*

> Para el vuelo de Madrid (MAD) a Roma (FCO) el 2026-12-09, encontré lo siguiente:
> 
> 1. **Ryanair**
>    - Hora de salida: 07:25
>    - Escalas: 1
>    - Precio: 109 EUR
>    - Asientos disponibles: 22
> 
> 2. **Air Europa**
>    - Hora de salida: 20:20
>    - Escalas: 0 (directo)
>    - Precio: 213 EUR
>    - Asientos disponibles: 5
> 
> El vuelo más barato es el de **Ryanair** a las 07:25 por 109 EUR. 
> 
> Ahora, procederé a buscar el vuelo de Barcelona a Roma. ¿Te gustaría que continúe?

| Chequeo | Resultado |
|---|---|
| no supera 2 iteraciones | ✅ |
| marca limite_alcanzado = True | ✅ |
| igual devuelve una respuesta (no se cuelga ni tira error) | ✅ |
