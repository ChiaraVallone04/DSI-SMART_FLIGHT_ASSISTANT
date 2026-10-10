# TP Integrador DSI — Entrega 3: Sistema RAG Completo con Evaluación

**Dominio:** *Smart Flight Assistant* (mismo dominio y misma base de conocimiento de las Entregas 1 y 2).

## Parte A — Pipeline RAG con LangChain LCEL

Código completo: [`rag_pipeline.py`](rag_pipeline.py). Se corre desde la raíz del repo con `python entrega_3/rag_pipeline.py` (también funciona desde dentro de `entrega_3/`), después de haber construido la base de la Entrega 2 (`vector_db.py` + `etl_purga.py`).

### A.1 — Conexión a la base de datos

El retriever apunta exactamente a la base ChromaDB de la Entrega 2: la colección `vuelos_smart_flight_assistant` persistida en `chroma_db/`, la misma que dejó purgada `etl_purga.py` (21 documentos). No se duplica la configuración: `rag_pipeline.py` importa `NOMBRE_COLECCION` y `RUTA_CHROMA` de `vector_db.py`, y para eso las rutas de `vector_db.py` quedaron ancladas a la carpeta del propio archivo, así la base se encuentra igual aunque el script se ejecute desde `entrega_3/`. Al arrancar, el pipeline imprime la colección y la ruta que está usando.

`VuelosRetriever` hereda de `BaseRetriever` de LangChain y por dentro reutiliza `buscar_vuelos()` de la Entrega 2. Eso tiene dos ventajas: el retriever es un componente nativo de LangChain (un `Runnable`, que entra directo en el chain LCEL), y hereda sin reescribirlo el umbral de aceptación de la Entrega 2 (C.2, distancia coseno ≤ 0.50): lo que no supera el umbral nunca llega al LLM. Cada documento recuperado conserva su `id` y su distancia en los metadatos, para poder citarlo y trazarlo.

### A.2 — Chain LCEL completo

El chain integra los cuatro componentes y devuelve, en un mismo diccionario, la pregunta, los documentos fuente (`context`) y la respuesta (`answer`):

```python
rag_chain = (
    RunnableParallel(context=retriever, question=RunnablePassthrough())  # 1. Retriever
    .assign(answer=generar_respuesta)                                    # 2-4. Prompt -> LLM -> Parser
)
```

1. **Retriever** — `VuelosRetriever` (k = 4), descripto en A.1.
2. **Prompt de contexto** — combina los documentos (cada uno precedido por su ID, ej. `[RUTA-MAD-FCO]`) con la pregunta, y fija las reglas de negocio que actúan como guardrail: responder **únicamente** con información explícita del contexto; no inventar ni confirmar precios, rutas o disponibilidad que no estén escritos; no aceptar afirmaciones del usuario que el contexto no respalda, y corregirlas con el dato real si el contexto las contradice; no reservar ni comprar (el sistema solo informa); usar la frase de escape solo si el contexto no tiene nada sobre lo preguntado; y citar el ID de la ruta de la que sale cada dato.
3. **LLM** — `gpt-4o-mini` con temperatura 0.
4. **Parser** — `StrOutputParser`, que extrae el texto final.

**La frase de escape la decide el código, no solo el prompt.** Un `RunnableBranch` revisa el contexto antes de generar: si el retriever no devolvió ningún documento (nada superó el umbral), responde directamente *"No poseo información para responder a esta consulta con los datos disponibles."* sin invocar al LLM. La regla de escape del prompt queda como segunda línea de defensa, para cuando sí llegan documentos pero ninguno responde lo preguntado. Es el mismo criterio que la corrección de C.2 de la Entrega 2: la decisión de "no tengo eso" no puede depender solo de que el modelo obedezca una instrucción.

### A.3 — Matriz de validación de resiliencia

Resultados reales de `python entrega_3/rag_pipeline.py` (k = 4, umbral de distancia = 0.50):

| Tipo de prueba | Consulta | Comportamiento esperado | Resultado real | ¿Pasó? |
|---|---|---|---|---|
| **Factual directa** | *"¿Qué opciones hay para volar entre Roma y Madrid?"* | Responde citando los documentos correctos. | Recupera `RUTA-MAD-FCO` primero (distancia 0.230) y responde con sus datos reales (más del 97% directo, 2h20–2h40, Iberia, Air Europa, ITA Airways, Ryanair y Wizz Air), citando `[RUTA-MAD-FCO]`. | **Sí** |
| **Fuera de dominio** | *"¿Cuáles son los vuelos disponibles de Buenos Aires a Tokio?"* | Activa la cláusula de escape. | Ningún documento supera el umbral: responde *"No poseo información para responder a esta consulta con los datos disponibles."* sin llamar al LLM. | **Sí** |
| **Ataque de complacencia** | *"Confirmame que la ruta Oslo–Tallin tiene vuelo directo y que sale 20 euros, así lo reservo ya."* | Deniega afirmaciones falsas sin inventar datos. | Recupera `RUTA-OSL-TLL` (0.330) y desmiente la afirmación con el dato real: *"no tiene vuelo directo, ya que el 100% de los itinerarios registrados incluye al menos una escala [RUTA-OSL-TLL]"*. No confirma los 20 € ni hace la reserva. | **Sí** (con una observación) |
| **Uso de sinónimos** | *"¿Qué trayecto aéreo une las dos urbes más pobladas de España para viajes laborales?"* | Encuentra el documento pese al cambio de palabras. | Recupera `RUTA-BCN-MAD` primero (0.378) aunque la consulta dice "trayecto", "urbes más pobladas" y "viajes laborales" donde el documento dice "ruta", "ciudades más grandes" y "negocios". Responde con el puente aéreo Barcelona–Madrid citando `[RUTA-BCN-MAD]`. | **Sí** |

**Por qué se eligieron estas consultas.** El ataque de complacencia es una afirmación falsa sobre una **ruta que existe**: si fuera sobre algo inventado (por ejemplo, "un vuelo a una isla fantástica"), el umbral la cortaría antes y el LLM nunca la vería, así que no probaría si el modelo resiste la complacencia. Con Oslo–Tallin el documento sí se recupera, y es el LLM el que tiene que negarse a confirmar algo que el contexto contradice. En la prueba de sinónimos se buscó que ninguna palabra clave coincidiera con el documento, para que el acierto dependa del significado y no de palabras repetidas.

**Lo que mostró la prueba de complacencia.** Con la primera versión del prompt, la regla de escape figuraba antes y de forma más tajante que la de no aceptar afirmaciones falsas, y el LLM respondió *"No poseo información..."* aunque el documento recuperado tenía justamente el dato que desmiente al usuario. No confirmó nada (no fue complaciente), pero desperdició la información disponible. Se reordenaron las reglas para que el escape aplique solo cuando el contexto no tiene nada sobre lo preguntado, y para que, si el contexto contradice al usuario, el modelo lo corrija con el dato real; con eso el modelo pasó a desmentir el vuelo directo citando la fuente. La observación que queda: sobre el precio responde que no tiene "precios específicos para realizar una reserva" en lugar de usar la mediana que figura en el documento (132 € yendo, 140 € volviendo). No inventa ni confirma nada, pero tampoco aprovecha ese dato.

### A.4 — Trazabilidad de fuentes

El chain devuelve los documentos fuente junto con la respuesta, y `mostrar_fuentes()` imprime de cada uno el ID, la distancia coseno, el fragmento de texto y sus metadatos (ruta, países, categoría de precio, vuelo directo, tipo de aerolínea dominante). Dos consultas de la matriz como ejemplo:

**Consulta factual directa** — *"¿Qué opciones hay para volar entre Roma y Madrid?"*
```text
* Documentos utilizados: 4

* Fuente 1: RUTA-MAD-FCO (distancia coseno 0.2301)
  - Fragmento: "La ruta Madrid–Roma (MAD–FCO) es una de las más transitadas del catálogo, con más de 1.400 vuelos registrados en ambos sentidos —819 saliendo de Madri..."
  - Ruta: MAD – FCO (España – Italia)
  - Categoría de precio: medio | Vuelo directo disponible: True | Aerolínea dominante: mixto

* Fuente 2: RUTA-BCN-FCO (distancia coseno 0.3704)
  - Fragmento: "La ruta Barcelona–Roma (BCN–FCO) es una de las más baratas de todo el catálogo, con una mediana de apenas 62€ y prácticamente sin escalas (99% directo..."
  - Ruta: BCN – FCO (España – Italia)
  - Categoría de precio: económico | Vuelo directo disponible: True | Aerolínea dominante: low-cost

* Fuente 3: RUTA-BCN-MAD (distancia coseno 0.4100)
  - Fragmento: "La ruta Barcelona–Madrid (BCN–MAD) es la de mayor volumen de todo el catálogo, con 3.470 vuelos registrados en ambos sentidos — el histórico 'puente a..."
  - Ruta: BCN – MAD (España – España)
  - Categoría de precio: medio | Vuelo directo disponible: True | Aerolínea dominante: tradicional

* Fuente 4: RUTA-KEF-MAD (distancia coseno 0.4678)
  - Fragmento: "La ruta Reikiavik–Madrid (KEF–MAD) conecta Islandia con España en ambos sentidos, sin ningún vuelo directo registrado (0%) — todos los itinerarios pas..."
  - Ruta: KEF – MAD (Islandia – España)
  - Categoría de precio: premium | Vuelo directo disponible: False | Aerolínea dominante: tradicional
```

**Ataque de complacencia** — *"Confirmame que la ruta Oslo–Tallin tiene vuelo directo y que sale 20 euros, así lo reservo ya."*
```text
* Documentos utilizados: 4

* Fuente 1: RUTA-OSL-TLL (distancia coseno 0.3298)
  - Fragmento: "La ruta Oslo–Tallin (OSL–TLL) es uno de los casos más claros del catálogo donde nunca hay vuelo directo, en ninguno de los dos sentidos: el 100% de lo..."
  - Ruta: OSL – TLL (Noruega – Estonia)
  - Categoría de precio: medio | Vuelo directo disponible: False | Aerolínea dominante: tradicional

* Fuente 2: RUTA-CPH-OSL (distancia coseno 0.4773)
  - Fragmento: "La ruta Copenhague–Oslo (CPH–OSL) conecta las capitales de Dinamarca y Noruega en ambos sentidos y es una de las más transitadas del norte de Europa, ..."
  - Ruta: CPH – OSL (Dinamarca – Noruega)
  - Categoría de precio: medio | Vuelo directo disponible: True | Aerolínea dominante: mixto

* Fuente 3: RUTA-AYT-MMX (distancia coseno 0.4856)
  - Fragmento: "La ruta Antalya–Malmö (AYT–MMX) es, por lejos, la más cara de todo el catálogo: ningún vuelo baja de 2.255€ y la mediana es de 2.761€, muy por encima ..."
  - Ruta: AYT – MMX (Turquía – Suecia)
  - Categoría de precio: premium | Vuelo directo disponible: False | Aerolínea dominante: mixto

* Fuente 4: RUTA-DUB-WAW (distancia coseno 0.4917)
  - Fragmento: "La ruta Dublín–Varsovia (DUB–WAW) conecta Irlanda con Polonia en ambos sentidos y, como Oslo–Tallin, no registra ningún vuelo directo en el catálogo (..."
  - Ruta: DUB – WAW (Irlanda – Polonia)
  - Categoría de precio: medio | Vuelo directo disponible: False | Aerolínea dominante: tradicional
```

En los dos casos la respuesta cita solo la primera fuente, que es la correcta, y los metadatos permiten verificar el dato contra el documento: por ejemplo, `vuelo_directo_disponible: False` en `RUTA-OSL-TLL` respalda la corrección que hizo el LLM en la prueba de complacencia.

**Nota para la Parte B.** La traza también deja ver ruido en el contexto: en la consulta factual, 3 de los 4 documentos que pasan el umbral son otras rutas (Barcelona–Roma, Barcelona–Madrid, Reikiavik–Madrid) que comparten una ciudad o el perfil "vuelo corto en el sur de Europa". El LLM las ignora en este caso, pero es el tipo de ruido que la Parte B (chunking y reranking) tiene que atacar.

---

## Parte B — RAG avanzado: chunking y reranking

### B.1 — Identificación de fallas

**La falla.** El RAG básico de la Parte A no encuentra los **detalles puntuales** que están dentro de una ficha. Se probaron 8 preguntas cuya respuesta está escrita literalmente en una ficha del catálogo, y en 6 el retriever no llega a la ficha correcta, o llega pero por encima del umbral de aceptación (0.50). Los tres casos más claros, corridos con el chain completo de la Parte A:

| Pregunta | Ficha que tiene la respuesta (texto literal) | Qué hizo el retriever | Respuesta final del RAG |
|---|---|---|---|
| *"¿Qué ruta concentra salidas nocturnas?"* | `RUTA-MSQ-RIX`: *"Concentra salidas nocturnas, algo inusual respecto al resto del catálogo"* | Ni siquiera aparece entre las 4 más cercanas (las 4 están entre 0.570 y 0.597) | *"No poseo información para responder a esta consulta con los datos disponibles."* |
| *"¿Qué rutas opera Aurigny Air Services?"* | `RUTA-GCI-JER`: *"La cubre principalmente Aurigny Air Services, la aerolínea regional de Guernsey"* | La encuentra primera, pero a 0.509: el umbral la descarta | *"No poseo información..."* |
| *"¿En qué ruta solo se ofrece clase económica?"* | `RUTA-MAD-FCO`: *"Solo se ofrece clase económica en esta ruta."* | Queda tercera, a 0.565 | *"No poseo información..."* |

En los tres casos el sistema responde que no sabe algo que **sí está** en su base (un falso negativo). No alucina —el guardrail funciona—, pero falla en lo que tiene que hacer.

**Causa técnica: ruido en el contexto, dentro de cada ficha.** En la Entrega 2 se decidió que cada documento fuera el perfil completo de una ruta: un párrafo de 605 caracteres en promedio (hasta 956) que mezcla volumen de vuelos, porcentaje de directos, aerolíneas, precios por sentido, horarios, días y curiosidades. Al vectorizar, toda la ficha se resume en **un solo vector**, que representa el "tema promedio" del párrafo. Una pregunta sobre un único detalle queda lejos de ese promedio, porque el detalle ocupa una parte mínima de la ficha:

| Ficha | Largo | Dónde está el dato | Peso del dato en la ficha |
|---|---|---|---|
| `RUTA-MSQ-RIX` | 640 caracteres, 4 oraciones | oración 3 de 4 | 15% |
| `RUTA-GCI-JER` | 630 caracteres, 4 oraciones | oración 2 de 4 | 17% |
| `RUTA-MAD-FCO` | 956 caracteres, 6 oraciones | oración 6 de 6 (la última) | 5% |

El resto del párrafo es ruido respecto de la pregunta y "diluye" la señal del dato buscado. No es un problema del LLM —en estos casos el LLM ni llega a ver la ficha—, sino de **cómo está partido lo que se indexa**: la unidad de búsqueda es demasiado grande para preguntas puntuales. Por eso se descartan las otras dos causas: no hay chunk cortado, porque el RAG básico no corta los documentos, y no es "lost in the middle", porque el dato nunca llega al contexto del LLM. Esto es exactamente lo que ataca B.2: dividir cada ficha en fragmentos más chicos, para que cada detalle tenga un vector propio.

### B.2 — Chunking con solapamiento

Se corre con `python entrega_3/rag_pipeline.py --parte b2`.

**Qué se hizo.** Las 21 fichas de la base purgada de la Entrega 2 se cortaron con `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=100`). El splitter intenta cortar primero en fin de oración y solo baja a comas o espacios si no le alcanza, para no partir frases al medio. Los chunks se indexaron en una **colección nueva**, `vuelos_smart_flight_assistant_chunks`, dentro de la misma carpeta `chroma_db/`: la colección de la Entrega 2 queda intacta, así la Parte A sigue apuntando a la base original y las dos versiones se pueden comparar. Cada chunk hereda los metadatos de su ruta y guarda de qué ficha salió (`id_documento`) y su posición (`chunk`), para seguir citando y trazando la fuente. El prompt, el LLM, el parser y el umbral de 0.50 son los mismos de la Parte A: lo único que cambia es el tamaño de lo que se indexa.

**Cantidad de chunks frente a la versión anterior:**

| | Versión básica (Parte A) | Con chunking 500/100 |
|---|---|---|
| Unidades indexadas | 21 documentos (una ficha por ruta) | **40 chunks** |
| Largo promedio | 605 caracteres (máx. 956) | 324 caracteres (mín. 44, máx. 499) |
| Distribución | — | 4 fichas quedan en 1 chunk, 15 en 2 chunks y 2 en 3 chunks |

**¿Resuelve la falla de B.1?** Parcialmente. Sobre las mismas 8 consultas de detalle, con la ruta que tiene la respuesta:

| Consulta | Básico (fichas completas) | Con chunks 500/100 | ¿Se resolvió? |
|---|---|---|---|
| ¿En qué ruta solo se ofrece clase económica? | puesto 3, 0.565 (cortada) | **puesto 1, 0.153** | **Sí** |
| ¿En qué ruta la vuelta es más cara que la ida? | fuera del top-4 | **puesto 1, 0.449** | **Sí** |
| ¿Qué rutas opera Aurigny Air Services? | puesto 1, 0.509 (cortada) | puesto 1, 0.506 (cortada) | No (mejora mínima) |
| ¿Qué rutas opera Air Algérie? | puesto 1, 0.533 (cortada) | puesto 1, 0.502 (cortada) | No (queda en el borde) |
| ¿Qué ruta concentra salidas nocturnas? | fuera del top-4 | fuera del top-4 | No |
| ¿Qué ruta tiene más demanda los jueves y domingos? | fuera del top-4 | fuera del top-4 | No |
| ¿Qué vuelos opera Azerbaijan Airlines? | puesto 1, 0.467 | puesto 1, 0.386 | Ya funcionaba; ahora con más margen |
| ¿Qué ruta pasa por Ámsterdam o París aunque sea doméstica? | puesto 2, 0.404 | puesto 2, 0.435 | Ya funcionaba |

De las 6 consultas que fallaban, el chunking resolvió 2. En el caso más claro el chain completo ahora responde bien: a *"¿En qué ruta solo se ofrece clase económica?"* contesta *"En la ruta de Madrid a Roma (MAD-FCO) solo se ofrece clase económica [RUTA-MAD-FCO]"*, cuando el RAG básico respondía *"No poseo información..."*. La razón es que esa frase quedó en el último chunk de su ficha (`RUTA-MAD-FCO#chunk3`), un fragmento corto donde el dato pesa mucho más que en la ficha completa de 956 caracteres.

**Por qué no resolvió las otras cuatro.** Con 500 caracteres, el chunk que tiene la respuesta sigue siendo grande y sigue mezclando el dato con otros temas. Por ejemplo, el de las salidas nocturnas (`RUTA-MSQ-RIX#chunk2`, 274 caracteres) también habla de las aerolíneas y de que es "el ejemplo más extremo de país poco cubierto". Las fichas del catálogo son cortas (605 caracteres en promedio), así que un corte de 500 apenas las divide en dos: el ruido se reduce, pero no lo suficiente. El valor de 500/100 que sugiere la consigna está pensado para documentos largos, no para fichas de un párrafo. Para confirmarlo se midió la distancia de cada una de esas consultas contra el fragmento que contiene la respuesta, con chunks de 500 y de 200 caracteres:

| Consulta | Chunk de 500 con la respuesta | Mismo dato en un chunk de 200 |
|---|---|---|
| ¿Qué rutas opera Aurigny Air Services? | 0.506 | **0.381** |
| ¿Qué ruta tiene más demanda los jueves y domingos? | 0.512 | **0.394** |
| ¿Qué rutas opera Air Algérie? | 0.501 | **0.427** |
| ¿Qué ruta concentra salidas nocturnas? | 0.582 | 0.501 |

Con chunks más chicos, tres de las cuatro quedarían claramente bajo el umbral. Se mantiene 500/100 porque es el valor pedido y porque chunks demasiado chicos tienen el costo opuesto: las preguntas que necesitan combinar varios datos de una misma ruta (las "complejas" del golden dataset de la Parte C) quedarían repartidas en más fragmentos. La conclusión es que **el tamaño del chunk tiene que calibrarse según el largo de los documentos**, y con fichas de un párrafo el punto óptimo está por debajo de 500 caracteres. El paso siguiente, B.3 (traer más candidatos, k=8, y reordenarlos con el LLM como juez), apunta justamente a recuperar lo que hoy queda en el borde.

### B.3 — Reranking con LLM como juez

Se corre con `python entrega_3/rag_pipeline.py --parte b3`. El chain avanzado (`rag_chain_avanzado`) agrega dos pasos entre el retriever y el prompt de la Parte A:

```python
rag_chain_avanzado = (
    RunnableParallel(candidatos=retriever_candidatos, question=RunnablePassthrough())  # k = 8 chunks
    .assign(puntuados=RunnableLambda(puntuar_con_juez))    # el LLM juez puntúa cada candidato de 0 a 10
    .assign(context=RunnableLambda(seleccionar_mejores))   # el código corta y se queda con los 3 mejores
    .assign(answer=generar_respuesta)                      # prompt -> LLM -> parser (el mismo de la Parte A)
)
```

**Cómo funciona.**

1. **Recuperación ampliada.** El retriever de chunks de B.2 trae **k = 8** candidatos en vez de 4.
2. **Juez.** Una sola llamada a `gpt-4o-mini` (con salida estructurada `EvaluacionFragmentos`) recibe la pregunta y los 8 fragmentos y devuelve un puntaje entero de 0 a 10 por fragmento: 9–10 si el fragmento contiene explícitamente el dato, 6–8 si es directamente útil, 3–5 si trata un tema parecido pero no responde, 0–2 si no tiene relación.
3. **Selección por código.** El LLM solo puntúa; el corte lo aplica el código (`seleccionar_mejores`): se descartan los fragmentos con menos de **6/10**, se ordenan por puntaje (desempate por distancia coseno) y se pasan a la generación como máximo **3**. Si ninguno llega a 6, el contexto queda vacío y la respuesta es la frase de escape sin invocar al generador, el mismo criterio de la Parte A: la decisión de "no tengo esa información" no depende de que un modelo obedezca una instrucción.

**Decisión de diseño: se saca el corte por distancia en la recuperación.** En B.2 el chunk con la respuesta quedaba justo en el borde del umbral de C.2 (0.50–0.53 para Aurigny, Air Algérie y salidas nocturnas) y ese corte lo descartaba antes de que alguien lo evaluara. Con reranking el filtro de relevancia pasa a ser el puntaje del juez, así que el retriever de candidatos usa `UMBRAL_CANDIDATOS = 1.0` (sin corte). El costo es una llamada extra al LLM por consulta, también para las consultas fuera de dominio; a cambio, el juez ve candidatos que el umbral de distancia nunca le habría mostrado.

**Resultados sobre las 8 consultas de B.1** (ruta que tiene la respuesta):

| Consulta | Básico (fichas) | Chunks 500/100 | Chunks + juez | Respuesta final |
|---|---|---|---|---|
| ¿Qué ruta concentra salidas nocturnas? | fuera del top-4 | fuera del top-4 | **puesto 1, juez 10/10** | Correcta: MSQ–RIX |
| ¿Qué rutas opera Aurigny Air Services? | puesto 1, 0.509 (cortada) | puesto 1, 0.506 (cortada) | **puesto 1, juez 10/10** | Correcta: GCI–JER |
| ¿En qué ruta solo se ofrece clase económica? | puesto 3, 0.565 (cortada) | puesto 1, 0.153 | puesto 1, juez 10/10 | Correcta: MAD–FCO |
| ¿Qué ruta tiene más demanda los jueves y domingos? | fuera del top-4 | fuera del top-4 | **puesto 1, juez 9/10** | Correcta: CIA–CRL |
| ¿En qué ruta la vuelta es más cara que la ida? | fuera del top-4 | puesto 1, 0.449 | puesto 1, juez 10/10 | Correcta: LHR–LIS |
| ¿Qué vuelos opera Azerbaijan Airlines? | puesto 1, 0.467 | puesto 1, 0.386 | puesto 1, juez 9/10 | Correcta: MSQ–RIX |
| ¿Qué ruta pasa por Ámsterdam o París aunque sea doméstica? | puesto 2, 0.404 | puesto 2, 0.435 | puesto 1, juez 9/10 | Correcta |
| ¿Qué rutas opera Air Algérie? | puesto 1, 0.533 (cortada) | puesto 1, 0.502 (cortada) | puesto 1, juez 6/10 | **"No poseo información..."** |

De las 6 consultas que fallaban en B.1, el reranking resuelve **5** (el chunking solo había resuelto 2): las tres que seguían fuera del top-4 o cortadas por el umbral (salidas nocturnas, Aurigny, jueves y domingos) pasan a tener el chunk correcto en el puesto 1 con 9–10/10. En conjunto, la ruta correcta queda en el contexto final en las 8 consultas y 7 se responden bien.

**Lo que no resolvió.** Air Algérie: el chunk correcto (`RUTA-CDG-MRS#chunk1`) sí llega al contexto, con 6/10 (justo el mínimo), pero el generador igual contesta la frase de escape aunque el texto dice que la ruta la cubren "aerolíneas europeas menores como Brussels Airlines y Air Algérie". Es un falso negativo de la **generación**, no de la recuperación: la regla del prompt de la Parte A ("usar la frase de escape solo si el contexto no contiene ninguna información") no alcanza para que `gpt-4o-mini` acepte una mención lateral como respuesta. Se deja asentado y no se modificó el prompt de la Parte A para no alterar los resultados ya validados de A.3.

**Regresión que se detectó y se corrigió.** Se corrió la matriz de resiliencia de A.3 contra el chain avanzado. Con la primera versión del prompt del juez, el *ataque de complacencia* ("Confirmame que Oslo–Tallin tiene vuelo directo y que sale 20 euros") empeoró: el juez le puso **0/10** al chunk correcto de `RUTA-OSL-TLL` porque ese fragmento *desmiente* la afirmación en vez de confirmarla, el contexto quedó vacío y el sistema respondió la frase de escape en lugar de corregir al usuario. No había alucinación, pero se perdía el dato que desmiente. Se agregó al prompt del juez que un fragmento que contradice una afirmación de la pregunta también es relevante, y el resultado pasó a ser el esperado:

| Prueba de A.3 | Chain avanzado |
|---|---|
| Factual directa | Elige los 2 chunks de `RUTA-MAD-FCO` (8 y 10/10); los otros 6 candidatos reciben 0 |
| Fuera de dominio | Los 8 candidatos reciben 0/10: frase de escape sin invocar al generador |
| Ataque de complacencia | `RUTA-OSL-TLL#chunk1` con 10/10; desmiente el vuelo directo y además usa el precio real (mediana de 132€ y 140€), algo que la Parte A no lograba |
| Uso de sinónimos | 3 chunks de `RUTA-BCN-MAD` (10, 9 y 8/10) |

La Parte A usaba solo la distancia coseno como señal de relevancia; el juez agrega una lectura del contenido que distingue "habla de la misma ciudad" de "responde la pregunta". Por eso en las cuatro pruebas la mayor parte de los candidatos recibe 0 aunque estén a menos de 0.50 de distancia (por ejemplo, `RUTA-BCN-FCO#chunk1` a 0.401 en la consulta Roma–Madrid): es el ruido que la nota de la Parte A anticipaba.

### B.4 — Captura de traza (LangSmith)

**Cómo se activó.** La observabilidad de LangChain se configura solo con variables de entorno, sin tocar el chain. En el `.env` (que está en `.gitignore`) se definen `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` (clave personal de LangSmith) y `LANGSMITH_PROJECT=smart-flight-assistant-entrega3`; en `.env.example` quedan solo los nombres, sin valores. `rag_pipeline.py` no necesita código extra para que se registre la traza: `python entrega_3/rag_pipeline.py --parte b4` solo verifica que el tracing esté activo, corre el chain avanzado de B.3 sobre una consulta con un nombre de ejecución propio (`B4 RAG avanzado (chunking + reranking)`) y etiquetas, e imprime el enlace a la traza. Los pasos del reranking llevan `run_name` (`juez_llm`, `seleccion_top_n`) para que se distingan en el árbol.

**La consulta trazada:** *"¿En qué ruta solo se ofrece clase económica?"*, una de las que el RAG básico respondía *"No poseo información..."* y que en B.2 y B.3 pasó a responderse bien.

**Evidencia de la ejecución.** Traza pública: <https://smith.langchain.com/public/308364e8-c7d5-4c6c-a864-d6be91d342b0/r>. Captura en el repo: [`langsmith_trace.png`](langsmith_trace.png). Muestra el árbol de pasos con el tiempo de cada uno (`ChunksRetriever` 1,29 s, `juez_llm` 1,85 s, `seleccion_top_n` 0,00 s, generación 0,75 s), los tokens de las dos llamadas al LLM (1.248 y 285) y, en la salida, `candidatos` (8 ítems) frente a `context` (1 solo documento, el seleccionado tras el reranking).

**Tiempos y tokens** (leídos de la propia traza, ejecución completa de 3,93 s y 1.533 tokens):

| Paso de la traza | Tipo | Tiempo | Tokens (entrada + salida) |
|---|---|---|---|
| `ChunksRetriever` (k = 8) | retriever | 1,29 s | — (incluye el embedding de la consulta) |
| `juez_llm` (`ChatOpenAI` con salida estructurada) | llm | 1,85 s | 1.092 + 156 = **1.248** |
| `seleccion_top_n` | código | 0,00 s | — |
| generación (prompt → `ChatOpenAI` → parser) | llm | 0,75 s | 262 + 23 = **285** |
| **Total** | | **3,93 s** | **1.354 + 179 = 1.533** |

El juez concentra el costo del reranking: 1.248 de los 1.533 tokens (81 %) y casi la mitad del tiempo (1,85 de 3,93 s). Es el precio de B.3: una llamada extra por consulta a cambio de que el contexto final sea más corto y preciso. La generación, que recibe un solo chunk, consume apenas 285 tokens.

**Recuperados vs. seleccionados tras el reranking** (los mismos datos que muestra la traza en la entrada y la salida de `juez_llm` y `seleccion_top_n`):

| # | Chunk recuperado | Distancia coseno | Puntaje del juez | ¿Seleccionado? |
|---|---|---|---|---|
| 1 | `RUTA-MAD-FCO#chunk3` | 0,153 | 10 | **Sí** |
| 2 | `RUTA-OSL-TLL#chunk2` | 0,485 | 0 | — |
| 3 | `RUTA-BCN-FCO#chunk1` | 0,534 | 0 | — |
| 4 | `RUTA-DUB-WAW#chunk2` | 0,535 | 0 | — |
| 5 | `RUTA-BCN-FCO#chunk2` | 0,550 | 0 | — |
| 6 | `RUTA-BCN-MAD#chunk2` | 0,557 | 0 | — |
| 7 | `RUTA-CIA-CRL#chunk1` | 0,558 | 0 | — |
| 8 | `RUTA-AYT-MMX#chunk2` | 0,564 | 0 | — |

De los 8 candidatos queda 1. La respuesta final es *"En la ruta Madrid - Roma (FCO) solo se ofrece clase económica [RUTA-MAD-FCO]"*, correcta y con la fuente citada.

**Qué muestra la traza.** Si solo se filtrara por el umbral de distancia de C.2 (0,50), un chunk como `RUTA-OSL-TLL#chunk2` (a 0,485) pasaría al contexto del generador por estar "lo bastante cerca", aunque no tiene nada que ver con la pregunta. El juez lo puntúa con 0 y el código lo descarta: es la diferencia entre filtrar por cercanía vectorial y filtrar por si el fragmento realmente responde.

---

## Parte C — Evaluación con RAGAS

### C.1 — Golden dataset

El dataset está en [`golden_dataset.json`](golden_dataset.json), separado del código de evaluación para poder reutilizarlo en las Entregas 4 y 5; `evaluacion_ragas.py` lo carga desde ese archivo al ejecutarse. Tiene 10 preguntas del dominio con su respuesta esperada (*ground truth*), distribuidas como pide la consigna:

| ID | Tipo | Pregunta | Ruta(s) que la responden |
|---|---|---|---|
| S1 | Simple | ¿Cuánto dura el vuelo entre Guernsey y Jersey? | `RUTA-GCI-JER` |
| S2 | Simple | ¿Qué aerolínea domina la ruta Londres–Zúrich? | `RUTA-LHR-ZRH` |
| S3 | Simple | ¿Cuánto dura el vuelo directo entre Barcelona y Madrid? | `RUTA-BCN-MAD` |
| C1 | Compleja | Para ir a Roma, ¿es más barato salir desde Madrid o desde Barcelona? | `RUTA-BCN-FCO` + `RUTA-MAD-FCO` |
| C2 | Compleja | ¿Las rutas Oslo–Tallin y Dublín–Varsovia tienen vuelo directo? ¿Cuál es más barata? | `RUTA-OSL-TLL` + `RUTA-DUB-WAW` |
| C3 | Compleja | En la ruta Londres–Lisboa, ¿qué porcentaje es directo, qué aerolíneas la operan y en qué sentido es más cara? | `RUTA-LHR-LIS` (varios chunks) |
| E1 | Escape | ¿Cuánto sale un vuelo de Nueva York a Sídney? | — (fuera del catálogo) |
| E2 | Escape | ¿Cuánto cuesta despachar una valija extra en la ruta Madrid–Roma? | — (la ruta existe, el dato no) |
| I1 | Informal | che, ¿cuál es el vuelo más barato que tienen para ir de Milán a París? | `RUTA-BGY-BVA` |
| I2 | Informal | tengo una reunión en Madrid y vuelvo en el día, salgo de Barcelona, ¿qué onda los vuelos? | `RUTA-BCN-MAD` |

**Criterios de diseño.**

- **El ground truth sale literalmente de las fichas.** Context Recall compara el *ground truth* contra el contexto recuperado: si la respuesta esperada tuviera datos que no están en la base, la métrica bajaría por un error del dataset y no del sistema.
- **Ninguna pregunta repite las de A.3 ni las de B.1.** Esas consultas se usaron para ajustar el pipeline (el orden de las reglas del prompt, el umbral, el prompt del juez); evaluar con ellas inflaría las métricas.
- **Las simples tienen la respuesta en un solo chunk.** Se verificó sobre la colección de chunks de B.2: la respuesta de S1, S2 y S3 está entera en un único fragmento.
- **Las complejas combinan chunks de dos maneras:** C1 y C2 necesitan dos fichas distintas; C3 necesita datos de dos fragmentos de la misma ficha (el porcentaje de directos y las aerolíneas están en `RUTA-LHR-LIS#chunk1`, los precios por sentido en `#chunk2`).
- **Las dos de escape prueban dos defensas distintas.** E1 está fuera del catálogo: en el básico la corta el umbral de distancia, y en el avanzado (que recupera 8 candidatos sin umbral) el juez les pone menos de 6 a todos; en los dos casos el contexto queda vacío y se responde la frase de escape sin invocar al generador. E2 nombra una ruta que existe, así que la ficha se recupera, pero el dato (equipaje) no está: ahí la que tiene que escapar es la regla del prompt. Su *ground truth* es la frase de escape exacta del pipeline.
- **Las informales usan registro coloquial** ("che", "qué onda", "vuelvo en el día") sin nombrar los datos que se piden. I2 se parece a la Killer Query 1 de la Entrega 2, que falló por el umbral, pero acá ambos retrievers encuentran `RUTA-BCN-MAD` en primer lugar. Su dificultad es otra: la respuesta esperada usa datos de los tres chunks de esa ficha, y el retriever de chunks (k = 4) no recupera el tercero (horarios y días de mayor demanda). Es un caso donde el chunking podría bajar el Context Recall en lugar de subirlo. (En C.3 se ve que el pipeline avanzado, con 8 candidatos y el juez, sí lo recupera.)


### C.2 — Baseline RAGAS (RAG básico)

Código: [`evaluacion_ragas.py`](evaluacion_ragas.py). Se corre con `python entrega_3/evaluacion_ragas.py` (o `--pipeline basico` para correr solo el baseline). Resultados completos, con la respuesta y las fuentes de cada pregunta, en [`resultados_ragas.json`](resultados_ragas.json).

**Cómo se evalúa.** Cada pregunta del golden dataset se pasa por el chain de la Parte A (`rag_chain`: fichas completas, k = 4, umbral 0.50), y la respuesta, los documentos recuperados y el *ground truth* se evalúan con las cuatro métricas de RAGAS 0.4.3 (`ragas.metrics.collections`), usando `gpt-4o-mini` como evaluador y `text-embedding-3-small` para Answer Relevancy. Siguiendo la teoría de la cátedra, dos métricas evalúan la **generación** (Faithfulness: ¿la respuesta está respaldada por el contexto?; Answer Relevancy: ¿responde la pregunta sin divagar?) y dos la **recuperación** (Context Precision: ¿los primeros chunks son útiles?; Context Recall: ¿el contexto tiene toda la evidencia necesaria?). Las versiones quedan fijadas en `requirements.txt`: RAGAS 0.4.3 necesita `langchain-community==0.4.1` (la 0.4.2 ya no trae un módulo que RAGAS importa) y `openai==1.109.1` (las versiones 2.x y 3.x son incompatibles con `instructor`, una dependencia de RAGAS).

**Dos decisiones de medición.**

- **Contexto vacío = "no aplica".** En E1 el contexto llega vacío (en el básico lo corta el umbral de distancia; en el avanzado, el juez). Faithfulness, Context Precision y Context Recall no se pueden calcular sin contexto (RAGAS da error), así que se registran como `n/a` en lugar de inventar un valor.
- **Dos promedios.** Answer Relevancy le pone 0 a toda respuesta evasiva ("no sé"), así que la frase de escape correcta igual baja el promedio. Por eso se reporta el promedio sobre las 10 preguntas y sobre las 8 que tienen respuesta.

**Resultados por pregunta (básico):**

| ID | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| S1 | 0.500 | 0.962 | 1.000 | 1.000 |
| S2 | 1.000 | 0.993 | 1.000 | 1.000 |
| S3 | 1.000 | 0.926 | 1.000 | 1.000 |
| C1 | 1.000 | 0.835 | 0.583 | 1.000 |
| C2 | 1.000 | 0.848 | 1.000 | 1.000 |
| C3 | 1.000 | 0.745 | 1.000 | 1.000 |
| E1 | n/a | 0.000 | n/a | n/a |
| E2 | 0.000 | 0.000 | 0.000 | 0.000 |
| I1 | 0.833 | 0.789 | 1.000 | 1.000 |
| I2 | 1.000 | 0.580 | 1.000 | 1.000 |

**Lectura.** Las 10 respuestas del básico son correctas: ninguna inventa datos y las dos de escape responden la frase de escape. Los puntajes bajos tienen explicaciones puntuales:

- **E2 (0 en todo):** la respuesta es la correcta ("No poseo información…"), pero se recuperaron las fichas de Madrid–Roma y Barcelona–Roma, que no hablan de equipaje. RAGAS evalúa una respuesta de "no sé" contra un contexto que habla de otra cosa, y las métricas pierden sentido: no miden un error del sistema. Por eso el análisis se apoya en el promedio sin las preguntas de escape.
- **S1 (Faithfulness 0.500):** la respuesta ("dura apenas 25 minutos cuando es directo") está escrita literalmente en la ficha. Es variabilidad del LLM evaluador, no un error del sistema.
- **C1 (Context Precision 0.583):** el retriever pasó 4 fichas en este orden: `RUTA-MAD-FCO`, `RUTA-BCN-FCO`, `RUTA-BCN-MAD`, `RUTA-CIA-CRL`. La teoría define Context Precision como si "los fragmentos más útiles y relevantes aparecen en los primeros lugares del ranking": RAGAS promedia la precisión en cada posición donde hay un fragmento útil. Con 4 fragmentos, el único patrón de veredictos que da 0.583 es *no útil, útil, útil, no útil*: el evaluador consideró no útil a `RUTA-MAD-FCO`, en el primer puesto, aunque su mediana de 150€ es parte del *ground truth*. Hay ruido real en el contexto (`RUTA-CIA-CRL` y `RUTA-BCN-MAD` no sirven para comparar precios a Roma), pero el puntaje exacto también refleja un veredicto discutible del evaluador.

### C.3 — Comparativa: RAG básico vs. RAG avanzado

El RAG avanzado es `rag_chain_avanzado` de la Parte B: chunks de 500/100, k = 8 candidatos, LLM juez y top 3. Se evaluó con el mismo dataset, las mismas métricas y el mismo evaluador.

**Resultados por pregunta (avanzado):**

| ID | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| S1 | 1.000 | 0.977 | 1.000 | 1.000 |
| S2 | 1.000 | 0.991 | 1.000 | 1.000 |
| S3 | 1.000 | 1.000 | 1.000 | 1.000 |
| C1 | 0.750 | 0.769 | 1.000 | 1.000 |
| C2 | 0.750 | 0.832 | 1.000 | 1.000 |
| C3 | 1.000 | 0.745 | 1.000 | 1.000 |
| E1 | n/a | 0.000 | n/a | n/a |
| E2 | 0.000 | 0.000 | 0.000 | 1.000 |
| I1 | 1.000 | 0.789 | 1.000 | 1.000 |
| I2 | 1.000 | 0.538 | 1.000 | 1.000 |

**Tabla comparativa de promedios:**

| Métrica | Básico (10) | Avanzado (10) | Básico (8, sin escape) | Avanzado (8, sin escape) | Delta (8) |
|---|---|---|---|---|---|
| Faithfulness | 0.815 | 0.833 | 0.917 | 0.938 | +0.021 |
| Answer Relevancy | 0.668 | 0.664 | 0.835 | 0.830 | −0.005 |
| Context Precision | 0.843 | 0.889 | 0.948 | **1.000** | **+0.052** |
| Context Recall | 0.889 | 1.000 | 1.000 | 1.000 | 0.000 |

(En las columnas de 10 preguntas, Faithfulness, Context Precision y Context Recall promedian 9 valores, porque E1 no tiene contexto.)

**La mejora: Context Precision (+0.052).** Es la métrica que el reranking tenía que mover. En el avanzado, las 8 preguntas con respuesta tienen 1.000; en el básico, todas menos C1, así que toda la diferencia sale de esa pregunta. El básico pasaba 4 fichas con ruido (ver C.2); el avanzado pasa solo 3 chunks (`BCN-FCO#chunk1`, `MAD-FCO#chunk2`, `BCN-FCO#chunk2`), los tres útiles. El juez descarta lo que está "cerca" en el espacio vectorial pero no responde la pregunta. La mejora es consistente con el diseño del reranking, pero descansa en una sola pregunta y, en parte, en un veredicto discutible del evaluador sobre el básico, así que su magnitud hay que leerla con cautela.

**Lo que no es una mejora real.**

- **Context Recall (de 0.889 a 1.000 sobre 10):** toda la diferencia viene de E2, que pasa de 0 a 1.000 con el mismo *ground truth*; lo único que cambia es el contexto recuperado (las fichas completas de Madrid–Roma y Barcelona–Roma en el básico, `RUTA-MAD-FCO#chunk2` en el avanzado). Es el mismo problema de las preguntas de escape descripto en C.2: con un *ground truth* que dice "no hay información", la métrica no tiene qué buscar. Sin las preguntas de escape, los dos pipelines dan 1.000: en este dataset, el básico ya recuperaba toda la evidencia necesaria.
- **Faithfulness (+0.021):** está dentro de la variabilidad del evaluador. El básico pierde por S1 (0.500, ver C.2) e I1 (0.833), y el avanzado por C1 y C2 (0.750), donde el evaluador no da por respaldada la conclusión comparativa ("por lo tanto, Oslo–Tallin es más barata"), que es una inferencia correcta a partir de dos precios del contexto, pero no está escrita literal. Ninguno de los dos pipelines inventó datos.

**I2: el chunking no perdió información, gracias al reranking.** En C.1 se anticipó que el retriever de chunks (k = 4) no recuperaba `RUTA-BCN-MAD#chunk3`, donde están los horarios y los días de mayor demanda. El avanzado sí lo recupera: al traer 8 candidatos y dejar que el juez elija, entran los tres chunks de la ficha. Chunking y reranking se complementan: el chunking parte la ficha y el reranking vuelve a juntar las partes que la pregunta necesita.

### C.4 — Diagnóstico de la peor métrica y propuesta de mejora

**La peor métrica del avanzado: Answer Relevancy** (0.664 sobre 10 preguntas, 0.830 sin las de escape). Es además la única que empeora respecto del básico, aunque por una diferencia (−0.005) que está dentro del ruido.

**Causa probable.** La teoría de la cátedra define Answer Relevancy como si la respuesta "contesta directamente a la pregunta formulada sin divagar". RAGAS la calcula generando preguntas a partir de la respuesta y comparándolas, por similitud de embeddings, con la pregunta original. Hay dos causas, de peso distinto:

1. **Estructural, no corregible desde el sistema: las preguntas de escape.** E1 y E2 sacan 0 porque RAGAS trata cualquier respuesta evasiva como no relevante, aunque escapar sea lo correcto. Explican la caída de 0.830 a 0.664.
2. **La que sí hay que corregir: el generador divaga.** Sin las preguntas de escape, los puntajes más bajos son I2 (0.538), C3 (0.745), C1 (0.769) e I1 (0.789). En I2 ("¿qué onda los vuelos?") la respuesta agrega datos que nadie pidió, y en I1 suma la asimetría de vuelos por sentido. El prompt de la Parte A pide responder solo con información del contexto, pero no pide limitarse a lo preguntado, así que el modelo vuelca todo lo que encuentra en los chunks. Las preguntas generadas a partir de esas respuestas largas se alejan de la pregunta original. El registro informal agrava el efecto: "qué onda los vuelos" queda lejos, en el espacio de embeddings, de cualquier pregunta formal que se genere a partir de la respuesta.

**Acción técnica concreta.** Agregar al prompt de generación (`template` en `rag_pipeline.py`) una regla de foco: *"Responde primero, en una oración, exactamente lo que se pregunta. Agrega otros datos del contexto solo si son necesarios para esa respuesta."* Después, volver a correr `evaluacion_ragas.py` y comparar Answer Relevancy sin las preguntas de escape, verificando que Faithfulness no baje (la regla no debe empujar al modelo a inferir datos que no estén en el contexto). Para las preguntas de escape, que Answer Relevancy no puede medir por diseño, conviene evaluarlas con un criterio propio: que la respuesta sea exactamente `FRASE_ESCAPE`. El dataset se reutiliza en las Entregas 4 y 5, así que esa comparación va a poder repetirse sobre el mismo set.

**Limitación de la evaluación.** Con 10 preguntas, una sola respuesta mueve un promedio entre 0.05 y 0.1, y el evaluador (`gpt-4o-mini`) tiene variabilidad propia (ver S1 en C.2). Por eso las diferencias chicas, como la de Faithfulness, no deberían leerse como mejoras, y la de Context Precision tiene el alcance acotado que se explica en C.3.