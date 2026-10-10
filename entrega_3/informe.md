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
