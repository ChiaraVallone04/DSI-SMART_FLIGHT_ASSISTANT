# TP Integrador DSI — Entrega 2: Del Prompt Saturado a la Base de Conocimiento Vectorial

**Dominio:** Asistente Virtual Inteligente de Planificación de Vuelos y Optimización de Itinerarios (*Smart Flight Assistant*). Se mantiene el mismo dominio definido en la Entrega 1; esta entrega construye la capa que faltaba, la Base de Conocimiento, migrando de un prompt con contexto estático a una arquitectura de recuperación vectorial.

---

## Parte A — Embeddings y Búsqueda Semántica (Clase 4)

### A.1 — Autopsia del contexto estático

Se documentan a continuación los tres problemas de meter toda la Base de Conocimiento directamente en el System Prompt, aplicados al dominio del Smart Flight Assistant.

| Problema | Aplicado al dominio |
|---|---|
| **Desangre de tokens** | La base construida en A.3 tiene **18 documentos**. |
| **Lost in the Middle** | Con 18 documentos ya es un riesgo real; a escala de catálogo, sería directamente inutilizable. |
| **Inconsistencia de estado concurrente** | El precio y la disponibilidad de un vuelo cambian constantemente, incluso dentro de la misma sesión de scraping. |

**Desangre de tokens.** Para medir el costo real (no una estimación) de mandar la base completa en cada consulta, se midieron con `tiktoken` (mismo método usado en A.4 de la Entrega 1, sobre el tokenizador de `gpt-4o`) los 18 documentos completos de `base_conocimiento.json` (texto + metadatos). El resultado: **5.345 tokens** solo por la base. Sumado al System Prompt actual de la Entrega 1 (211 tokens), se estarían mandando **5.556 tokens de contexto fijo en cada consulta**, antes de que el usuario escriba una sola palabra de su pregunta. Y esto es con apenas 18 documentos: si se quisiera cubrir las 4.147 rutas reales que existen en el dataset completo (`europe_flights_google_prices.csv`), a un promedio de ~297 tokens por documento, serían **~1,23 millones de tokens por consulta**, inviable tanto en costo como en el límite de contexto del modelo. Lo más grave es que el 94% de esos 18 documentos (17 de 18) resultan irrelevantes para cualquier pregunta puntual del usuario, y se pagarían igual, en cada consulta, todo el tiempo.

**Lost in the Middle.** Si los 18 documentos se insertaran en el orden en que están en `base_conocimiento.json`, uno como `RUTA-BRE-FRA` (posición 9 de 18) queda enterrado justo en el medio de un bloque de ~5.300 tokens. Es exactamente el fenómeno descripto en la literatura de "lost in the middle": los LLM prestan más atención a lo que está al principio y al final del contexto, y degradan su capacidad de usar correctamente la información que cae en el medio de un bloque largo. Con 18 documentos el riesgo ya es real; con cientos o miles sería directamente inutilizable, el modelo podría "ver" la ruta Bremen-Fráncfort en su contexto y aun así no usarla para responder, simplemente por dónde quedó ubicada.

**Inconsistencia de estado concurrente.** Se buscó evidencia directa de esto en el propio CSV, en vez de asumirlo en abstracto. En una muestra de 400.000 filas se encontraron **673 casos donde el mismo vuelo** (misma aerolínea, mismo número de vuelo, mismo horario de salida — la clave primaria definida para la tabla `vuelos` en la Entrega 1) **aparece con dos precios distintos** registrados casi al mismo instante de scraping. Hay un caso todavía más simple y determinista: la columna `days_left` baja exactamente en 1 cada día que pasa, para cada vuelo futuro, sin ninguna excepción — cualquier dato que se "congele" hoy sobre "faltan 14 días para este vuelo" queda mal mañana. Si esa información viviera embebida en un párrafo estático de la Base de Conocimiento, el vector nunca se entera de que cambió — queda desactualizado desde el minuto uno, el mismo problema de fondo ya diagnosticado en A.2 de la Entrega 1 con la alucinación de precios del LLM genérico.

**Cierre — ¿por qué un `SELECT ... WHERE descripcion LIKE '%...%'` tampoco alcanza?**
Porque `LIKE` compara texto literal, no significado: si el usuario escribe "quiero una escapada barata a Italia" y ningún documento contiene exactamente esas palabras (dicen "Roma", "low-cost", "mediterráneo"), el `LIKE` no encuentra nada aunque el documento correcto exista. Tampoco resuelve el desangre de tokens ni el lost in the middle — seguiría siendo necesario decidir qué mandar y en qué orden. Y a diferencia de un embedding, `LIKE` no tiene noción de "cuán parecido" es un resultado a otro para priorizar: es una coincidencia binaria (sí/no), no un ranking por relevancia semántica.

---

### A.2 — Similitud coseno a mano

**Los dos ejes elegidos.** En vez de ejes abstractos, se usaron dos de los metadatos ya definidos en A.3 — así el ejercicio queda conectado con la base real en vez de ser un cálculo aislado:

- **Eje X — Precio** (`categoria_precio` escalada a 0-10): económico → 2, medio → 5, premium → 9.
- **Eje Y — Comodidad** (% de vuelos directos ÷ 10): de 0 (siempre con escala) a 10 (siempre directo).

**Los vectores (documentos reales de la base).**

| Vector | Ruta | Precio (X) | % Directo (Y) |
|---|---|---|---|
| **A** | RUTA-BCN-FCO (económico, 99,2% directo) | 2 | 9,9 |
| **B** | RUTA-KEF-MAD (premium, 0% directo) | 9 | 0,0 |
| **C** | RUTA-MAD-FCO (medio, 97,7% directo) | 5 | 9,8 |
| **Q** (consulta) | "algo barato y sin escalas" | 1 | 10,0 |

**Cálculo a mano — los 3 pasos (ejemplo completo: Q vs. A).**

$$\text{Similitud} = \frac{A \cdot B}{\|A\| \times \|B\|}$$

1. **Producto punto:** $Q \cdot A = (1)(2) + (10)(9{,}9) = 2 + 99 = 101$
2. **Normas:** $\|Q\| = \sqrt{1^2+10^2} = \sqrt{101} = 10{,}05$ — $\|A\| = \sqrt{2^2+9{,}9^2} = \sqrt{102{,}01} = 10{,}10$
3. **División:** $\dfrac{101}{10{,}05 \times 10{,}10} = \dfrac{101}{101{,}5} = \mathbf{0{,}995}$

Resultado de las 3 comparaciones (a mano y validado con NumPy):

| Comparación | Producto punto | Normas | Similitud |
|---|---|---|---|
| Q vs. A (BCN-FCO) | 101 | 10,05 × 10,10 | **0,995** |
| Q vs. C (MAD-FCO) | 103 | 10,05 × 11,00 | **0,932** |
| Q vs. B (KEF-MAD) | 9 | 10,05 × 9,00 | **0,100** |

**Validación con NumPy** (misma función que da la consigna):

```python
import numpy as np

def similitud_coseno(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

A = np.array([2, 9.9]);  B = np.array([9, 0.0]);  C = np.array([5, 9.8])
Q = np.array([1, 10.0])

similitud_coseno(Q, A)  # 0.9950
similitud_coseno(Q, B)  # 0.0995
similitud_coseno(Q, C)  # 0.9316
```

Los tres resultados coinciden con el cálculo manual, y tienen sentido con la intuición del dominio: la consulta ("barato y sin escalas") es casi idéntica en dirección al vector de BCN-FCO (0,995 — ambos apuntan fuerte hacia "muy directo, poco caro"), bastante parecida a MAD-FCO (0,932 — también muy directo pero de precio medio), y casi ortogonal a KEF-MAD (0,100 — apunta para el lado contrario: caro y con escala).

**Reflexión — el umbral de aceptación.**
Con estos números, un umbral razonable estaría en algo como **0,75-0,80**: por debajo de eso, la dirección del vector ya no comparte lo suficiente con lo que pidió el usuario como para considerarlo una respuesta válida — a 0,10 (el caso de KEF-MAD) el sistema literalmente estaría devolviendo lo opuesto de lo que se buscó. Si ninguna coincidencia supera ese umbral, el sistema no debe forzar el resultado más parecido igual — eso es alucinación por sustitución: mostrar el mejor de los peores como si fuera bueno. Lo correcto es que responda algo del estilo "no tengo una ruta que coincida con eso en el catálogo", que es exactamente lo que se pone a prueba después con el Killer Query #3 de B.6.

**Un límite real, encontrado al probar con más consultas y las 18 rutas completas.**
Al ampliar las pruebas más allá del ejemplo principal (corriendo varias consultas contra las 18 rutas de la base, no solo contra A, B y C) aparecieron dos problemas que vale la pena documentar en vez de esconder:

1. **Rutas totalmente distintas colapsan en el mismo vector.** `RUTA-BRE-FRA`, `RUTA-KEF-MAD`, `RUTA-DUB-TIA`, `RUTA-MSQ-RIX` y `RUTA-AYT-MMX` — Alemania, Islandia, Albania, Bielorrusia y la ruta más cara del catálogo (2.761€) — dan **1,0000 de similitud entre sí** frente a cualquier consulta, porque con solo 2 ejes de 3 valores posibles cada uno todas caen en el mismo punto `(9, 0)`.
2. **Un falso positivo perfecto.** La consulta "lujo, sin escalas" (`Q=(9,10)`) obtuvo su similitud más alta (**1,0000**) no con una ruta premium, sino con `RUTA-CDG-MRS` — precio medio y apenas 55,6% directa. Esto ocurre porque la similitud coseno mide **ángulo, no magnitud**: `(5, 5,6)` y `(9, 10)` apuntan casi exactamente para el mismo lado aunque estén en escalas muy distintas, así que el coseno los trata como "iguales" pese a que uno es mediocre en los dos ejes y el otro sería excelente en ambos.

Esto no invalida el ejercicio — al contrario, lo justifica. No es un error de cálculo: es una propiedad matemática real de la similitud coseno (invariante a la escala del vector), agravada acá por usar solo 2 ejes muy toscos hechos a mano. Es la evidencia que respalda dos decisiones de diseño que vienen más adelante: primero, por qué A.4 usa un embedding real de 1.536 dimensiones en vez de ejes armados a mano — con mucha más resolución, este tipo de colisión es mucho menos probable; y segundo, por qué el filtro de metadatos de B.4 no es opcional — si la búsqueda semántica sola puede confundir una ruta mediocre con una de lujo, un umbral de similitud alto no alcanza para blindarse: hace falta el filtro exacto (`categoria_precio`, `vuelo_directo_disponible`) como red de seguridad, el mismo argumento detrás del Killer Query #2 ("el metadato salva el día").

---

### A.3 — `base_conocimiento.json`: documentos y justificación del esquema

Código completo: [`base_conocimiento.json`](base_conocimiento.json) — **18 documentos** (mínimo pedido: 15).

**Qué representa cada documento.** Cada registro es el perfil de una **ruta** (un par de aeropuertos, ej. Madrid–Roma), no una fila individual del CSV ni un país. Se descartó "una fila = un documento" porque el dataset tiene 1.112.738 filas transaccionales (precio puntual de un vuelo puntual) — vectorizarlas todas repite el mismo error que A.1 diagnostica (desangre de tokens) y mezcla en la búsqueda semántica datos que son responsabilidad del SQL, no de la Base de Conocimiento. También se descartó "un país = un documento": España sola tiene 30 aeropuertos en el dataset repartidos en tres bloques con perfiles de viaje muy distintos (Península, Baleares, Canarias), así que agrupar por país mezclaría en un mismo párrafo viajes de negocios cortos con turismo de playa de larga distancia, perdiendo la coherencia semántica que necesita un embedding. El nivel "ruta" (par de aeropuertos) es el que ya usaba la Matriz de Intenciones de la Entrega 1 (`origen`, `destino` como códigos IATA), así que además queda directamente alineado con C.1.

Cada documento cubre **ambas direcciones de la ruta** (ej. Madrid→Roma y Roma→Madrid) con un solo vector, en vez de duplicarlo: los datos agregados (aerolíneas, % directo, rango de precio) surgen de mezclar las filas de los dos sentidos, y separarlos generaría dos párrafos casi idénticos entre sí — exactamente el tipo de casi-duplicado que B.5 pide luego detectar y purgar. La dirección se resuelve del lado de la consulta (B.4), no del documento: el filtro de metadatos arma un `$or` con las dos combinaciones posibles de `origen`/`destino`, así que no se pierde cobertura por guardar un solo sentido fijo. En los 3 casos donde el catálogo realmente **no tiene** vuelos registrados en el sentido inverso (`RUTA-CIA-CRL`, `RUTA-AYT-MMX`, y casi por completo `RUTA-BGY-BVA`), la `descripcion_semantica` lo aclara explícitamente en vez de fingir una simetría que los datos no respaldan.

**Justificación del esquema — la Regla del Arquitecto.** La consigna es clara: todo campo sobre el que haya que aplicar un filtro duro va como metadato; lo narrativo o sensorial queda dentro del texto, nunca al revés. Así se aplicó en cada documento:

| Campo | Dónde vive | Por qué |
|---|---|---|
| `origen`, `destino` (códigos IATA) | Metadato | Es exactamente el filtro duro que ya usaba la Matriz de la Entrega 1 (B.3) — un usuario que busca "vuelos MAD→FCO" necesita coincidencia exacta, no aproximada por significado. |
| `pais_origen`, `pais_destino` | Metadato (categórico) | Permite filtrar "todo lo que sea a Italia" sin depender de que el usuario haya escrito el nombre de la ciudad. |
| `vuelo_directo_disponible` | Metadato (booleano de estado) | Es blanco o negro — no admite matices — y es el filtro que el Killer Query "el metadato salva el día" (B.6) necesita para descartar rutas con escala aunque el texto las mencione. |
| `categoria_precio` (económico / medio / premium) | Metadato (categórico) | Se calculó a partir de la mediana real de cada ruta (económico <100€, medio 100-250€, premium >250€) para poder filtrar por rango de precio sin tener que interpretar el número dentro del párrafo. |
| `tipo_aerolinea_dominante` (low-cost / tradicional / mixto) | Metadato (categórico) | Mismo criterio: es un dato que conviene poder filtrar exacto ("solo low-cost"), no inferir por similitud semántica. |
| `tags_regionales` | Metadato (lista) | Es la jerga y los sinónimos con los que la gente pregunta en la práctica ("escapada a Roma", "finde económico") — se separa del párrafo principal para reforzar el matching semántico sin inflar la narrativa. |
| Aerolíneas que cubren la ruta, % directo real, contexto de por qué es popular, asimetría de precio por dirección, curiosidades del catálogo | `descripcion_semantica` (texto) | Es lo narrativo/interpretativo: no hay un `WHERE` que capture "por qué esta ruta es popular para escapadas de finde" — es exactamente lo que un embedding sabe comparar por significado, y lo que un filtro exacto no puede expresar. |

Los datos detrás de cada documento (mediana y rango de precio, % de vuelos directos, aerolíneas dominantes con su porcentaje real, duración típica) se calcularon agrupando las filas reales del dataset por par de aeropuertos — no se inventó ningún número — lo cual cumple el requisito de "documentos reales o creíbles" de la consigna.

---

### A.4 — Índice FAISS (`pipeline_vectorial.py`)

Código completo: [`pipeline_vectorial.py`](pipeline_vectorial.py).

**Qué hace el script.** Lee `OPENAI_API_KEY` desde `.env` (nunca hardcodeada, con `load_dotenv(override=True)` para que el `.env` del proyecto no quede pisado por una variable de entorno del sistema — problema real que apareció al probarlo). Genera embeddings de las 18 `descripcion_semantica` de `base_conocimiento.json` con `text-embedding-3-small` (1.536 dimensiones), normaliza los vectores con `faiss.normalize_L2` y los carga en un `IndexFlatIP`: producto interno sobre vectores normalizados es matemáticamente equivalente a similitud coseno, la misma métrica que ya se usó a mano en A.2 y que va a usar ChromaDB en B.1 (`hnsw:space: cosine`) — se mantiene un criterio de similitud consistente en las tres partes del TP. El índice se persiste con `faiss.write_index()` en `faiss_index/index.faiss`, junto con un `ids.json` que guarda el orden de los IDs (para poder mapear cada posición del índice de vuelta a un documento, algo que FAISS no guarda por sí solo). Al arrancar, si el índice ya existe en disco y sus IDs coinciden con los de `base_conocimiento.json`, se recarga con `faiss.read_index()` sin volver a llamar a la API.

**Resultado de las 3 consultas de prueba (top-3, similitud coseno):**

| Consulta | # | Ruta | Similitud |
|---|---|---|---|
| "quiero una escapada barata y directa a Italia" | 1 | RUTA-BCN-FCO | 0,5727 |
| | 2 | RUTA-MAD-FCO | 0,4936 |
| | 3 | RUTA-BGY-BVA | 0,4269 |
| "busco un vuelo de lujo, sin escalas, no me importa el precio" | 1 | RUTA-OSL-TLL | 0,4722 |
| | 2 | RUTA-BCN-FCO | 0,4369 |
| | 3 | RUTA-BRE-FRA | 0,4283 |
| "ruta entre Alemania y algún país báltico" | 1 | RUTA-BRE-FRA | 0,5315 |
| | 2 | RUTA-MSQ-RIX | 0,5015 |
| | 3 | RUTA-OSL-TLL | 0,4805 |

Las consultas 1 y 3 devuelven exactamente lo esperado: rutas baratas/directas a Italia, y las dos rutas del catálogo que tocan el Báltico. La consulta 2 expone el mismo tipo de falso positivo que ya se había encontrado en A.2 con ejes hechos a mano — solo que ahora con un embedding real de 1.536 dimensiones, no con 2 ejes toscos: el resultado #1 para "sin escalas" es `RUTA-OSL-TLL`, que según su propia `descripcion_semantica` **nunca tiene vuelo directo**. El embedding capta bien el registro semántico de "lujo" (vocabulario, tono) pero no un hecho puntual y verificable como "0% de vuelos directos" — exactamente el argumento por el cual el filtro de metadatos de B.4 (`vuelo_directo_disponible: true`) no es opcional, ni siquiera con un embedding de calidad: la búsqueda semántica sola puede sonar convincente y estar objetivamente mal.

---

### A.5 — Prueba destructiva: volatilidad de la RAM

Se reprodujeron los dos escenarios que pide la consigna, borrando y recreando `faiss_index/` para simular un reinicio del entorno.

**Sin persistencia (o el entorno se reinició antes de guardar):**
```
$ rm -rf faiss_index
$ python pipeline_vectorial.py
[api] No hay índice en disco — generando embeddings para 18 documentos...
```
El índice no está en disco, así que no hay otra opción que volver a llamar a la API de embeddings para los 18 documentos, se pagan tokens de nuevo aunque el contenido de `base_conocimiento.json` no haya cambiado un solo carácter desde la corrida anterior.

**Con persistencia (`faiss.write_index()` ya se ejecutó):**
```
$ python pipeline_vectorial.py
[disco] Índice recargado desde 'faiss_index\index.faiss' — sin llamadas a la API de embeddings.
```
Misma consulta, mismo resultado, cero llamadas a OpenAI: el índice se reconstruye desde los bytes en disco en milisegundos.

**Reflexión.** En producción, si el servidor se reinicia (deploy, crash, escalado) y el índice solo vivía en RAM, hay que regenerar embeddings de toda la base antes de poder responder la primera consulta — tiempo muerto y costo repetido en cada reinicio, y en un catálogo real (miles de documentos, no 18) ese tiempo de arranque en frío deja de ser trivial. El caso de **dos servidores** es peor todavía si cada uno mantiene su propio índice solo en memoria: no solo se paga el costo de generar embeddings dos veces, sino que nada garantiza que ambos índices queden idénticos si la base se actualizó entre una regeneración y la otra, dos instancias respondiendo con "verdades" ligeramente distintas. Persistir en disco (y, mejor todavía, en un storage compartido entre instancias) es lo que evita que la disponibilidad del servicio dependa de que la RAM de un proceso nunca se reinicie.

---

## Parte B — ChromaDB, Filtrado Híbrido y ETL (Clase 5)

### B.1 — Migración a ChromaDB

Código completo: [`vector_db.py`](vector_db.py).

**Qué hace el script.** El script carga la misma base de A.3 (`base_conocimiento.json`, 18 rutas) en una colección de ChromaDB. Se utiliza `chromadb.PersistentClient(path="chroma_db")`, no `Client()` volátil, de modo que la colección persiste ante un reinicio del proceso. A diferencia del índice de FAISS de A.4, no es necesario mantener un `ids.json` auxiliar: ChromaDB almacena el ID, el documento y los metadatos como un único objeto atómico. La colección se crea con `metadata={"hnsw:space": "cosine"}`, la misma métrica de similitud empleada a mano en A.2 y mediante vectores normalizados en A.4. Para la generación de embeddings se reutiliza el modelo de A.4 (`text-embedding-3-small`, a través de `OpenAIEmbeddingFunction` de ChromaDB), de manera que la colección queda ubicada en el mismo espacio vectorial que el índice de FAISS. La carga se realiza con `coleccion.upsert(...)`, no `add`, para permitir re-ejecutar el script sin duplicar registros cuando `base_conocimiento.json` no cambió.

**Verificación.** El script se ejecutó en cinco corridas consecutivas, con salida idéntica en todas:

```
$ python vector_db.py
Colección 'vuelos_smart_flight_assistant' actualizada: 18 rutas indexadas.
```

`coleccion.count()` se mantuvo en 18 en las cinco corridas: el `upsert` sobrescribió los IDs existentes en lugar de duplicarlos en cada ejecución. El directorio `chroma_db/` generado queda excluido del control de versiones (`.gitignore`); la fuente de verdad continúa siendo `base_conocimiento.json`, desde donde la colección se reconstruye por completo.


### B.2 — Los tres límites de FAISS que ChromaDB resuelve
| Límite de FAISS | Cómo se manifiesta en el Smart Flight Assistant | Cómo lo resuelve ChromaDB |
|---|---|---|
| **1. Sin persistencia transaccional / atomicidad** (Sin guardar cambios en caliente) | FAISS trabaja únicamente en la memoria RAM. Si una aerolínea cambia la disponibilidad o el precio de una ruta, hay que reescribir todo el archivo `.faiss` en disco. Si el servidor se apaga a mitad del proceso, los datos se pierden. | Usa una base de datos interna (**SQLite**). Cada alta, baja o modificación de una ruta se guarda en disco al instante de forma atómica y segura. |
| **2. Sin filtrado híbrido nativo** (Sin filtros exactos combinados) | FAISS solo entiende de "similitud por significado". Si el usuario pide *"rutas a Roma pero SOLO vuelos directos"*, FAISS no puede aplicar el filtro de "vuelo directo" en la búsqueda. Hay que traer muchos resultados y filtrarlos a mano con un `if` en Python. | Permite combinar búsqueda semántica con **filtros duros directos en la consulta** (`where={"vuelo_directo_disponible": True}`). Descarta los datos que no cumplen las reglas duras antes de calcular la similitud. |
| **3. CRUD ineficiente / sin concurrencia** (Modificar o borrar es muy complejo) | En FAISS no existe una forma sencilla de borrar o actualizar un solo documento. Modificar una ruta exige reconstruir casi todo el índice o gestionar arreglos de IDs a mano que suelen romper el código. | Funciona como una base de datos tradicional con comandos nativos (`upsert`, `update`, `delete`). Permite modificar o borrar una ruta específica por su ID en milisegundos. |

### B.3 — Evento de negocio en caliente
Para simular una actualización operativa en tiempo real, se modificó el estado de la ruta **`RUTA-KEF-MAD`** (Reikiavik–Madrid). En la base original esta ruta no tenía vuelos directos y su categoría era `premium`. Se simuló la inauguración de un tramo directo por parte de una aerolínea low-cost, cambiando su categoría a `medio` y habilitando el flag directo.

**¿Por qué upsert y no add ni update?**
**No add**: Si el ID ya existe en ChromaDB, add lanza un error de clave duplicada (IDAlreadyExistsError) y cancela la operación.

**No update**: Si por algún motivo el ID no existiera previamente en la base, update falla al no encontrar el registro a modificar.

**Por qué upsert**: Es una operación atómica e idempotente. Si el documento existe, actualiza su texto, embedding y metadatos en el acto. Si no existe, lo crea. Es la opción más segura para procesar cambios en tiempo real sin romper el flujo de la aplicación.


### Evidencia de Ejecución en Caliente

Se ejecutó `ejecutar_evento_en_caliente()` de `vector_db.py` verificando la inserción/actualización mediante el método `upsert` y la posterior consulta directa a la base de datos vectorial ChromaDB. EL output fue el siguiente

```text
=== Ejecutando Evento en Caliente para ID: RUTA-KEF-MAD ===
 Actualización realizada con éxito en ChromaDB.

=== Estado Verificado en ChromaDB ===
ID: RUTA-KEF-MAD
Documento: Ruta directa inaugurada entre Reikiavik (KEF) y Madrid (MAD). Opción económica e ideal para turismo de auroras boreales y viajes nórdicos sin escalas.
Metadatos actualizados: {'pais_origen': 'Islandia', 'tags_regionales': ['auroras boreales', 'islandia', 'escapada nórdica', 'directo'], 'categoria_precio': 'medio', 'pais_destino': 'España', 'origen': 'KEF', 'tipo_aerolinea_dominante': 'low-cost', 'vuelo_directo_disponible': True, 'destino': 'MAD'}
```

Esta actualización impacta directamente en el almacenamiento persistente de **ChromaDB** (`chroma_db/`), que es el motor consultado en tiempo real por el asistente. 

El archivo estático de origen (`base_conocimiento.json`) conserva el registro histórico inicial (`vuelo_directo_disponible: false`, `categoria_precio: premium`). Esto demuestra la capacidad del sistema para gestionar eventos en caliente (`upsert`) sobre la base vectorial sin requerir una re-vectorización ni modificación del dataset estático original.


### B.4 — CLI de búsqueda híbrida
Se implementó la función `buscar_vuelos()` en `vector_db.py`, la cual integra búsqueda semántica por embeddings (`query_texts`) combinada con filtrado estructurado nativo (`where`) en ChromaDB.

**Construcción nativa del filtro `where`**

Para evitar procesar filtros en Python y optimizar el cálculo de similitud, las restricciones se resuelven nativamente en la base de datos usando `$eq`, `$and` y `$or`; no hay ningún `if` de Python después de la query:

- **Filtro simple:** Si se aplica una sola restricción (ej. `solo_directos`), se envía directamente la condición `{"vuelo_directo_disponible": {"$eq": True}}`.
- **Filtro compuesto (`$and`):** Si se aplican múltiples restricciones (ej. categoría de precio y vuelo directo), se encapsulan bajo el operador nativo:
  ```python
  where_filter = {
      "$and": [
          {"categoria_precio": {"$eq": "medio"}},
          {"vuelo_directo_disponible": {"$eq": True}}
      ]
  }
  ```
- **Ruta en ambos sentidos (`$or`):** como se justificó en A.3, cada documento cubre los dos sentidos de la ruta pero guarda uno solo en `origen`/`destino` — `RUTA-BCN-MAD` figura como BCN→MAD aunque el usuario pregunte por "Madrid y Barcelona". El parámetro `ruta=[A, B]` arma un `$or` con las dos combinaciones posibles, sin importar en qué orden se pasen los códigos:
  ```python
  {"$or": [
      {"$and": [{"origen": {"$eq": "MAD"}}, {"destino": {"$eq": "BCN"}}]},
      {"$and": [{"origen": {"$eq": "BCN"}}, {"destino": {"$eq": "MAD"}}]},
  ]}
  ```
  Este `$or` se combina con los demás filtros dentro de un `$and` externo.

> **Corrección:** una versión anterior armaba `{"$or": [{"origen": A}, {"destino": B}]}`, que mezcla las dos columnas en vez de cubrir las dos direcciones. Para "entre Madrid y Barcelona" dejaba afuera `RUTA-BCN-MAD` (origen BCN, destino MAD) y devolvía en su lugar `RUTA-MAD-FCO`, porque sale de Madrid. Con el filtro de las dos combinaciones, el TEST 3 devuelve las rutas BCN–MAD.

Se ejecutaron las pruebas del script (`python vector_db.py`) obteniendo las siguientes respuestas desde el motor de búsqueda vectorial. Los textos y metadatos están abreviados; `dist` es la distancia coseno de ChromaDB (0 = idéntico). La corrida es sobre la base de B.1, **antes de la purga de B.5**, por eso en los TEST 1 y 3 aparecen dos documentos de la misma ruta (un original y su casi-duplicado):
```text
TEST 1: Búsqueda Semántica + Filtro $or de ambos sentidos (Roma–Madrid, guardada como MAD→FCO)
-> [dist=0.2604] La ruta Madrid–Roma (MAD–FCO) es una de las más transitadas del catálogo, con más de 1.400 vuelos...
   Metadatos: origen=MAD, destino=FCO, categoria_precio=medio, vuelo_directo_disponible=True
-> [dist=0.3190] El trayecto entre Madrid Barajas y Roma Fiumicino posee alta demanda para viajes cortos...
   Metadatos: origen=MAD, destino=FCO, categoria_precio=medio, vuelo_directo_disponible=True

TEST 2: Búsqueda Semántica + Categoría de Precio
-> [dist=0.4592] El trayecto entre Madrid Barajas y Roma Fiumicino posee alta demanda para viajes cortos...
   Metadatos: origen=MAD, destino=FCO, categoria_precio=medio, vuelo_directo_disponible=True
-> [dist=0.4873] La ruta Madrid–Roma (MAD–FCO) es una de las más transitadas del catálogo...
   Metadatos: origen=MAD, destino=FCO, categoria_precio=medio, vuelo_directo_disponible=True

TEST 3: Filtro Combinado ($and nativo con $or de ambos sentidos, categoría y directo)
-> [dist=0.2779] Vuelos entre Barcelona y Madrid, el histórico corredor conocido como puente aéreo...
   Metadatos: origen=BCN, destino=MAD, categoria_precio=medio, vuelo_directo_disponible=True
-> [dist=0.2978] La ruta Barcelona–Madrid (BCN–MAD) es la de mayor volumen de todo el catálogo, con 3.470 vuelos...
   Metadatos: origen=BCN, destino=MAD, categoria_precio=medio, vuelo_directo_disponible=True

TEST 4: Consulta fuera de catálogo (Buenos Aires a Tokio)
-> Sin resultados: ninguno superó el umbral de similitud.
```

Los TEST 1 y 3 validan el `$or`: la consulta se hace en el orden inverso al que está guardado el documento (Roma–Madrid contra MAD→FCO; Madrid–Barcelona contra BCN→MAD) y aun así lo recupera. El TEST 4 muestra el umbral de C.2: la función descarta los resultados con distancia coseno mayor a `UMBRAL_DISTANCIA` (0.50), y como ningún vecino lo supera no devuelve nada; ver C.2 para la justificación del valor.

Aplicar el filtrado de metadatos dentro de ChromaDB previo al cálculo de similitud evita procesar documentos irrelevantes, optimiza el consumo y garantiza respuestas precisas e integras.



### B.5 — ETL y purga semántica
Para validar la solidez del pipeline ante datos inconsistentes y redundantes, se introdujeron 5 registros de prueba sucios en la base de conocimiento (base_conocimiento.json), elevando temporalmente el total inicial a 23 registros:

RUTA-MAD-FCO-DUP-JERGA: Misma información de Madrid–Roma, pero escrita en lenguaje informal.

RUTA-BCN-MAD-DUP-TEXTO: Copia del puente aéreo Barcelona–Madrid cambiando el orden de las palabras y sintaxis.

RUTA-BGY-BVA-DUP-CONCEPTUAL: Resumen en otras palabras de la ruta Bérgamo–Beauvais.

RUTA-TEST-CLAVE-INCORRECTA: Ruta de prueba (Bilbao–Málaga) con el nombre de un campo mal escrito (is_direct en vez de vuelo_directo_disponible).

RUTA-TEST-BOOL-STRING: Ruta de prueba (Valencia–Palma) con un tipo de dato inconsistente ("True" como texto en lugar del booleano true).

Adicionalmente, `etl_purga.py` simula en tiempo de ejecución una **segunda fuente de datos** (por ejemplo, otro proveedor o scraper) que reutiliza el ID `RUTA-TEST-CLAVE-INCORRECTA` para un registro de contenido distinto (ruta Sevilla–Oporto). Este registro no se persiste en `base_conocimiento.json` —solo vive dentro del script— para no romper el `upsert` de `vector_db.py` (B.1), que exige IDs únicos dentro de una misma llamada; en cambio, `etl_purga.py` sí puede resolverlo porque normaliza en memoria antes de tocar ChromaDB. Con este registro, el total procesado por el ETL sube a 24.

**Resolución de colisión de IDs.** El bloque `ids_vistos` de `etl_purga.py` detecta cuando un `id` ya fue visto en una pasada anterior del ETL y le agrega el sufijo `_dup` antes de seguir procesando, en vez de dejar que el segundo registro pise al primero en silencio. La corrida sobre los 24 registros lo confirma en la consola:

```text
[COLISIÓN DE ID DETECTADA] 'RUTA-TEST-CLAVE-INCORRECTA' ya existía en la base -> renombrado a 'RUTA-TEST-CLAVE-INCORRECTA_dup' para no sobrescribir el registro original.
```

`RUTA-TEST-CLAVE-INCORRECTA` (el registro original, de la clave mal nombrada) conserva su ID; el registro simulado de la segunda fuente (Sevilla–Oporto) queda indexado como `RUTA-TEST-CLAVE-INCORRECTA_dup`. Se verificó con `coleccion.get(ids=["RUTA-TEST-CLAVE-INCORRECTA", "RUTA-TEST-CLAVE-INCORRECTA_dup"])` que ambos registros conviven en la base final sin pisarse. La purga semántica que sigue confirma que este renombre no fue una coincidencia con la deduplicación por contenido: `RUTA-TEST-CLAVE-INCORRECTA_dup` no aparece en ninguno de los pares detectados más abajo, es decir, sobrevive intacto porque es otra ruta (Sevilla–Oporto frente a Bilbao–Málaga), así que el bloqueo por ruta de la purga ni siquiera lo compara con el original — la colisión era puramente de `id`, no de contenido duplicado, y el pipeline la resuelve como un problema distinto al que ataca la purga por distancia coseno.


**Primera versión y su límite: un único umbral para todos los pares**

La primera versión de `etl_purga.py` comparaba cada documento contra todos los demás (276 pares) con un único umbral de distancia coseno de 0.20. Con ese corte se purgaron 2 de los 3 casi-duplicados, pero `RUTA-BCN-MAD-DUP-TEXTO` (0.2138) sobrevivió y terminó contaminando la Killer Query 1 de B.6. Subir el umbral a 0.25 lo atrapaba, pero a cambio borraba `RUTA-BCN-FCO`, una ruta real, por estar a 0.2103 de `RUTA-MAD-FCO`: entre el duplicado que había que borrar y la ruta que había que conservar había un margen de apenas 0.0035.

El problema de fondo no era el número elegido, sino que el orden estaba invertido: dos rutas **distintas** (Madrid–Roma y Barcelona–Roma, las dos "vuelos baratos y directos a Roma") quedaban más cerca entre sí (0.2103) que una ruta y su propia paráfrasis (0.2138). Con ese orden, ningún umbral global puede borrar el duplicado sin borrar también la ruta real: cualquier corte que atrape a uno atrapa al otro.

**Corrección: bloqueo por ruta antes de medir la distancia**

Por diseño (A.3), la base tiene un único documento por ruta, que cubre las dos direcciones. Eso implica que dos documentos de rutas distintas nunca pueden ser duplicados entre sí, por más que sus textos se parezcan. La ruta es un dato duro que ya vive en los metadatos (`origen`, `destino`), así que se aplica la misma Regla del Arquitecto que en la búsqueda: lo exacto se resuelve con metadatos, y la similitud semántica se usa solo donde hace falta interpretar. `etl_purga.py` agrupa primero los documentos por ruta con `clave_ruta()` —que trata MAD–FCO y FCO–MAD como la misma ruta, con el mismo criterio bidireccional de A.3— y solo mide la distancia coseno entre documentos de la misma ruta. De los 276 pares posibles se comparan 3. El par Madrid–Roma / Barcelona–Roma ya no se compara: el falso positivo no se esquiva con un número, queda excluido por construcción.

**Justificación del umbral (0.25)**

Con el bloqueo, el umbral solo tiene que distinguir una paráfrasis de un documento genuinamente distinto *dentro de una misma ruta*. Las tres distancias medidas entre cada ruta y su casi-duplicado son 0.1509, 0.1600 y 0.2138: 0.25 las cubre a todas, con un margen de 0.036 sobre la más lejana, y es el mismo valor que ya se había probado antes, descartado únicamente por el falso positivo entre rutas que ahora es imposible. No se eligió un valor mucho más alto a propósito: si en el futuro llegara un segundo documento de la misma ruta con un texto muy distinto (por ejemplo, datos contradictorios de otra fuente), es preferible conservarlo antes que borrarlo en silencio, porque probablemente aporte información que el original no tiene.

La ejecución del script arrojó el siguiente log de consola (umbral 0.25, con bloqueo por ruta):
```text
Total registros cargados iniciales: 24
[COLISIÓN DE ID DETECTADA] 'RUTA-TEST-CLAVE-INCORRECTA' ya existía en la base -> renombrado a 'RUTA-TEST-CLAVE-INCORRECTA_dup' para no sobrescribir el registro original.
Registros tras normalización ETL: 24

Pares comparados (misma ruta): 3 de 276 posibles.
Se eliminaron 3 casi-duplicados semánticos.
Total registros finales purgados: 21

[PURGA DETECTADA - Distancia: 0.16]
  - Mantener (RUTA-MAD-FCO): La ruta Madrid–Roma (MAD–FCO) es una de las más transitadas del catálogo, con más de 1.400 vuelos re...
  - Eliminar (RUTA-MAD-FCO-DUP-JERGA): El trayecto entre Madrid Barajas y Roma Fiumicino posee alta demanda para viajes cortos a Italia o E...

[PURGA DETECTADA - Distancia: 0.2138]
  - Mantener (RUTA-BCN-MAD): La ruta Barcelona–Madrid (BCN–MAD) es la de mayor volumen de todo el catálogo, con 3.470 vuelos regi...
  - Eliminar (RUTA-BCN-MAD-DUP-TEXTO): Vuelos entre Barcelona y Madrid, el histórico corredor conocido como puente aéreo. Vuelos directos d...

[PURGA DETECTADA - Distancia: 0.1509]
  - Mantener (RUTA-BGY-BVA): La ruta Bérgamo–París Beauvais (BGY–BVA) es la más barata de todo el catálogo, con una mediana de ap...
  - Eliminar (RUTA-BGY-BVA-DUP-CONCEPTUAL): Ruta súper barata operada por Ryanair conectando Bérgamo y Beauvais (aeropuertos secundarios de Milá...

¡ChromaDB actualizada con éxito! Total indexados: 21
```

**Verificación sobre la colección.** Con `coleccion.get()` se confirmó que la colección final tiene 21 documentos, que ninguno de los 3 casi-duplicados sigue presente, que `RUTA-BCN-FCO` y `RUTA-TEST-CLAVE-INCORRECTA_dup` se conservaron, y que la colección mantiene `hnsw:space: cosine`. Sobre los embeddings ya guardados, la distancia entre `RUTA-MAD-FCO` y `RUTA-BCN-FCO` sigue siendo 0.2103 —por debajo de 0.25—: sin el bloqueo por ruta, el nuevo umbral la habría borrado; con el bloqueo, nunca se comparan.

**Limpieza ETL**: Normalizó con éxito la clave is_direct → vuelo_directo_disponible, convirtió la cadena "True" al booleano true, y resolvió la colisión de ID entre `RUTA-TEST-CLAVE-INCORRECTA` y el registro simulado de la segunda fuente con el mismo id, renombrando a este último a `RUTA-TEST-CLAVE-INCORRECTA_dup`.

**Purga semántica**: Eliminó los 3 casi-duplicados: RUTA-MAD-FCO-DUP-JERGA (0.1600), RUTA-BCN-MAD-DUP-TEXTO (0.2138) y RUTA-BGY-BVA-DUP-CONCEPTUAL (0.1509).

**Resultado final**: La base final consta de 21 registros: las 18 rutas reales, las 2 rutas de prueba corregidas por el ETL (BIO-AGP y VLC-PMI) y el registro de la colisión de ID (`RUTA-TEST-CLAVE-INCORRECTA_dup`). No queda ningún casi-duplicado y no se perdió ninguna ruta real.

| Versión del ETL | Pares comparados | Casi-duplicados purgados | Rutas reales borradas por error | Registros finales |
|---|---|---|---|---|
| Anterior: umbral 0.20, todos contra todos | 276 | 2 de 3 | 0 | 22 |
| Anterior: umbral 0.25, todos contra todos | 276 | 3 de 3 | 1 (`RUTA-BCN-FCO`) | 20 |
| **Actual: umbral 0.25, solo dentro de la misma ruta** | **3** | **3 de 3** | **0** | **21** |

Un SELECT DISTINCT no habría encontrado estos duplicados porque realiza una comparación de texto literal, evaluando cadenas exactas, por lo que redactar la misma idea con distintas palabras lo toma como registros diferentes. Además, al tener identificadores distintos (id), la base de datos los considera filas independientes, y cualquier variación en las claves o tipos de datos rompe la coincidencia exacta a nivel de bytes.

Tampoco alcanzaría con un `SELECT DISTINCT origen, destino`: el bloqueo por ruta solo indica qué documentos *pueden* ser duplicados, y es la distancia coseno la que confirma que efectivamente cuentan lo mismo. Si una misma ruta tuviera dos documentos con información realmente distinta, un DISTINCT por ruta los colapsaría igual y se perderían datos.

---

### B.6 — Killer Queries

Se diseñaron y ejecutaron tres Killer Queries contra la colección ya purgada de ChromaDB (21 rutas). Cada consulta se resuelve en tres pasos: recuperación semántica sobre la colección (con o sin filtro `where` nativo, según el caso), **corte por el umbral de distancia de C.2** y, solo si quedó algún resultado, el contexto se pasa a `gpt-4o-mini` (temperatura 0) con un system prompt que lo instruye a responder solo con lo que está en el contexto. Si ningún resultado supera el umbral, el script responde "no dispongo de esa información" **sin invocar al LLM**.

La tabla con las tres consultas, qué pone a prueba cada una, el resultado esperado vs. el real, y el log íntegro de la ejecución (candidatos con su distancia, IDs que sobrevivieron al umbral y respuesta) está en [`resultados_killer_queries.md`](resultados_killer_queries.md). Dos pasaron (2b y 3), una pasó solo parcialmente (2a) y una no (1):


- **Query 2 (pasó, con contraste):** la consulta "ruta directa a Tallin" se corrió dos veces: sin filtro y con `where={"vuelo_directo_disponible": True}`. **Sin filtro**, la semántica cruda recupera `RUTA-OSL-TLL` (distancia 0.436, bajo el umbral de 0.50), una ruta que **no tiene vuelo directo** y que viola la restricción del usuario; el mismo falso positivo que A.4 ya había encontrado. En este caso el LLM la atajó por su system prompt ("la ruta Oslo–Tallin no tiene vuelos directos"), pero esa corrección depende de que el modelo obedezca una instrucción, no de una garantía del sistema. **Con filtro**, `RUTA-OSL-TLL` queda excluida de la búsqueda: la ruta directa más cercana (`RUTA-LHR-LIS`, 0.629) supera el umbral y se descarta, por lo que responde "no dispongo" sin llamar al LLM. El embedding capta el registro semántico ("directa", "Tallin") pero no un hecho verificable como "0% de vuelos directos": por eso el filtro de metadatos es una garantía del retriever y no una cortesía del modelo, el argumento de B.2 y de la Regla del Arquitecto.
- **Query 3 (pasó, y ahora por el motivo correcto):** ChromaDB devuelve igual los 3 vecinos más cercanos (`RUTA-OSL-TLL` 0.561, `RUTA-ATH-LHR` 0.574, `RUTA-MAD-FCO` 0.605), pero los tres superan el umbral de 0.50, así que el retriever los descarta y el LLM nunca los ve. Antes esta query "pasaba" porque `gpt-4o-mini` lo decidía por prompt (ver C.2).
- **Query 1 (no pasó — limitación conocida):** la jerga ("escapada barata en el puente aéreo para laburar en el día") queda a 0.623 de `RUTA-BCN-FCO` y a 0.633 de `RUTA-BCN-MAD`, es decir, más lejos que la consulta fuera de catálogo de la Query 3 (0.561). Con embeddings `text-embedding-3-small` y descripciones de rutas largas, una frase coloquial y larga se aleja de todos los documentos, y ningún umbral puede aceptarla sin aceptar también Buenos Aires–Tokio. Se prefirió rechazarla (falso negativo) antes que dejar pasar una ruta inexistente (alucinación). Cabe notar que ya antes del umbral el retriever ponía primero a `BCN-FCO` y no a `BCN-MAD`, y que el LLM terminaba respondiendo "no dispongo de esa información": el umbral solo hace explícito en el código un rechazo que antes dependía del azar del prompt. La mejora natural es reescribir la consulta del usuario a una forma más cercana al catálogo antes de embeberla (query rewriting), que corresponde al orquestador de C.3.

---

## Parte C — Coherencia e Informe

### C.1 — Cadena de coherencia con la Entrega 1

| Elemento de la Entrega 1 (`informe.md`) | Cómo se implementa en la Entrega 2 |
|---|---|
| Columna "Base de Conocimiento" del PEAS, sección A.3 (entrega 1) | ← la colección ChromaDB `vuelos_smart_flight_assistant` con los 21 documentos de rutas, sección B.5 (entrega 2) |
| Campos de filtrado de la Matriz de Intenciones, sección B.3 (entrega 1) | ← los metadatos de la base (`origen`, `destino`, `vuelo_directo_disponible`, `categoria_precio`), sección A.3 (entrega 2) |
| Parámetros que el LLM extraía del `texto_libre` sección B.5 (entrega 1) | ← el destino del filtro cambia: de `WHERE` SQL a `where` nativo de ChromaDB. La extracción vía LLM desde `texto_libre` no está implementada, dado que los scripts reciben el filtro armado a mano, secciones B.4 y B.6 (entrega 2). |

El PEAS marcaba la Base de Conocimiento como *"Todavía sin base vectorial ni datos en vivo"* — acá se cubre ese hueco.

Los campos de filtrado se mantienen, solo cambia dónde se aplican: antes en el `WHERE` de SQL, ahora como metadatos nativos de ChromaDB. Excepción: `fecha_desde`/`fecha_hasta` siguen en el SQL transaccional, sin equivalente en la base vectorial (mismo motivo por el que la base es por ruta y no por fila).

La "Regla de oro" (*"en ningún caso el LLM toma decisiones... eso lo calcula siempre el backend"*) sigue aplicando al destino del filtro, pero acá llega armado a mano, no derivado de un LLM parseando el `texto_libre`.

---

### C.2 — El umbral de aceptación

**Qué se estimó en A.2 y por qué no servía.** En A.2 se estimó un umbral de similitud coseno de 0.75-0.80 sobre el ejercicio de 2 ejes hechos a mano. Con los embeddings reales de 1536 dimensiones ese número no aplica: las similitudes máximas reales son mucho más bajas (0.43-0.57 en A.4), así que un corte en 0.75 rechazaría incluso las mejores coincidencias del catálogo. En la versión anterior de este informe se concluyó de ahí que el umbral no hacía falta y que alcanzaba con el system prompt del LLM. **Esa conclusión era incorrecta:** el problema era el valor, no la idea de tener un corte. La consigna pide un umbral en la recuperación, y delegarlo al LLM tiene dos defectos: (1) el retriever igual entrega como "relevante" el mejor de los peores resultados (alucinación por sustitución), y (2) la decisión depende de que el modelo obedezca un prompt, no de una regla verificable.

**Qué se implementó.** Una constante `UMBRAL_DISTANCIA = 0.50` y una función `filtrar_por_umbral()` en `vector_db.py`, aplicadas en `buscar_vuelos()` (B.4) y en `ejecutar_rag()` de `B6_test_killer_queries.py` (B.6). El corte se hace sobre la **distancia coseno** que devuelve ChromaDB (0 = idéntico; es `1 - similitud`), igual que el `UMBRAL_DUPLICADO` de la ETL de B.5. El umbral equivale a una similitud mínima de 0.50, y se aplica a *cada* resultado, no solo al primero: de un top-3 pueden sobrevivir 0, 1, 2 o 3. Si no sobrevive ninguno, no se llama al LLM y se responde "no dispongo de esa información en el catálogo". El system prompt de B.6 se mantiene como segunda línea de defensa, no como la primera.

**Cómo se eligió 0.50.** Se midió la distancia del mejor resultado sobre la colección purgada para dos grupos de consultas:

| Grupo | Consultas | Mejor distancia |
|---|---|---|
| Con match real en el catálogo | `MAD-FCO`, `BCN-MAD`, `KEF-MAD`, `BCN-FCO`, "económicos a Europa", etc. | 0.276 – 0.456 |
| Fuera de catálogo | Buenos Aires–Tokio, Nueva York–Sídney, Montevideo–Ciudad del Cabo, hoteles en París, restaurante en Lima, receta de tortilla | 0.561 – 0.820 |

Entre 0.456 y 0.561 hay una brecha limpia, y 0.50 queda en el medio con aproximadamente 0.05 de margen de cada lado. Para el catálogo actual ese margen es razonable, pero la muestra es chica (21 documentos, 14 consultas), así que el valor debe recalibrarse si el catálogo crece o se cambia el modelo de embeddings.

**Costo conocido.** El umbral es un corte binario sobre un solo número, y las consultas coloquiales largas (la jerga de la Killer Query 1, a 0.623) quedan más lejos que algunas consultas fuera de catálogo (Buenos Aires–Tokio, a 0.561). No existe un umbral que acepte la primera y rechace la segunda, así que se eligió rechazar ambas: un falso negativo (el usuario reformula) es preferible a una alucinación (se le ofrece una ruta inexistente). Detalle en B.6.

---

### C.3 — Cierre: qué falta para una respuesta real al usuario

`buscar_vuelos()` de vector_db.py devuelve el diccionario crudo que entrega `coleccion.query()`: `ids`, `documents`, `metadatas` y `distances` de ChromaDB, sin traducir a lenguaje natural. Es el mismo punto de corte que ya se había señalado en C.5 de la Entrega 1 ("le falta la Base de Conocimiento"): esa base ya existe (Parte A y B de esta entrega), pero el eslabón que falta ahora es el siguiente, no el mismo.

B6_test_killer_queries.py prototipa a mano el paso que le falta a B.4: toma el `documents` recuperado, lo concatena como `contexto` y se lo pasa a `gpt-4o-mini` junto con un system prompt para que redacte la respuesta en español en vez de exponer el JSON. Pero lo hace con la `query` y el `where_filter` de cada Killer Query hardcodeados en el propio script (ver `ejecutar_rag(...)` en B6) — es un caso de prueba fijado a mano, no un flujo genérico que reciba el `texto_libre` de cualquier usuario.

Falta el **orquestador RAG con LangChain** (Unidad 4, próxima entrega): el componente que tome el `texto_libre` de una consulta real, decida qué filtro `where` armar y con qué texto llamar a `coleccion.query()` (hoy decidido por quien invoca la función a mano), dispare la recuperación híbrida de B.4 y encadene el resultado con un LLM que redacte la respuesta final — lo que B.6 ya prueba de forma manual y aislada por consulta, convertido en un pipeline único y reutilizable.

---