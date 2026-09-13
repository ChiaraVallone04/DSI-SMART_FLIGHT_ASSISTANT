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

**Desangre de tokens.** Para medir el costo real (no una estimación) de mandar la base completa en cada consulta, se midieron con `tiktoken` (mismo método usado en A.4 de la Entrega 1, sobre el tokenizador de `gpt-4o`) los 18 documentos completos de `base_conocimiento.json` (texto + metadatos). El resultado: **5.345 tokens** solo por la base. Sumado al System Prompt actual de la Entrega 1 (211 tokens), se estarían mandando **5.556 tokens de contexto fijo en cada consulta**, antes de que el usuario escriba una sola palabra de su pregunta. Y esto es con apenas 18 documentos: si se quisiera cubrir las 4.147 rutas reales que existen en el dataset completo (`europe_flights_google_prices.csv`), a un promedio de ~297 tokens por documento, serían **~1,23 millones de tokens por consulta** — inviable tanto en costo como en el límite de contexto del modelo. Lo más grave es que el 94% de esos 18 documentos (17 de 18) resultan irrelevantes para cualquier pregunta puntual del usuario, y se pagarían igual, en cada consulta, todo el tiempo.

**Lost in the Middle.** Si los 18 documentos se insertaran en el orden en que están en `base_conocimiento.json`, uno como `RUTA-BRE-FRA` (posición 9 de 18) queda enterrado justo en el medio de un bloque de ~5.300 tokens. Es exactamente el fenómeno descripto en la literatura de "lost in the middle": los LLM prestan más atención a lo que está al principio y al final del contexto, y degradan su capacidad de usar correctamente la información que cae en el medio de un bloque largo. Con 18 documentos el riesgo ya es real; con cientos o miles sería directamente inutilizable — el modelo podría "ver" la ruta Bremen-Fráncfort en su contexto y aun así no usarla para responder, simplemente por dónde quedó ubicada.

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
El índice no está en disco, así que no hay otra opción que volver a llamar a la API de embeddings para los 18 documentos — se pagan tokens de nuevo aunque el contenido de `base_conocimiento.json` no haya cambiado un solo carácter desde la corrida anterior.

**Con persistencia (`faiss.write_index()` ya se ejecutó):**
```
$ python pipeline_vectorial.py
[disco] Índice recargado desde 'faiss_index\index.faiss' — sin llamadas a la API de embeddings.
```
Misma consulta, mismo resultado, cero llamadas a OpenAI: el índice se reconstruye desde los bytes en disco en milisegundos.

**Reflexión.** En producción, si el servidor se reinicia (deploy, crash, escalado) y el índice solo vivía en RAM, hay que regenerar embeddings de toda la base antes de poder responder la primera consulta — tiempo muerto y costo repetido en cada reinicio, y en un catálogo real (miles de documentos, no 18) ese tiempo de arranque en frío deja de ser trivial. El caso de **dos servidores** es peor todavía si cada uno mantiene su propio índice solo en memoria: no solo se paga el costo de generar embeddings dos veces, sino que nada garantiza que ambos índices queden idénticos si la base se actualizó entre una regeneración y la otra — dos instancias respondiendo con "verdades" ligeramente distintas. Persistir en disco (y, mejor todavía, en un storage compartido entre instancias) es lo que evita que la disponibilidad del servicio dependa de que la RAM de un proceso nunca se reinicie.
