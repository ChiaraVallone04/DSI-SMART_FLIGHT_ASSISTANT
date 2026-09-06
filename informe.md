# ARTE A: Diagnóstico y PEAS

## 1. Definición del caso de dominio

### A.1 — El caso
**Dominio**
Asistente Virtual Intuitivo de Planificación de Vuelos y Optimización de Itinerarios (*Smart Flight Assistant*).

**Descripción del caso**
Un usuario interactúa con un agente conversacional para buscar rutas de vuelo dentro de Europa. El sistema debe:
- **Procesar la solicitud:** el sistema analiza el diálogo con el usuario para identificar los parámetros clave del viaje (origen, destino, rango de fechas, pasajeros, máximo de escalas y criterio de ordenamiento).
- **Consulta de datos:** traduce estos parámetros a consultas SQL estructuradas sobre el dataset europe_flights_google_prices.csv (1,15 millones de filas para los períodos mayo-junio y septiembre-diciembre de 2026) con el fin de recuperar opciones de vuelos reales. Usa el comportamiento histórico (lo que ya pasó en mayo-junio) junto con las tarifas que se están vendiendo hoy en tiempo real (septiembre-diciembre) para predecir si el precio actual de un vuelo a salir en noviembre, por ejemplo, va a subir o bajar en los días que quedan.
- **Análisis predictivo y recomendación:** a partir de variables como la antelación de la reserva (days_left), la aerolínea, el día de la semana y el número de escalas, el sistema evalúa la tendencia de precios para estimar tarifas futuras y aconsejar al usuario el momento óptimo de compra.

### Prueba de alucinación del modelo (Prompt & Respuesta)
### A.2 — Evidencia de la necesidad (réplica de "El Proveedor Enojado")
**Prompt sin darle la fuente de verdad :** *"Sos un asistente de viajes. Decime cuánto cuesta volar de Madrid (MAD) a Berlín (BER) para 2 personas a mediados de octubre de 2026 y cuándo me conviene comprar el pasaje."*

**Respuesta del Modelo (Alucinada):** Para dos personas, el vuelo directo de ida desde Madrid a Berlín a mediados de octubre de 2026 cuesta desde **76 €** en aerolíneas *low-cost* (como Ryanair) y desde **167 €** en aerolíneas tradicionales (como Iberia), ubicándose los pasajes de ida y vuelta entre **150 € y 320 €** en total; considerando que falta poco más de un mes para la fecha, lo conveniente es **comprar el pasaje de inmediato** antes de que las tarifas sigan subiendo.

**Diagnóstico del fallo:**
- **Alucinación de precios:** inventó cifras de costo (€76, €167) usando su memoria y patrones probabilísticos en lugar de consultar la base de datos real.
- **Inferencia temporal errónea:** asumió erróneamente el tiempo restante de compra y aconsejó "comprar de inmediato" sin analizar la variable de anticipación (days_left) del dataset.

Estos errores se deben a que no tiene acceso a los datos reales. 

### Matriz PEAS Extendida
### A.3 — PEAS extendido (5 pilares)
| Componente | Definición | Aplicación en el Smart Flight Assistant |
| :--- | :--- | :--- |
| **P**erformance<br>*(Rendimiento)* | Criterios de éxito con los que se evalúa el comportamiento del agente. | • Exactitud en las recomendaciones de vuelos y precios.<br>• Minimización del costo de los pasajes encontrados.<br>• Precisión en la extracción de intenciones y parámetros de búsqueda.<br>• Rapidez de respuesta. |
| **E**nvironment<br>*(Entorno)* | Todo lo que rodea al agente y con lo que interactúa. | • Base de datos relacional de vuelos (`europe_flights_google_prices.csv`).<br>• Interfaz de usuario (chat / consola).<br>• API de LLM (para el parseo estructurado). |
| **A**ctuators<br>*(Actuadores)* | Los medios por los cuales el agente ejecuta acciones en el entorno. | • Consultas a la base de datos (SQL / Pandas).<br>• Respuestas de texto para el usuario.<br>• Estructura JSON generada con Pydantic. |
| **S**ensors<br>*(Sensores)* | Los medios por los cuales el agente percibe la información del entorno. | • Prompt / mensaje de texto ingresado por el usuario.<br>• Tablas y filas devueltas por la base de datos tras la consulta. |
### A.4 — Anatomía del token
Para evaluar el impacto del idioma en el consumo de recursos, se tokenizó mediante la librería tiktoken el texto de la consulta utilizada en el apartado A.2 ("El Proveedor Enojado"), comparando su versión original en español contra su traducción al inglés.

(El código se encuentra en A4_tokentest.py)

**Resultado:** 

ES = 47 tokens 
EN = 43 tokens · 
Diferencia = +4 tokens (9% más en español).

**Reflexión:** aun con una consulta relativamente sencilla (sin un uso intensivo de tildes ni nombres propios largos), el español consume un 9% más de tokens que el inglés para solicitar lo mismo. Con miles de consultas diarias en el *Smart Flight Assistant*, este margen se traduce en un sobrecosto sistemático tanto en las entradas como en las respuestas generadas, sumándose directamente al costo de inyectar el contexto del dataset de vuelos en cada prompt.

Nota: Las mediciones presentadas fueron calculadas con la herramienta `tiktoken` sobre el tokenizador de OpenAI (`gpt-4o`). En este proyecto se utilizará modelos de Gemini.

