# Análisis de costos y latencia por camino del grafo (LangSmith)

Proyecto `smart-flight-assistant-entrega3`, runs raíz `consulta_smart_flight` con tag `entrega_4` de las últimas 48 h: **105 consultas**, costo total **US$ 0.0489** (US$ 0.00047 por consulta en promedio). Generado el 2026-11-09 13:32.

| Camino | Consultas | Tokens prom. (in + out) | Herram. prom. | Costo prom. (US$) | Costo por 1.000 consultas | Latencia prom. | Latencia p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fuera_de_alcance` | 17 | 732 (683 + 49) | 0.0 | 0.00013 | US$ 0.13 | 1.3 s | 2.4 s |
| `conversacion` | 6 | 1,031 (964 + 67) | 0.0 | 0.00018 | US$ 0.18 | 6.2 s | 14.1 s |
| `rag_directo` | 42 | 2,729 (2,476 + 253) | 1.0 | 0.00051 | US$ 0.51 | 9.6 s | 18.8 s |
| `agente_react` | 40 | 4,280 (4,027 + 253) | 1.0 | 0.00060 | US$ 0.60 | 9.6 s | 25.1 s |

Caché de prompts de OpenAI: a precio de lista estas consultas habrían costado US$ 0.0556; con los tokens servidos desde la caché costaron US$ 0.0489 (12% menos).

Los inputs bloqueados por los guardrails no figuran porque no llaman al LLM (costo cero).
