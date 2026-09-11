"""Corre el mismo lote de casos de prueba contra Claude (Anthropic) y genera resultados_lote_claude_*.md."""
import sys

from pydantic import ValidationError

from casos_test import CASOS_TEST, resumir
from config_claude import MODELO_LLM
from extraccion_claude import extraer_intencion

TECNICA = "zero-shot" if "--zero-shot" in sys.argv else "cot"

# Guarda una fila resumida por cada caso de prueba, para armar la tabla final


def main():
    filas = []
    # Corre cada caso de prueba contra la API y junta los resultados en 'filas'
    for index, caso in enumerate(CASOS_TEST, start=1):
        try:
            datos = extraer_intencion(caso, canal="test", tecnica=TECNICA)
            print(f"Caso {index} [OK]: {datos.model_dump()}")
            dump = datos.model_dump(exclude={"razonamiento"})
            campos = ", ".join(
                f"{k}={v}" for k, v in dump.items() if v is not None)
            filas.append(
                (index, resumir(caso), campos or "(todos los campos en null)", "Sí", "—"))
        except ValidationError as e:
            print(f"Caso {index} [RECHAZO POR VALIDACION]: {e}")
            filas.append((index, resumir(caso), resumir(
                e.errors()[0]["msg"], 90), "No", "ValidationError"))
        except Exception as e:
            print(f"Caso {index} [ERROR DEL SISTEMA]: {e}")
            filas.append((index, resumir(caso), resumir(
                str(e), 90), "No", "Error del sistema"))
    lineas = [
        "# Resultados del lote de prueba (Claude)",
        "",
        f"Modelo usado: `{MODELO_LLM}` · {len(filas)} inputs corridos contra la API real.",
        "",
        "| # | Input (resumido) | Salida del modelo | ¿Validó Pydantic? | Tipo de error si falló |",
        "|---|---|---|---|---|",
    ]
    for fila in filas:
        lineas.append(
            f"| {fila[0]} | {fila[1]} | {fila[2]} | {fila[3]} | {fila[4]} |")

    archivo_salida = f"resultados_lote_claude_{TECNICA.replace('-', '_')}.md"
    with open(archivo_salida, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")

    print(f"\nTabla escrita en {archivo_salida}")


if __name__ == "__main__":
    main()
