"""Punto de entrada CLI: lee una consulta y muestra la extracción por consola."""
import sys
import httpx
from pydantic import ValidationError
from extraccion import extraer_intencion

# Punto de entrada: lee la consulta del usuario y la procesa con extraer_intencion


def main():
    entrada = " ".join(sys.argv[1:]) or input(
        "Ingresá tu consulta de vuelos: ").strip()
    try:
        respuesta = extraer_intencion(entrada, canal="cli")
        print(respuesta.model_dump())
    except ValidationError as e:
        print(f"[ERROR DE VALIDACION] El JSON no cumplió el schema: {e}")
    except httpx.HTTPError as e:
        print(f"[ERROR DE RED] No se pudo contactar la API: {e}")
    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")


# Ejecutar aplicación
if __name__ == "__main__":
    main()
