"""Smart Flight Assistant — punto de entrada del TP Final (CLI).

    python main.py ingest                          construye la base vectorial (carga + ETL + chunks)
    python main.py ask --session demo "mensaje"     un turno de conversación con el agente (con memoria)
    python main.py serve                           levanta la API REST (http://127.0.0.1:8000/docs)
    python main.py matriz                          matriz de validación del agente (LLM real)
    python main.py verificar [--paso1|--paso2]     verificación end-to-end contra la API levantada
    python main.py costos [--horas 24]             costos y latencia por camino del grafo (LangSmith)
    python main.py eval [--pipeline ambos]         evaluación RAGAS del RAG de la Entrega 3
    pytest                                         suite de seguridad (tests/security_tests.py)

Cada comando importa solo lo que necesita: `python main.py --help` no abre ChromaDB ni llama a ninguna API.
La configuración no secreta está en config.toml; las claves, en .env (ver .env.example).
"""
import argparse
import sys


def ingest(_args):
    from src.ingest import construir_base

    construir_base()


def ask(args):
    from src.graph import responder

    resultado = responder(args.mensaje, args.session)
    print(f"[{resultado['camino']} · {resultado['intencion']} · herramientas={resultado['herramientas_usadas']} · "f"US$ {resultado['costo_usd']:.5f}]\n{resultado['respuesta']}")


def serve(args):
    import uvicorn

    from src.config import API_HOST, API_PORT

    uvicorn.run("src.api:app", host=args.host or API_HOST, port=args.port or API_PORT)


def matriz(_args):
    from src.validation import main as correr_matriz

    correr_matriz()


def verificar(args):
    from src.verify import ejecutar

    sys.exit(ejecutar(args.url, "paso1" if args.paso1 else "paso2" if args.paso2 else None))


def costos(args):
    from src.costs import analizar

    analizar(args.horas)


def evaluar(args):
    import asyncio

    from src import config  # agrega entrega_2/ y entrega_3/ al path

    import evaluacion_ragas  # Entrega 3, Parte C

    elegidos = list(evaluacion_ragas.PIPELINES) if args.pipeline == "ambos" else [args.pipeline]
    asyncio.run(evaluacion_ragas.main(elegidos, evaluacion_ragas.CARPETA / args.salida))
    print(f"(el detalle quedó en {config.RAIZ_REPO.name}/entrega_3/{args.salida})")


def main():
    parser = argparse.ArgumentParser(prog="main.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    comandos = parser.add_subparsers(dest="comando", required=True)

    comandos.add_parser("ingest", help="construye la base vectorial desde data/base_conocimiento.json").set_defaults(
        funcion=ingest)

    p = comandos.add_parser("ask", help="un turno de conversación con el agente")
    p.add_argument("--session", required=True, help="thread_id de la conversación (misma sesión = misma memoria)")
    p.add_argument("mensaje")
    p.set_defaults(funcion=ask)

    p = comandos.add_parser("serve", help="levanta la API REST")
    p.add_argument("--host", default=None, help="por defecto, el de config.toml")
    p.add_argument("--port", type=int, default=None, help="por defecto, el de config.toml")
    p.set_defaults(funcion=serve)

    comandos.add_parser("matriz", help="matriz de validación del agente (resultados/resultados_matriz.md)") \
        .set_defaults(funcion=matriz)

    p = comandos.add_parser("verificar", help="verificación end-to-end contra la API levantada")
    p.add_argument("--url", default="http://localhost:8000")
    grupo = p.add_mutually_exclusive_group()
    grupo.add_argument("--paso1", action="store_true", help="prueba de reinicio: crea la conversación")
    grupo.add_argument("--paso2", action="store_true", help="prueba de reinicio: verifica la memoria")
    p.set_defaults(funcion=verificar)

    p = comandos.add_parser("costos", help="costos y latencia por camino del grafo (LangSmith)")
    p.add_argument("--horas", type=int, default=24)
    p.set_defaults(funcion=costos)

    p = comandos.add_parser("eval", help="evaluación RAGAS del RAG de la Entrega 3 (golden dataset)")
    p.add_argument("--pipeline", choices=["basico", "avanzado", "ambos"], default="ambos")
    p.add_argument("--salida", default="resultados_ragas.json", help="archivo de resultados dentro de entrega_3/")
    p.set_defaults(funcion=evaluar)

    args = parser.parse_args()
    args.funcion(args)


if __name__ == "__main__":
    main()
