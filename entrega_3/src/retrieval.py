"""Recuperación: el RAG avanzado de la Entrega 3, adaptado para usarlo dentro del agente.

Mismo retriever (chunks 500/100 en ChromaDB), mismos 8 candidatos, mismo juez LLM y mismo top 3 que el RAG
avanzado evaluado con RAGAS en la Entrega 3 (entrega_3/rag_pipeline.py).

Lo único que cambia es la regla 4 del prompt: en la E3 el sistema NO reservaba y la regla lo decía; ahora el
agente sí reserva (con crear_reserva), así que esa regla lo hacía contradecirse ("no realizo reservas ni
compras"). El archivo de la E3 no se toca: el prompt se deriva acá reemplazando solo esa línea.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough

from src import config  # noqa: F401  (agrega entrega_2/ y entrega_3/ al sys.path)

import rag_pipeline as e3  # noqa: E402  (Entrega 3)
from rag_pipeline import FRASE_ESCAPE  # noqa: E402

REGLA_4_E3 = "4. No realizas reservas ni compras: si te lo piden, aclara que solo brindas información."
REGLA_4_AGENTE = ("4. Las reservas y compras las gestiona el asistente con otra herramienta: nunca digas que no se "
                  "puede reservar o comprar. Responde solo la parte informativa de la pregunta.")
if REGLA_4_E3 not in e3.template:
    raise RuntimeError("Cambió la regla 4 del prompt de la Entrega 3: actualizar REGLA_4_E3 en src/retrieval.py")

prompt_rag_agente = ChatPromptTemplate.from_template(
    e3.template.replace(REGLA_4_E3, REGLA_4_AGENTE)).partial(frase_escape=FRASE_ESCAPE)

_generar = RunnableBranch(
    (lambda x: not x["context"], RunnableLambda(lambda _: FRASE_ESCAPE)),  # escape decidido por código (E3)
    RunnablePassthrough.assign(context=lambda x: e3.format_docs(x["context"])) | prompt_rag_agente | e3.llm
    | StrOutputParser(),
)

# chunks -> 8 candidatos -> juez LLM -> top 3 -> prompt (regla 4 adaptada) -> LLM -> respuesta citada
rag_chain_agente = (
    RunnableParallel(candidatos=e3.retriever_candidatos, question=RunnablePassthrough())
    .assign(puntuados=RunnableLambda(e3.puntuar_con_juez).with_config(run_name="juez_llm"))
    .assign(context=RunnableLambda(e3.seleccionar_mejores).with_config(run_name="seleccion_top_n"))
    .assign(answer=_generar)
)
