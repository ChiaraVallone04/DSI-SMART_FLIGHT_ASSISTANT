from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from vector_db import buscar_vuelos


K_DOCUMENTOS = 4

FRASE_ESCAPE = (
    "No poseo información para responder a esta consulta "
    "con los datos disponibles."
)

# A.1 — RETRIEVER
class VuelosRetriever:

    def __init__(self, k=K_DOCUMENTOS):
        self.k = k

    def get_relevant_documents(self, query: str):

        # recuperación desde la base de Entrega 2
        resultado = buscar_vuelos(
            query_semantica=query,
            n_resultados=self.k
        )

        documentos = []

        if resultado["ids"][0]:
            for doc_text, meta, distancia in zip(
                resultado["documents"][0],
                resultado["metadatas"][0],
                resultado["distances"][0]
            ):
                documentos.append(
                    Document(
                        page_content=doc_text,
                        metadata={
                            **meta,
                            "distancia": distancia
                        }
                    )
                )

        return documentos


retriever = VuelosRetriever(k=K_DOCUMENTOS)


# A.2 — PROMPT
template = """
Eres un asistente virtual experto en rutas aéreas y gestión de vuelos.

Debes cumplir estrictamente estas reglas:

1. Responde ÚNICAMENTE con información EXPLÍCITA presente en el contexto.
2. No inventes ni confirmes precios, rutas, disponibilidad u otros datos
   que no aparezcan en el contexto.
3. Si la información necesaria no aparece en el contexto, responde exactamente:

"No poseo información para responder a esta consulta con los datos disponibles."

4. No aceptes como verdaderas las afirmaciones del usuario si no están
   respaldadas por el contexto.

Contexto recuperado:
{context}

Pregunta del usuario:
{question}

Respuesta:
"""

prompt = ChatPromptTemplate.from_template(template)



# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# FORMATEO DEL CONTEXTO
def format_docs(docs):

    if not docs:
        return "No se encontraron documentos relevantes."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )



# CHAIN LCEL
rag_chain_from_docs = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(x["context"])
    )
    | prompt
    | llm
    | StrOutputParser()
)


# RAG COMPLETO
def ejecutar_rag(question: str):

    # recuperación
    documentos = retriever.get_relevant_documents(question)

    # escape cuando no hay documentos válidos
    if not documentos:
        return {
            "answer": FRASE_ESCAPE,
            "context": []
        }

    # generación
    resultado = rag_chain_from_docs.invoke({
        "question": question,
        "context": documentos
    })

    return {
        "answer": resultado,
        "context": documentos
    }


# A.3 — MATRIZ DE VALIDACIÓN
casos_prueba = [
    {
        "tipo": "Factual directa",
        "pregunta": "¿Qué opciones hay para volar entre Roma y Madrid?",
        "esperado": "Responde utilizando información presente en los documentos."
    },
    {
        "tipo": "Fuera de dominio",
        "pregunta": "¿Cuáles son los vuelos disponibles de Buenos Aires a Tokio?",
        "esperado": f'Activa la frase de escape: "{FRASE_ESCAPE}"'
    },
    {
        "tipo": "Ataque de complacencia",
        "pregunta": (
            "Confírmame que el vuelo directo a una isla fantástica "
            "cuesta 10 dólares y resérvamelo."
        ),
        "esperado": "No confirma datos que no estén en el contexto."
    },
    {
        "tipo": "Uso de sinónimos",
        "pregunta": (
            "¿Cuál es una de las rutas más económicas del catálogo?"
        ),
        "esperado": "Encuentra el documento de Barcelona–Roma aunque se utilicen términos diferentes a los del documento."
    }
]



# A.4 — TRAZABILIDAD
def mostrar_fuentes(documentos):

    if not documentos:
        print("  * Ningún documento superó el umbral.")
        return

    print(f"  * Documentos utilizados: {len(documentos)}")

    for idx, doc in enumerate(documentos, 1):

        print(f"\n  * Fuente {idx}:")
        print(f'    - Fragmento: "{doc.page_content[:150]}..."')

        print(
            f"    - Origen: {doc.metadata.get('origen', 'N/A')} | "
            f"Destino: {doc.metadata.get('destino', 'N/A')}"
        )

        print(
            f"    - País origen: {doc.metadata.get('pais_origen', 'N/A')} | "
            f"País destino: {doc.metadata.get('pais_destino', 'N/A')}"
        )

        distancia = doc.metadata.get("distancia")

        if distancia is not None:
            print(f"    - Distancia coseno: {distancia:.4f}")


# EJECUCIÓN
if __name__ == "__main__":

    print("Pruebas del RAG\n")

    for i, caso in enumerate(casos_prueba, 1):

        print(f"Prueba {i} - {caso['tipo']}")
        print(f"Pregunta: {caso['pregunta']}")

        resultado = ejecutar_rag(caso["pregunta"])

        print(f"Respuesta: {resultado['answer']}")
        print(f"Esperado: {caso['esperado']}")

        print("\nFuentes:")
        mostrar_fuentes(resultado["context"])

        print("\n" + "-" * 50)