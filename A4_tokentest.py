"""
A.4 - Anatomia del token.

Compara cuantos tokens consume la misma consulta (ES vs EN) segun el
tokenizador: las 4 familias de encoding de OpenAI (tiktoken), varios
modelos reales de Anthropic (count_tokens) y varios modelos reales de
Gemini (count_tokens). Las partes de Anthropic y Gemini requieren
ANTHROPIC_API_KEY / GEMINI_API_KEY en .env; si falta alguna, esa parte
se omite con un aviso en vez de fallar.
"""

import os

import tiktoken
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)  # fuerza los valores de .env por sobre variables de entorno viejas del sistema

consulta_es = (
    "Sos un asistente de viajes. Decime cuánto cuesta volar de Madrid (MAD) a Berlín (BER) "
    "para 2 personas a mediados de octubre de 2026 y cuándo me conviene comprar el pasaje."
)

consulta_en = (
    "You are a travel assistant. Tell me how much it costs to fly from Madrid (MAD) to Berlin (BER) "
    "for 2 people in mid-October 2026 and when I should buy the ticket."
)

# --- Base: gpt-4o (o200k_base) ---
enc = tiktoken.encoding_for_model("gpt-4o")
es_tokens = enc.encode(consulta_es)
en_tokens = enc.encode(consulta_en)

print(f"ES: {len(es_tokens)} tokens")
print(f"EN: {len(en_tokens)} tokens")

diferencia = len(es_tokens) - len(en_tokens)
porcentaje = (len(es_tokens) / len(en_tokens) - 1) * 100
print(f"Diferencia: +{diferencia} tokens ({porcentaje:.0f}% mas en ES)")

# --- Comparacion entre las 4 familias de encoding de OpenAI ---
print("\n--- Comparacion entre encodings de OpenAI ---")
encodings = {
    "o200k_base": "gpt-4o / gpt-4o-mini",
    "cl100k_base": "gpt-4-turbo, gpt-4, gpt-3.5-turbo, embeddings-3",
    "p50k_base": "Codex, text-davinci-002/003",
    "r50k_base": "GPT-3 (davinci, gpt2)",
}

for enc_name, modelos in encodings.items():
    enc_i = tiktoken.get_encoding(enc_name)
    es_i = len(enc_i.encode(consulta_es))
    en_i = len(enc_i.encode(consulta_en))
    dif_i = (es_i / en_i - 1) * 100
    print(f"{enc_name:<14} ({modelos:<45}) ES={es_i:>3} EN={en_i:>3}  +{dif_i:.1f}% en ES")

# --- Conteo real de tokens con varios modelos de Claude (Anthropic count_tokens) ---
# tiktoken NO sirve para Claude -- usa un tokenizador distinto. Anthropic
# expone un endpoint dedicado y gratuito para contar tokens reales, y el
# conteo es especifico por modelo (no todos tokenizan igual).
print("\n--- Comparacion entre modelos de Anthropic (Claude) ---")

if not os.getenv("ANTHROPIC_API_KEY"):
    print("Falta ANTHROPIC_API_KEY en .env -- se omite esta parte.")
else:
    client_claude = Anthropic()
    modelos_claude = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"]

    def contar_tokens_claude(texto: str, modelo: str) -> int:
        resp = client_claude.messages.count_tokens(
            model=modelo,
            messages=[{"role": "user", "content": texto}],
        )
        return resp.input_tokens

    for modelo in modelos_claude:
        es_c = contar_tokens_claude(consulta_es, modelo)
        en_c = contar_tokens_claude(consulta_en, modelo)
        dif_c = (es_c / en_c - 1) * 100
        print(f"{modelo:<20} ES={es_c:>3} EN={en_c:>3}  +{dif_c:.1f}% en ES")

# --- Conteo real de tokens con varios modelos de Gemini (count_tokens) ---
print("\n--- Comparacion entre modelos de Gemini ---")

if not os.getenv("GEMINI_API_KEY"):
    print("Falta GEMINI_API_KEY en .env -- se omite esta parte.")
else:
    from google import genai

    client_gemini = genai.Client()
    modelos_gemini = ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-pro-latest"]

    def contar_tokens_gemini(texto: str, modelo: str) -> int:
        resp = client_gemini.models.count_tokens(model=modelo, contents=texto)
        return resp.total_tokens

    for modelo in modelos_gemini:
        es_g = contar_tokens_gemini(consulta_es, modelo)
        en_g = contar_tokens_gemini(consulta_en, modelo)
        dif_g = (es_g / en_g - 1) * 100
        print(f"{modelo:<26} ES={es_g:>3} EN={en_g:>3}  +{dif_g:.1f}% en ES")
