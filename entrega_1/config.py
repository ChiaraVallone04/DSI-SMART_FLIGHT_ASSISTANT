"""Configuración: credenciales y modelo LLM."""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# API key y modelo LLM
API_KEY = os.getenv("GEMINI_API_KEY")
MODELO_LLM = os.getenv("GEMINI_MODEL_NAME")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY no encontrada en el entorno.")
elif not MODELO_LLM:
    raise ValueError("GEMINI_MODEL_NAME no encontrada en el entorno.")

client = genai.Client()
