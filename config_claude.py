"""Configuración: credenciales y modelo LLM (backend Anthropic/Claude)."""
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

# API key y modelo LLM
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODELO_LLM = os.getenv("ANTHROPIC_MODEL_NAME")

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY no encontrada en el entorno.")
elif not MODELO_LLM:
    raise ValueError("ANTHROPIC_MODEL_NAME no encontrada en el entorno.")

client = Anthropic()
