"""Parte D — guardrails deterministas (en código, fuera del LLM).

Entrada (antes de que el texto llegue al agente):
  1. validar_longitud: rechaza textos vacíos o más largos que MAX_CARACTERES.
  2. detectar_inyeccion: rechaza intentos de prompt injection (cambiar el rol o las reglas del asistente,
     pedir el prompt del sistema, falsas "instrucciones del sistema").
Salida (antes de que el texto llegue al usuario, a /estado o a los logs):
  3. enmascarar_pii: reemplaza emails, teléfonos, tarjetas, documentos/pasaportes y nombres de pasajeros.

Son reglas en Python y no instrucciones en el prompt a propósito: el usuario controla el texto que le llega al
modelo, pero no puede modificar este código. Un input bloqueado nunca llega al LLM, así que no cuesta tokens.
"""
import re
import unicodedata
from dataclasses import dataclass

from src.config import MAX_CARACTERES  # config.toml -> [guardrails]

MENSAJE_BLOQUEO_LONGITUD = (
    "Tu mensaje es demasiado largo ({largo} caracteres; el máximo es {maximo}). "
    "¿Podés resumir tu consulta sobre vuelos en pocas líneas?"
)
MENSAJE_BLOQUEO_VACIO = "No recibí ninguna consulta. ¿En qué ruta o vuelo te puedo ayudar?"
# no le decimos al usuario qué patrón lo frenó: eso le daría pistas para reformular el ataque
MENSAJE_BLOQUEO_INYECCION = (
    "No puedo procesar ese pedido. Puedo ayudarte con información de rutas aéreas, "
    "disponibilidad de vuelos y reservas."
)


@dataclass
class ResultadoValidacion:
    valido: bool
    motivo: str | None = None      # código interno para logs y métricas (ej. "inyeccion_prompt")
    mensaje: str | None = None     # respuesta amable para el usuario


def _normalizar(texto: str) -> str:
    """Minúsculas y sin tildes, para que 'Ignorá' e 'ignora' caigan en el mismo patrón."""
    sin_tildes = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in sin_tildes if not unicodedata.combining(c)).lower()


# ── 1. Longitud ──────────────────────────────────────────────────────────────────────────────────────────────────

def validar_longitud(texto: str, maximo: int = MAX_CARACTERES) -> ResultadoValidacion:
    """Un input gigante puede diluir las instrucciones del sistema o disparar el costo de tokens."""
    if not texto or not texto.strip():
        return ResultadoValidacion(False, "input_vacio", MENSAJE_BLOQUEO_VACIO)
    if len(texto) > maximo:
        return ResultadoValidacion(
            False, "longitud_excedida", MENSAJE_BLOQUEO_LONGITUD.format(largo=len(texto), maximo=maximo))
    return ResultadoValidacion(True)


# ── 2. Prompt injection ──────────────────────────────────────────────────────────────────────────────────────────

# se aplican sobre el texto normalizado (minúsculas, sin tildes). Cubren español rioplatense/neutro e inglés
# Cada patrón apunta a las instrucciones DEL ASISTENTE, no a cualquier palabra parecida: "olvidate de Madrid, ¿qué
# reglas de equipaje tiene Ryanair?" o "quiero viajar sin restricciones de equipaje" son consultas legítimas
_OBJETO_REGLAS = r"(instruccion|regla|indicacion|restriccion|directiva|norma|politica)\w*"
PATRONES_INYECCION = [
    # "ignorá / olvidate de / saltate (todas) las|tus instrucciones / reglas (anteriores)"
    r"\b(ignor\w*|olvid\w*|saltea\w*|saltate|omit\w*|desobedec\w*|anul\w*|descarta\w*)\s+(de\s+)?(todas?\s+)?"
    r"(las|tus|sus|esas|estas|cualquier|toda)\s+(\w+\s+){0,2}" + _OBJETO_REGLAS,
    r"\b(ignore|disregard|forget|override)\b.{0,30}\b(instructions?|rules?|prompt|guidelines)",
    # cambio de rol: "a partir de ahora / desde este momento sos...", "tu nuevo rol es...", "you are now"
    r"\b(a partir de ahora|desde (este momento|ahora)|de ahora en mas)\b.{0,30}\b(sos|eres|seras|tu (nuevo )?rol|"
    r"actua\w*|respond\w*|tenes que|tienes que|debes|obedec\w*)\b",
    r"\bahora (sos|eres) (un|una) (\w+ ){0,2}(asistente|ia|bot|modelo|agente|sistema)\b",
    r"\b(tu|su) nuevo (rol|papel|objetivo|proposito|modo|trabajo)\b",
    r"\b(actua|actuar|comportate|hace de cuenta)\b.{0,20}\bcomo\b.{0,30}\b(sin (restricciones|reglas|limites|filtros)|otro asistente|desarrollador|administrador)",
    r"\byou are now\b|\bact as\b.{0,30}\b(unrestricted|dan|developer)",
    # pedir o revelar las instrucciones internas
    r"\b(revela\w*|mostra\w*|mostrame|deci\w*|dame|imprim\w*|repet\w*|copi\w*)\b.{0,30}\b(tu|tus|el|las|sus)\b.{0,15}\b(prompt|instrucciones (del sistema|internas|originales|iniciales))",
    r"\b(system prompt|prompt del sistema|prompt de sistema|mensaje del sistema)\b",
    r"\b(copi|repet|reproduc|transcrib|escrib|imprim)\w*\s+(textual\w*|literal\w*|palabra por palabra)\b.{0,60}"
    r"\b(antes|arriba|anterior\w*|instruccion\w*|prompt|te dijeron|te indicaron|configur\w*)",
    # falsas instrucciones del sistema embebidas en el mensaje: "[SYSTEM]", "<system>", "SYSTEM: ..." al inicio
    r"\[\s*(system|sistema|instruccion\w*|admin\w*|developer)\b",
    r"(^|\n)\s*(system|sistema|admin|administrador|developer|root)\s*:",
    r"\b(nueva|actualizada|actualizacion de la)\s+instruccion\w*\b|\binstruccion\w* (del sistema )?actualizada",
    r"<\s*/?\s*(system|sistema)\s*>",
    # modos de jailbreak conocidos
    r"\b(modo (desarrollador|dios|sin restricciones|developer)|developer mode|jailbreak|do anything now)\b",
    # "respondé / contestá / actuá sin restricciones" (el pedido de responder sin reglas, no viajar sin ellas)
    r"\b(responde|respondeme|contesta|contestame|actua|habla)\w*\s+sin\s+(ninguna\s+|tus\s+)?(restriccion|regla|filtro|limite|censura)\w*",
]
_REGEX_INYECCION = [re.compile(p) for p in PATRONES_INYECCION]


def detectar_inyeccion(texto: str) -> ResultadoValidacion:
    normalizado = _normalizar(texto)
    for patron in _REGEX_INYECCION:
        if patron.search(normalizado):
            return ResultadoValidacion(False, "inyeccion_prompt", MENSAJE_BLOQUEO_INYECCION)
    return ResultadoValidacion(True)


VALIDADORES_ENTRADA = [validar_longitud, detectar_inyeccion]


def validar_entrada(texto: str) -> ResultadoValidacion:
    """Corre los validadores de entrada en orden; devuelve el primero que bloquea (o válido)."""
    for validador in VALIDADORES_ENTRADA:
        resultado = validador(texto)
        if not resultado.valido:
            return resultado
    return ResultadoValidacion(True)


# ── 3. Enmascarado de datos personales (salida) ──────────────────────────────────────────────────────────────────

# el orden importa: primero lo más específico (email, tarjeta) para que un patrón más general no lo parta.
# Los patrones evitan a propósito los datos del dominio que NO son personales: flight_id (MADFCO-20261205-2),
# fechas (2026-12-05), códigos de reserva (SFA-7K2P9Q), precios y códigos IATA
# nombre y apellido(s) en la MISMA línea (con \s se comería la primera palabra del renglón siguiente)
_NOMBRE = r"[A-ZÁÉÍÓÚÑ][a-záéíóúüñ]+(?:[ \t]+(?:de[ \t]+|del[ \t]+|la[ \t]+)?[A-ZÁÉÍÓÚÑ][a-záéíóúüñ]+){1,3}"
# cada patrón: (tipo, regex). Si el regex tiene un grupo "pre", ese prefijo (la etiqueta, ej. "DNI ") se conserva
# y solo se reemplaza el dato
PATRONES_PII = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    # IBAN (va antes que TARJETA, que si no se come sus bloques de dígitos): país + control + bloques
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b")),
    # tarjeta: 4 bloques de 4 dígitos (con espacio o guion) o 13 a 22 dígitos seguidos (tarjeta, CBU). NO toma una
    # lista de precios ("129 135 182 194 98"), que también son dígitos separados por espacios
    ("TARJETA", re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,7}\b|\b\d{13,22}\b")),
    # teléfono: con prefijo internacional (+34 612 345 678), o 10 dígitos en 3 bloques (11 5555-1234)
    ("TELEFONO", re.compile(
        r"(?<![\w-])\+\d{1,3}[\s-]?(?:\(?\d{1,4}\)?[\s-]?){2,5}\d{2,4}\b|\b\d{2,4}[\s-]\d{3,4}[\s-]\d{4}\b")),
    # documento precedido de su etiqueta: "DNI 30.123.456", "pasaporte: AB123456", "documento nº X1234567".
    # El valor tiene que tener al menos un dígito: si no, "documento de identidad" o "pasaporte vigente" se
    # tomaban como números de documento
    ("DOCUMENTO", re.compile(
        r"(?i)(?P<pre>\b(?:dni|documento|pasaporte|passport|nie|cuit|cuil)\b\s*(?:n(?:ro|°|º|úmero|umero)?\.?\s*)?"
        r":?\s*)(?=[A-Z.\-]*\d)[A-Z0-9][A-Z0-9.\-]{4,14}[A-Z0-9]\b")),
    # pasaporte suelto: 1 a 3 letras + 6 a 9 dígitos (AB1234567)
    ("DOCUMENTO", re.compile(r"\b[A-Z]{1,3}\d{6,9}\b")),
    # DNI argentino con puntos (30.123.456)
    ("DOCUMENTO", re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}\b")),
    # nombre del pasajero cuando viene anunciado: "a nombre de Ana Pérez", "Pasajero: Juan Gómez",
    # "**Nombre:** Ana Pérez", "Nombre completo: ...", "Titular: ..." (con o sin negritas de markdown)
    ("NOMBRE", re.compile(
        r"(?P<pre>\b[Aa] nombre de\s+(?:\*\*)?|(?:\*\*)?\b(?:[Nn]ombre(?: del pasajero| completo)?|[Pp]asajer[oa]|"
        r"[Tt]itular)(?:\*\*)?\s*:\s*(?:\*\*)?\s*)" + _NOMBRE)),
]


def enmascarar_valores(texto: str, valores_sensibles: dict[str, list[str]]) -> tuple[str, list[str]]:
    """Enmascara valores EXACTOS ya conocidos (ej. el nombre y el documento que el agente le pasó a crear_reserva).

    Es la capa más confiable: no depende de cómo redacte el modelo la respuesta, porque busca el dato literal.
    """
    encontrados = []
    for tipo, valores in valores_sensibles.items():
        for valor in sorted({v.strip() for v in valores if v and len(v.strip()) >= 4}, key=len, reverse=True):
            patron = re.compile(re.escape(valor), re.IGNORECASE)
            texto, cantidad = patron.subn(f"[oculto: {tipo.lower()}]", texto)
            if cantidad:
                encontrados.append(tipo)
    return texto, encontrados


def enmascarar_pii(texto: str, valores_sensibles: dict[str, list[str]] | None = None) -> tuple[str, list[str]]:
    """Devuelve (texto enmascarado, tipos de dato encontrados). No bloquea: la respuesta llega sin el dato.

    Primero reemplaza los valores sensibles conocidos (si se pasan) y después aplica los patrones generales.
    """
    if not texto:
        return texto, []
    encontrados = []
    if valores_sensibles:
        texto, encontrados = enmascarar_valores(texto, valores_sensibles)
    for tipo, patron in PATRONES_PII:
        if "pre" in patron.groupindex:
            nuevo, cantidad = patron.subn(lambda m, t=tipo: f"{m.group('pre')}[oculto: {t.lower()}]", texto)
        else:
            nuevo, cantidad = patron.subn(f"[oculto: {tipo.lower()}]", texto)
        if cantidad:
            encontrados.append(tipo)
            texto = nuevo
    return texto, sorted(set(encontrados))
