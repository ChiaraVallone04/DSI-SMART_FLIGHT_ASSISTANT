"""Parte A — las tres herramientas del agente: el retriever de la Entrega 3, un sensor y un actuador.

- buscar_info_rutas (conocimiento): el RAG avanzado de la Entrega 3 sobre el catálogo de rutas.
- consultar_disponibilidad (sensor): LEE el estado del mundo -> vuelos, horarios, precios y asientos de una fecha.
- crear_reserva (actuador): CAMBIA el estado del mundo -> guarda una reserva y descuenta el asiento.

El docstring de cada herramienta es lo que el LLM lee para decidir cuándo usarla, así que dice para qué sirve,
cuándo usarla, cuándo NO, qué recibe y qué devuelve.

El sensor no tiene una API real de aerolíneas detrás: simula la oferta del día a partir de los metadatos de la
ruta en ChromaDB (categoría de precio, si hay vuelo directo, aerolíneas que la operan) con una semilla fija por
ruta y fecha. Así la misma consulta devuelve siempre los mismos vuelos (la matriz de validación es reproducible)
y los asientos sí cambian cuando el actuador crea reservas, porque se descuentan de la base de reservas.
"""
import hashlib
import json
import random
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta

from langchain_core.tools import tool

from src.config import RUTA_RESERVAS
from src.retrieval import FRASE_ESCAPE, rag_chain_agente

from vector_db import obtener_coleccion  # noqa: E402  (Entrega 2; src.config agrega entrega_2/ al path)

# ── Datos del sensor ─────────────────────────────────────────────────────────────────────────────────────────────

# rango de precio base (EUR, ida, clase económica) según categoria_precio de la ruta
RANGOS_PRECIO = {"económico": (25, 90), "medio": (70, 220), "premium": (180, 520)}
HORARIOS = ["06:10", "07:25", "08:40", "10:05", "11:50", "13:15", "15:30", "17:05", "18:45", "20:20", "21:55"]
AEROLINEAS_CONOCIDAS = [
    "Iberia", "Air Europa", "Vueling", "ITA Airways", "Ryanair", "Wizz Air", "easyJet", "British Airways",
    "TAP", "Aegean", "Lufthansa", "Turkish Airlines", "Pegasus", "SAS", "Norwegian", "Finnair", "LOT",
    "Icelandair", "Aurigny", "Blue Islands", "Air France", "Transavia", "KLM", "Swiss", "airBaltic", "Aer Lingus",
    "Air Algérie", "Azerbaijan Airlines", "Belavia", "Volotea", "Eurowings",
]
AEROLINEAS_POR_TIPO = {
    "low-cost": ["Ryanair", "Wizz Air", "easyJet"],
    "tradicional": ["Iberia", "Lufthansa", "British Airways"],
    "mixto": ["Vueling", "Iberia", "Ryanair"],
}
DIAS_MAXIMOS_ANTICIPACION = 330
PATRON_FLIGHT_ID = re.compile(r"^([A-Z]{3})([A-Z]{3})-(\d{8})-(\d)$")


def _catalogo():
    """Rutas del catálogo tal como están HOY en ChromaDB (incluye los eventos en caliente de la Entrega 2)."""
    resultado = obtener_coleccion().get(include=["metadatas", "documents"])
    rutas = {}
    for id_ruta, meta, texto in zip(resultado["ids"], resultado["metadatas"], resultado["documents"]):
        rutas[frozenset((meta["origen"], meta["destino"]))] = {"id": id_ruta, "texto": texto, **meta}
    return rutas


def _aerolineas_de(ruta: dict):
    mencionadas = [a for a in AEROLINEAS_CONOCIDAS if a.lower() in ruta["texto"].lower()]
    return mencionadas or AEROLINEAS_POR_TIPO.get(ruta.get("tipo_aerolinea_dominante"), ["Aerolínea regional"])


def _rng(*partes: str):
    """Generador aleatorio con semilla fija por ruta y fecha: misma consulta -> mismos vuelos."""
    semilla = int(hashlib.sha256("|".join(partes).encode()).hexdigest()[:12], 16)
    return random.Random(semilla)


# ── Base de reservas (la que modifica el actuador) ───────────────────────────────────────────────────────────────

@contextmanager
def _conexion_reservas():
    """Conexión en modo autocommit (las transacciones se abren a mano) que siempre se cierra al salir."""
    conexion = sqlite3.connect(RUTA_RESERVAS, timeout=10, isolation_level=None)
    try:
        conexion.execute("""
            CREATE TABLE IF NOT EXISTS reservas (
                codigo TEXT PRIMARY KEY,
                flight_id TEXT NOT NULL,
                origen TEXT NOT NULL,
                destino TEXT NOT NULL,
                fecha TEXT NOT NULL,
                hora TEXT NOT NULL,
                aerolinea TEXT NOT NULL,
                precio_eur REAL NOT NULL,
                pasajero_nombre TEXT NOT NULL,
                pasajero_documento TEXT NOT NULL,
                email TEXT,
                creada_en TEXT NOT NULL
            )""")
        yield conexion
    finally:
        conexion.close()


def reservas_del_vuelo(flight_id: str) -> int:
    with _conexion_reservas() as conexion:
        return conexion.execute("SELECT COUNT(*) FROM reservas WHERE flight_id = ?", (flight_id,)).fetchone()[0]


def listar_reservas():
    with _conexion_reservas() as conexion:
        conexion.row_factory = sqlite3.Row
        return [dict(fila) for fila in conexion.execute("SELECT * FROM reservas ORDER BY creada_en")]


# ── Lógica del sensor (función común: la usan el sensor y el actuador para validar el vuelo) ─────────────────────

def _validar_fecha(fecha: str):
    try:
        dia = datetime.strptime(fecha, "%Y-%m-%d").date()
    except ValueError:
        return None, f"Fecha inválida '{fecha}': usar el formato AAAA-MM-DD (ej. 2026-12-05)."
    hoy = date.today()
    if dia < hoy:
        return None, f"La fecha {fecha} ya pasó (hoy es {hoy.isoformat()}). Pedile al usuario una fecha futura."
    if dia > hoy + timedelta(days=DIAS_MAXIMOS_ANTICIPACION):
        return None, f"Solo hay vuelos publicados hasta {DIAS_MAXIMOS_ANTICIPACION} días adelante."
    return dia, None


def simular_vuelos(origen: str, destino: str, fecha: str, descontar_reservas: bool = True):
    """Devuelve (vuelos, error). Los vuelos son dicts con flight_id, horario, precio y asientos libres.

    Con descontar_reservas=False devuelve los asientos publicados (sin restar reservas): lo usa el actuador, que
    hace la resta él mismo dentro de una transacción."""
    origen, destino = origen.strip().upper(), destino.strip().upper()
    dia, error = _validar_fecha(fecha.strip())
    if error:
        return None, error

    ruta = _catalogo().get(frozenset((origen, destino)))
    if ruta is None:
        disponibles = sorted(f"{r['origen']}-{r['destino']}" for r in _catalogo().values())
        return None, (
            f"La ruta {origen}-{destino} no está en el catálogo. Rutas disponibles (en cualquier sentido): "
            + ", ".join(disponibles)
        )

    rng = _rng(origen, destino, dia.isoformat())
    aerolineas = _aerolineas_de(ruta)
    minimo, maximo = RANGOS_PRECIO.get(ruta.get("categoria_precio"), RANGOS_PRECIO["medio"])
    # comprar con poca anticipación o volar viernes/domingo sale más caro
    recargo = (1.35 if (dia - date.today()).days < 7 else 1.0) * (1.15 if dia.weekday() in (4, 6) else 1.0)

    vuelos = []
    for n, hora in enumerate(sorted(rng.sample(HORARIOS, rng.randint(2, 4))), 1):
        directo = bool(ruta.get("vuelo_directo_disponible")) and rng.random() < 0.85
        flight_id = f"{origen}{destino}-{dia:%Y%m%d}-{n}"
        asientos_publicados = rng.choice([0, 2, 5, 9, 14, 23])
        vuelos.append({
            "flight_id": flight_id,
            "aerolinea": rng.choice(aerolineas),
            "fecha": dia.isoformat(),
            "hora_salida": hora,
            "escalas": 0 if directo else 1,
            "precio_eur": round(rng.uniform(minimo, maximo) * recargo * (0.9 if not directo else 1.0)),
            "asientos_disponibles": (max(0, asientos_publicados - reservas_del_vuelo(flight_id))
                                     if descontar_reservas else asientos_publicados),
        })
    return {"ruta": ruta["id"], "origen": origen, "destino": destino, "vuelos": vuelos}, None


# ── Las tres herramientas ────────────────────────────────────────────────────────────────────────────────────────

@tool
def buscar_info_rutas(consulta: str) -> str:
    """Busca información GENERAL y estable del catálogo de rutas aéreas europeas (base de conocimiento RAG).

    Usar SIEMPRE que el usuario pregunte por características de una ruta: qué aerolíneas la operan, duración
    típica, si suele haber vuelo directo, rango y mediana de precios históricos, días y horarios con más
    vuelos, clases disponibles, temporadas, o para comparar rutas entre sí. También sirve para averiguar
    qué rutas existen y sus códigos IATA.

    NO usar para disponibilidad, precio o asientos de una fecha concreta (para eso: consultar_disponibilidad).

    Args:
        consulta: la pregunta en lenguaje natural, autocontenida (con las ciudades o rutas nombradas,
            sin pronombres como "esa ruta").

    Returns:
        Una respuesta redactada solo con datos del catálogo, citando entre corchetes el ID de la ruta
        (ej. [RUTA-MAD-FCO]). Si el catálogo no tiene la información, devuelve exactamente la frase
        "No poseo información para responder a esta consulta con los datos disponibles."
    """
    resultado = rag_chain_agente.invoke(consulta, config={"run_name": "tool_buscar_info_rutas"})
    fuentes = sorted({d.metadata["id"] for d in resultado["context"]})
    respuesta = resultado["answer"]
    if respuesta.strip() == FRASE_ESCAPE or not fuentes:
        return respuesta
    return f"{respuesta}\n(Fuentes: {', '.join(fuentes)})"


@tool
def consultar_disponibilidad(origen: str, destino: str, fecha: str) -> str:
    """Sensor: consulta los vuelos publicados para una ruta y una fecha concreta, con horario, precio y asientos.

    Usar cuando el usuario quiere viajar en un día específico, pregunta cuánto sale volar una fecha, si quedan
    lugares, o antes de crear una reserva (para obtener el flight_id). Es la única fuente de precios y
    asientos del día: no inventarlos.

    NO usar para preguntas generales sobre la ruta (para eso: buscar_info_rutas).

    Args:
        origen: código IATA de 3 letras del aeropuerto de salida (ej. "MAD").
        destino: código IATA de 3 letras del aeropuerto de llegada (ej. "FCO").
        fecha: fecha del vuelo en formato AAAA-MM-DD (convertir "el 5 de diciembre" o "el viernes que viene"
            usando la fecha de hoy que figura en las instrucciones).

    Returns:
        JSON con la ruta y la lista de vuelos (flight_id, aerolinea, hora_salida, escalas, precio_eur,
        asientos_disponibles). Un vuelo con 0 asientos está agotado. Si la ruta no existe o la fecha es
        inválida, devuelve un texto que empieza con "Error:" explicando qué corregir.
    """
    datos, error = simular_vuelos(origen, destino, fecha)
    if error:
        return f"Error: {error}"
    return json.dumps(datos, ensure_ascii=False)


@tool
def crear_reserva(flight_id: str, pasajero_nombre: str, pasajero_documento: str, email: str | None = None) -> str:
    """Actuador: crea una reserva REAL de un asiento en un vuelo y la guarda en el sistema de reservas.

    Es una acción con efectos (ocupa un asiento). Usarla SOLO cuando se cumplan las tres condiciones:
    1. el flight_id salió de una llamada previa a consultar_disponibilidad y tiene asientos disponibles;
    2. el usuario dio su nombre completo y su documento o pasaporte;
    3. el usuario confirmó explícitamente que quiere reservar ESE vuelo (ej. "sí, reservalo").
    Si falta algo, NO llamar a esta herramienta: pedirle al usuario el dato o la confirmación que falta.

    Args:
        flight_id: identificador exacto del vuelo devuelto por consultar_disponibilidad (ej. "MADFCO-20261205-2").
        pasajero_nombre: nombre y apellido del pasajero.
        pasajero_documento: número de documento o pasaporte (6 a 12 letras o dígitos).
        email: correo de contacto (opcional).

    Returns:
        Texto de confirmación con el código de reserva (formato SFA-XXXXXX), o un texto que empieza con
        "Error:" si el vuelo no existe, está agotado o algún dato del pasajero es inválido.
    """
    coincidencia = PATRON_FLIGHT_ID.match(flight_id.strip().upper())
    if not coincidencia:
        return "Error: flight_id inválido. Usar exactamente uno devuelto por consultar_disponibilidad."
    origen, destino, fecha_compacta, _ = coincidencia.groups()
    fecha = f"{fecha_compacta[:4]}-{fecha_compacta[4:6]}-{fecha_compacta[6:]}"

    datos, error = simular_vuelos(origen, destino, fecha, descontar_reservas=False)
    if error:
        return f"Error: {error}"
    vuelo = next((v for v in datos["vuelos"] if v["flight_id"] == flight_id.strip().upper()), None)
    if vuelo is None:
        return f"Error: el vuelo {flight_id} no existe para esa fecha. Volver a consultar la disponibilidad."
    agotado = f"Error: el vuelo {flight_id} está agotado. Ofrecer otro vuelo de la misma fecha."
    asientos_publicados = vuelo["asientos_disponibles"]
    if asientos_publicados <= 0:
        return agotado

    nombre = " ".join(pasajero_nombre.split())
    if len(nombre.split()) < 2 or not re.fullmatch(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ' -]+", nombre):
        return "Error: el nombre del pasajero debe tener nombre y apellido (solo letras)."
    documento = re.sub(r"[\s.\-]", "", pasajero_documento).upper()
    if not re.fullmatch(r"[A-Z0-9]{6,12}", documento):
        return "Error: el documento debe tener entre 6 y 12 letras o dígitos."
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", email.strip()):
        return "Error: el email no tiene un formato válido."

    codigo = "SFA-" + "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
    # contar los asientos ocupados e insertar la reserva en UNA transacción: BEGIN IMMEDIATE toma el lock de
    # escritura de SQLite, así dos reservas simultáneas del último asiento no pueden pasar las dos (la segunda
    # espera y, cuando cuenta, ya ve el asiento ocupado). Funciona aunque haya varios procesos
    with _conexion_reservas() as conexion:
        conexion.execute("BEGIN IMMEDIATE")
        try:
            ocupados = conexion.execute(
                "SELECT COUNT(*) FROM reservas WHERE flight_id = ?", (vuelo["flight_id"],)).fetchone()[0]
            if ocupados >= asientos_publicados:
                conexion.execute("ROLLBACK")
                return agotado
            conexion.execute(
                "INSERT INTO reservas VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (codigo, vuelo["flight_id"], origen, destino, fecha, vuelo["hora_salida"], vuelo["aerolinea"],
                 vuelo["precio_eur"], nombre, documento, email.strip() if email else None,
                 datetime.now().isoformat(timespec="seconds")),
            )
            conexion.execute("COMMIT")
        except Exception:
            conexion.execute("ROLLBACK")
            raise
    return (
        f"Reserva confirmada. Código: {codigo}. Vuelo {vuelo['flight_id']} de {vuelo['aerolinea']}, "
        f"{origen}->{destino}, {fecha} a las {vuelo['hora_salida']}, {vuelo['precio_eur']} EUR, "
        f"a nombre de {nombre}."
    )


HERRAMIENTAS = [buscar_info_rutas, consultar_disponibilidad, crear_reserva]


if __name__ == "__main__":
    # prueba rápida de las tres herramientas, sin el agente
    from datetime import date as _date

    fecha_prueba = (_date.today() + timedelta(days=30)).isoformat()
    print("== buscar_info_rutas ==")
    print(buscar_info_rutas.invoke({"consulta": "¿Qué aerolíneas operan la ruta Madrid–Roma?"}))
    print("\n== consultar_disponibilidad ==")
    salida = consultar_disponibilidad.invoke({"origen": "MAD", "destino": "FCO", "fecha": fecha_prueba})
    print(salida)
    print(consultar_disponibilidad.invoke({"origen": "EZE", "destino": "NRT", "fecha": fecha_prueba})[:120], "...")
    print("\n== crear_reserva ==")
    vuelo = next(v for v in json.loads(salida)["vuelos"] if v["asientos_disponibles"] > 0)
    print(crear_reserva.invoke({"flight_id": vuelo["flight_id"], "pasajero_nombre": "Ana Prueba",
                                "pasajero_documento": "AB123456"}))
    print("asientos después:", next(v for v in json.loads(consultar_disponibilidad.invoke(
        {"origen": "MAD", "destino": "FCO", "fecha": fecha_prueba}))["vuelos"]
        if v["flight_id"] == vuelo["flight_id"])["asientos_disponibles"], "(antes:", vuelo["asientos_disponibles"], ")")
