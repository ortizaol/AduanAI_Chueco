"""
Persistent company profile — saved to a local JSON file so users
don't have to re-enter context on every session.
"""

import json
from pathlib import Path

PROFILE_PATH = Path(__file__).parent / ".user_profile.json"

SECTORS = [
    "Fabricación de autobuses / carrocerías",
    "Autopartes / vehículos",
    "Textil / confección",
    "Alimentos y bebidas",
    "Construcción / materiales",
    "Maquinaria y equipo industrial",
    "Químicos / plásticos / caucho",
    "Electrónica / tecnología",
    "Retail / comercio general",
    "Otro",
]

PRODUCT_TYPES = [
    "Partes y piezas para fabricación",
    "Materia prima / insumos",
    "Maquinaria y equipos",
    "Productos terminados para venta",
    "Mixto",
]

ORIGINS = [
    "Asia (China, Corea, India, etc.)",
    "Europa",
    "América del Norte (EE.UU., Canadá, México)",
    "América Latina",
    "Mixto / varios",
]


def load() -> dict:
    if PROFILE_PATH.exists():
        try:
            return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"sector": SECTORS[0], "product_type": PRODUCT_TYPES[0], "origin": ORIGINS[0]}


def save(data: dict) -> None:
    PROFILE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def as_text(profile: dict) -> str:
    return (
        f"Sector: {profile.get('sector', '—')}\n"
        f"Tipo de importaciones: {profile.get('product_type', '—')}\n"
        f"Origen principal de proveedores: {profile.get('origin', '—')}"
    )
