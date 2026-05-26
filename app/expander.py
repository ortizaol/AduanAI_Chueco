"""
Query expansion: cheap LLM call that rewrites user product descriptions
into formal tariff vocabulary before BM25 retrieval.

This bridges vocabulary gaps like:
  "racor" → "accesorio tubería empalme manguito"
  "tinta azul rojizo" → "pintura barniz polímeros sintéticos colorante"
  "tanque combustible" → "depósito carburante tanque vehículo"
"""

EXPANDER_SYSTEM = """Eres experto en el Arancel de Aduanas de Colombia (Decreto 1881/2021).
Tu tarea es reescribir descripciones de productos usando la terminología técnica y legal
exacta del arancel.

Responde SOLO con una lista de 8-12 términos arancelarios separados por espacio.
No uses signos de puntuación. No expliques nada. Solo los términos.

Ejemplos:
- "racor recto sistema contra incendios" → "accesorio tubería empalme manguito acero fundición hierro"
- "tinta azul rojizo" → "pintura barniz polímeros sintéticos colorante pigmento"
- "tanque combustible polietileno autobus" → "depósito carburante tanques polietileno vehículo automóvil"
- "silla pasajero autobus" → "asiento vehículo automóvil pasajeros transporte"
- "difusor ventilación plástico" → "artículo plástico ventilación distribuidor aire manufactura"
"""


def expand_query_anthropic(query: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=80,
        system=EXPANDER_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )
    expanded = resp.content[0].text.strip()
    # Combine original + expanded to keep both signals
    return query + " " + expanded


def expand_query_openai(query: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=80,
        messages=[
            {"role": "system", "content": EXPANDER_SYSTEM},
            {"role": "user", "content": query},
        ],
    )
    expanded = resp.choices[0].message.content.strip()
    return query + " " + expanded


def expand_query(query: str, provider: str, api_key: str) -> str:
    try:
        if provider == "anthropic":
            return expand_query_anthropic(query, api_key)
        else:
            return expand_query_openai(query, api_key)
    except Exception:
        return query  # fall back to original query silently
