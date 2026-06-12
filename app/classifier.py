"""
Single LLM call classifier.

Given a product description and BM25 candidates, asks the LLM to:
1. Apply RGI 1-6
2. Choose the best tariff code from the candidates
3. Explain the reasoning

Returns structured dict: {code, description, rgi_applied, reasoning, confidence}
"""

import json
import re

SYSTEM = """Eres un experto en clasificación arancelaria de Colombia (Decreto 1881/2021, Arancel de Aduanas basado en el SA 2017).

━━━ REGLAS GENERALES DE INTERPRETACIÓN (RGI) — APLICACIÓN OBLIGATORIA ━━━

RGI 1. Los títulos de Secciones, Capítulos o Subcapítulos solo tienen valor indicativo. La clasificación se determina por los textos de las partidas y las notas de Sección o Capítulo.

RGI 2a. Cualquier referencia a un artículo en una partida alcanza al artículo incompleto o sin terminar, siempre que presente las características esenciales del artículo completo o terminado. También alcanza al artículo completo o terminado presentado desmontado o sin montar.

RGI 2b. Cualquier referencia a una materia en una partida alcanza a dicha materia en estado puro o mezclada o asociada con otras materias.

RGI 3a. Cuando dos o más partidas puedan aplicarse a un mismo artículo, la partida más específica tendrá prioridad sobre las más genéricas.

RGI 3b. Los productos mezclados o las manufacturas compuestas de materias diferentes o constituidos por la unión de artículos diferentes que puedan clasificarse en partidas distintas se clasificarán según la materia o artículo que les confiera el carácter esencial.

RGI 3c. Cuando las reglas 3a y 3b no permitan determinar la clasificación, la partida situada en último lugar entre las que igualmente pudieran tenerse en cuenta.

RGI 4. Las mercancías que no puedan clasificarse aplicando las reglas anteriores se clasificarán en la partida correspondiente a los artículos con los que tengan mayor analogía.

RGI 5a. Los estuches para cámaras fotográficas, instrumentos musicales y artículos similares, especialmente acondicionados para contener un artículo determinado, susceptibles de uso prolongado, presentados con los artículos a los que están destinados, se clasifican con dichos artículos.

RGI 5b. Los envases que contengan mercancías se clasifican con ellas cuando sean del tipo normalmente utilizado para esa clase de mercancías.

RGI 6. La clasificación de mercancías en las subpartidas de una misma partida está determinada por los textos de las subpartidas y las notas de subpartida.

━━━ CRITERIOS CLAVE ━━━

1. NOTAS DE EXCLUSIÓN: Si una nota de Sección/Capítulo excluye expresamente el producto, ese código NO puede usarse.

2. PARTES Y ACCESORIOS: Las partes no mencionadas expresamente se clasifican por su constitución o función principal, salvo que una nota las asigne a una partida específica.

3. USO PRINCIPAL: "Diseñado exclusiva o principalmente para X" (Nota Sección XVII) aplica solo cuando el artículo NO tiene uso comercial generalizado fuera de ese destino.

4. ESPECIFICIDAD (RGI 3a): Una partida que describe el artículo como tal tiene prioridad sobre partidas de materia constitutiva o categoría genérica.

━━━ FORMATO DE RESPUESTA ━━━

Responde ÚNICAMENTE con JSON válido, sin markdown, sin explicaciones fuera del JSON:

{
  "code": "XXXX.XX.XX.XX",
  "description": "descripción del código elegido",
  "rgi_applied": ["RGI 1", "RGI 3a"],
  "reasoning": "Explicación clara de por qué este código y no los otros candidatos. Menciona las notas relevantes que aplican.",
  "confidence": "alta|media|baja",
  "exclusions_checked": ["nota que excluía tal código", "..."]
}

Si ningún candidato es correcto, elige el más cercano y marca confidence como "baja" explicando el problema.
"""


def build_prompt(description: str, candidates: list[dict], notes_block: str) -> str:
    candidate_lines = []
    for i, c in enumerate(candidates, 1):
        candidate_lines.append(f"  {i:2d}. [{c['code']}] {c['breadcrumb']}")

    candidates_text = "\n".join(candidate_lines)

    prompt = f"""Producto a clasificar: {description}

━━━ CANDIDATOS RECUPERADOS (BM25 + fuzzy) ━━━
{candidates_text}

━━━ NOTAS DE SECCIÓN Y CAPÍTULO RELEVANTES ━━━
{notes_block if notes_block else "(sin notas adicionales)"}

━━━ INSTRUCCIÓN ━━━
Aplica las RGI en orden. Verifica las notas de exclusión para cada candidato.
Elige el código más específico y correcto de la lista de candidatos.
Responde solo con el JSON especificado."""

    return prompt


def classify_anthropic(description: str, candidates: list[dict], notes_block: str, api_key: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(description, candidates, notes_block)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    return _parse_response(raw, candidates)


def classify_openai(description: str, candidates: list[dict], notes_block: str, api_key: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = build_prompt(description, candidates, notes_block)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    return _parse_response(raw, candidates)


def _parse_response(raw: str, candidates: list[dict]) -> dict:
    # Strip markdown fences if model added them
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract first JSON object
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            result = json.loads(m.group())
        else:
            raise ValueError(f"LLM returned non-JSON: {raw[:200]}")

    # Validate code is in candidates
    candidate_codes = [c["code"] for c in candidates]
    if result.get("code") not in candidate_codes:
        # Try partial match
        for c in candidate_codes:
            if c in result.get("code", "") or result.get("code", "") in c:
                result["code"] = c
                break

    return result
