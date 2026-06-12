"""
Document extractor — reads PDFs and images with a vision-capable LLM
and returns structured product information for tariff classification.

Supports up to 4 files per call.
Anthropic: native PDF + image support.
OpenAI: images natively; PDFs converted to PNG pages via PyMuPDF.
"""

import base64
import json
import re
from pathlib import Path

EXTRACTOR_SYSTEM = """Eres experto en clasificación arancelaria del Arancel de Colombia (Decreto 1881/2021).
Se te proporcionan documentos técnicos (fichas técnicas, planos, catálogos, fotografías) de un producto.

Analiza TODO el contenido y extrae la información necesaria para clasificarlo correctamente.

Responde ÚNICAMENTE con JSON válido, sin markdown:
{
  "product_name": "nombre técnico/comercial del producto",
  "material": "materiales principales de construcción (sé específico: ABS, acero inoxidable, EPDM, etc.)",
  "application": "uso final y sector de aplicación (sé específico: asiento para autobús, resorte de gas para mecanismo, etc.)",
  "tech_specs": "especificaciones técnicas clave: dimensiones, capacidades, normas, temperaturas, etc.",
  "suggested_description": "descripción de 1-2 líneas usando terminología del arancel de aduanas, incluyendo material y uso"
}"""


def _encode_image(data: bytes, mime: str) -> dict:
    b64 = base64.standard_b64encode(data).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}


def _encode_pdf_anthropic(data: bytes) -> dict:
    b64 = base64.standard_b64encode(data).decode()
    return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}}


def _pdf_to_images(data: bytes, max_pages: int = 3) -> list[bytes]:
    """Convert first N pages of a PDF to PNG bytes using PyMuPDF."""
    import fitz  # pymupdf
    doc = fitz.open(stream=data, filetype="pdf")
    images = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        mat = fitz.Matrix(1.5, 1.5)  # 1.5× zoom → ~108 DPI
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def _build_content_anthropic(files: list[dict], company_context: str) -> list:
    content = []
    for f in files:
        mime = f["mime"]
        data = f["data"]
        name = f["name"]
        content.append({"type": "text", "text": f"Documento: {name}"})
        if mime == "application/pdf":
            content.append(_encode_pdf_anthropic(data))
        else:
            content.append(_encode_image(data, mime))
    content.append({
        "type": "text",
        "text": f"Contexto de la empresa importadora:\n{company_context}\n\nExtrae la información del producto."
    })
    return content


def _build_content_openai(files: list[dict], company_context: str) -> list:
    content = []
    for f in files:
        mime = f["mime"]
        data = f["data"]
        name = f["name"]
        content.append({"type": "text", "text": f"Documento: {name}"})
        if mime == "application/pdf":
            # Convert PDF pages to images
            page_images = _pdf_to_images(data, max_pages=2)
            for img_bytes in page_images:
                b64 = base64.standard_b64encode(img_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
                })
        else:
            b64 = base64.standard_b64encode(data).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}
            })
    content.append({
        "type": "text",
        "text": f"Contexto de la empresa importadora:\n{company_context}\n\nExtrae la información del producto."
    })
    return content


def _parse(raw: str) -> dict:
    raw = re.sub(r"^```json\s*", "", raw.strip())
    raw = re.sub(r"```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"Respuesta no es JSON válido: {raw[:200]}")


def extract_anthropic(files: list[dict], company_context: str, api_key: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    content = _build_content_anthropic(files, company_context)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=EXTRACTOR_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return _parse(resp.content[0].text)


def extract_openai(files: list[dict], company_context: str, api_key: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    content = _build_content_openai(files, company_context)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=512,
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
    )
    return _parse(resp.choices[0].message.content)


def extract(files: list[dict], company_context: str, provider: str, api_key: str) -> dict:
    """
    files: list of {"name": str, "mime": str, "data": bytes}
    Returns: {"product_name", "material", "application", "tech_specs", "suggested_description"}
    """
    if provider == "anthropic":
        return extract_anthropic(files, company_context, api_key)
    else:
        return extract_openai(files, company_context, api_key)
