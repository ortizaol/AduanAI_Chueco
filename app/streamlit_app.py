"""
AduanAI – Clasificador Arancelario Colombia
Streamlit app: Query expansion → BM25 retrieval → single LLM classification

Run:  streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from retriever import retrieve, get_relevant_notes
from classifier import classify_anthropic, classify_openai
from expander import expand_query

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AduanAI – Clasificador Arancelario",
    page_icon="🛃",
    layout="wide",
)

st.markdown("""
<style>
  .result-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 5px solid #0284c7;
    border-radius: 8px;
    padding: 20px 24px;
    margin-top: 12px;
  }
  .code-big { font-size: 2rem; font-weight: 700; color: #0284c7; font-family: monospace; }
  .desc-text { font-size: 1.1rem; color: #334155; margin-top: 6px; }
  .rgi-chip {
    display: inline-block;
    background: #e0f2fe; color: #0369a1;
    border-radius: 4px; padding: 2px 8px; margin: 2px;
    font-size: 0.82rem; font-weight: 600;
  }
  .conf-alta  { color: #16a34a; font-weight: 700; }
  .conf-media { color: #d97706; font-weight: 700; }
  .conf-baja  { color: #dc2626; font-weight: 700; }
  .step-label { color: #6b7280; font-size: 0.8rem; text-transform: uppercase;
                letter-spacing: .05em; margin-bottom: 2px; }
  .expanded-query { font-family: monospace; font-size: 0.82rem; color: #475569;
                    background: #f8fafc; border-radius: 4px; padding: 6px 10px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuración")

    provider_label = st.radio("Proveedor de IA", ["Anthropic (Claude)", "OpenAI (GPT)"])
    provider = "anthropic" if "Anthropic" in provider_label else "openai"

    if provider == "anthropic":
        api_key = st.text_input("API Key Anthropic", type="password")
        main_model = "claude-sonnet-4-6"
        router_model = "claude-haiku-4-5-20251001"
    else:
        api_key = st.text_input("API Key OpenAI", type="password")
        main_model = "gpt-4.1-mini"
        router_model = "gpt-4.1-mini"

    st.caption(f"Clasificación: `{main_model}`  \nExpansión: `{router_model}`")

    st.divider()
    top_k = st.slider("Candidatos BM25", 8, 25, 15,
                       help="Cuántos códigos candidatos se pasan al LLM")

    st.divider()
    st.caption(
        "**Arancel de Colombia** — Decreto 1881/2021 · SA 2017  \n"
        "7.941 códigos hoja · BM25 + expansión semántica + RGI 1–6"
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🛃 AduanAI — Clasificador Arancelario Colombia")

query = st.text_area(
    "Descripción del producto",
    placeholder=(
        "Ej: Tanque de combustible en polietileno para autobús\n"
        "Ej: Racor recto 8×1/4 sistema contra incendios Fogmaker\n"
        "Ej: Tinta azul rojizo de baja intensidad PT125\n"
        "Ej: Silla pasajero izquierda doble pata Promiurban"
    ),
    height=90,
)

classify_btn = st.button("Clasificar →", type="primary")

# ── Pipeline ──────────────────────────────────────────────────────────────────
if classify_btn:
    if not query.strip():
        st.warning("Ingresa la descripción del producto.")
        st.stop()
    if not api_key:
        st.warning("Ingresa tu API Key en la barra lateral.")
        st.stop()

    col_prog, _ = st.columns([3, 1])

    # Step 1: Query expansion
    with st.spinner("Paso 1/3 — Expandiendo vocabulario arancelario…"):
        expanded = expand_query(query.strip(), provider, api_key)

    with st.expander("📝 Consulta expandida", expanded=False):
        st.markdown(
            f'<div class="expanded-query">{expanded}</div>',
            unsafe_allow_html=True,
        )

    # Step 2: BM25 retrieval
    with st.spinner("Paso 2/3 — Recuperando candidatos con BM25…"):
        candidates = retrieve(expanded, top_k=top_k)

    with st.expander(f"🔍 {len(candidates)} candidatos recuperados", expanded=False):
        for i, c in enumerate(candidates, 1):
            st.markdown(
                f"`{i:2d}.` **{c['code']}** — {c['breadcrumb'][:110]}",
            )

    # Step 3: Collect notes + LLM classification
    with st.spinner("Paso 3/3 — Clasificando con LLM (RGI 1–6)…"):
        codes = [c["code"] for c in candidates]
        notes_block = get_relevant_notes(codes)
        try:
            if provider == "anthropic":
                result = classify_anthropic(query.strip(), candidates, notes_block, api_key)
            else:
                result = classify_openai(query.strip(), candidates, notes_block, api_key)
        except Exception as e:
            st.error(f"Error al llamar la API: {e}")
            st.stop()

    # ── Result ────────────────────────────────────────────────────────────────
    code = result.get("code", "—")
    description = result.get("description", "")
    rgi_applied = result.get("rgi_applied", [])
    reasoning = result.get("reasoning", "")
    confidence = result.get("confidence", "media").lower()
    exclusions = result.get("exclusions_checked", [])

    rgi_chips = "".join(f'<span class="rgi-chip">{r}</span>' for r in rgi_applied)
    conf_class = f"conf-{confidence}"

    st.markdown(f"""
<div class="result-box">
  <div class="code-big">{code}</div>
  <div class="desc-text">{description}</div>
  <div style="margin-top:10px;line-height:2">
    <span style="color:#6b7280;font-size:0.82rem">RGI: </span>{rgi_chips}
    &nbsp;
    <span style="color:#6b7280;font-size:0.82rem">Confianza: </span>
    <span class="{conf_class}">{confidence.upper()}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Razonamiento")
    st.write(reasoning)

    if exclusions:
        st.markdown("#### Exclusiones verificadas")
        for ex in exclusions:
            st.markdown(f"- {ex}")

    if notes_block:
        with st.expander("📋 Notas arancelarias utilizadas"):
            st.text(notes_block[:10000] + ("…" if len(notes_block) > 10000 else ""))
