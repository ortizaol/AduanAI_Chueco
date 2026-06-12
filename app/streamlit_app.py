"""
AduanAI – Clasificador Arancelario Colombia

Flujo con documentos:
  Perfil empresa → Upload docs → Extracción LLM → Revisión editable
  → Expansión vocabulario → BM25 → Clasificación LLM (RGI 1–6)

Flujo sin documentos:
  Perfil empresa → Descripción manual → mismo pipeline desde expansión

Run:  streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import profile as prof
from retriever import retrieve, get_relevant_notes
from classifier import classify_anthropic, classify_openai
from expander import expand_query
from extractor import extract

ACCEPTED_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
MIME_MAP = {
    "pdf":  "application/pdf",
    "png":  "image/png",
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AduanAI – Clasificador Arancelario",
    page_icon="🛃",
    layout="wide",
)

st.markdown("""
<style>
  .result-box {
    background:#f0f9ff; border:1px solid #bae6fd;
    border-left:5px solid #0284c7; border-radius:8px; padding:20px 24px; margin-top:12px;
  }
  .code-big  { font-size:2rem; font-weight:700; color:#0284c7; font-family:monospace; }
  .desc-text { font-size:1.05rem; color:#334155; margin-top:6px; }
  .rgi-chip  { display:inline-block; background:#e0f2fe; color:#0369a1;
               border-radius:4px; padding:2px 8px; margin:2px; font-size:.82rem; font-weight:600; }
  .conf-alta  { color:#16a34a; font-weight:700; }
  .conf-media { color:#d97706; font-weight:700; }
  .conf-baja  { color:#dc2626; font-weight:700; }
  .profile-saved { background:#f0fdf4; border:1px solid #bbf7d0;
                   border-radius:6px; padding:8px 12px; font-size:.85rem; color:#15803d; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [("extraction", None), ("profile_saved", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuración")

    provider_label = st.radio("Proveedor de IA", ["Anthropic (Claude)", "OpenAI (GPT)"])
    provider = "anthropic" if "Anthropic" in provider_label else "openai"
    if provider == "anthropic":
        api_key = st.text_input("API Key Anthropic", type="password")
        main_model, router_model = "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
    else:
        api_key = st.text_input("API Key OpenAI", type="password")
        main_model = router_model = "gpt-4.1-mini"
    st.caption(f"Clasificación: `{main_model}` · Extracción/Expansión: `{router_model}`")

    st.divider()
    st.markdown("### 🏭 Perfil de empresa")
    st.caption("Se guarda automáticamente — no hay que volver a ingresarlo.")

    saved_profile = prof.load()
    sector = st.selectbox(
        "Sector industrial", prof.SECTORS,
        index=prof.SECTORS.index(saved_profile.get("sector", prof.SECTORS[0])),
    )
    prod_type = st.selectbox(
        "¿Qué importa principalmente?", prof.PRODUCT_TYPES,
        index=prof.PRODUCT_TYPES.index(saved_profile.get("product_type", prof.PRODUCT_TYPES[0])),
    )
    origin = st.selectbox(
        "Origen principal de proveedores", prof.ORIGINS,
        index=prof.ORIGINS.index(saved_profile.get("origin", prof.ORIGINS[0])),
    )
    profile_data = {"sector": sector, "product_type": prod_type, "origin": origin}

    if st.button("💾 Guardar perfil", use_container_width=True):
        prof.save(profile_data)
        st.session_state.profile_saved = True

    if st.session_state.profile_saved:
        st.markdown('<div class="profile-saved">✓ Perfil guardado</div>', unsafe_allow_html=True)

    st.divider()
    top_k = st.slider("Candidatos BM25", 8, 25, 15)
    st.divider()
    st.caption("**Arancel Colombia** — Decreto 1881/2021 · SA 2017  \n7.941 códigos · BM25 + RGI 1–6")

company_context = prof.as_text(profile_data)


# ── Shared pipeline ───────────────────────────────────────────────────────────
def run_pipeline(enriched_query: str, display_query: str):
    """expand → BM25 → classify → render result."""
    st.divider()
    st.markdown(f"**Clasificando:** _{display_query}_")

    with st.spinner("Paso 1/3 — Expandiendo vocabulario arancelario…"):
        expanded = expand_query(enriched_query, provider, api_key)

    with st.expander("📝 Consulta expandida", expanded=False):
        st.code(expanded, language=None)

    with st.spinner("Paso 2/3 — Recuperando candidatos BM25…"):
        candidates = retrieve(expanded, top_k=top_k)

    with st.expander(f"🔍 {len(candidates)} candidatos recuperados", expanded=False):
        for i, c in enumerate(candidates, 1):
            st.markdown(f"`{i:2d}.` **{c['code']}** — {c['breadcrumb'][:110]}")

    with st.spinner("Paso 3/3 — Clasificando con LLM (RGI 1–6)…"):
        codes = [c["code"] for c in candidates]
        notes_block = get_relevant_notes(codes)
        try:
            if provider == "anthropic":
                result = classify_anthropic(display_query, candidates, notes_block, api_key)
            else:
                result = classify_openai(display_query, candidates, notes_block, api_key)
        except Exception as e:
            st.error(f"Error al clasificar: {e}")
            return

    code        = result.get("code", "—")
    description = result.get("description", "")
    rgi_applied = result.get("rgi_applied", [])
    reasoning   = result.get("reasoning", "")
    confidence  = result.get("confidence", "media").lower()
    exclusions  = result.get("exclusions_checked", [])

    rgi_chips  = "".join(f'<span class="rgi-chip">{r}</span>' for r in rgi_applied)

    st.markdown(f"""
<div class="result-box">
  <div class="code-big">{code}</div>
  <div class="desc-text">{description}</div>
  <div style="margin-top:10px;line-height:2.2">
    <span style="color:#6b7280;font-size:.82rem">RGI: </span>{rgi_chips}
    &nbsp;
    <span style="color:#6b7280;font-size:.82rem">Confianza: </span>
    <span class="conf-{confidence}">{confidence.upper()}</span>
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


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🛃 AduanAI — Clasificador Arancelario Colombia")

tab_docs, tab_text = st.tabs(["📎 Con documentos técnicos", "✏️ Solo descripción"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Document upload + extraction
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("Sube hasta **4 archivos** (fichas técnicas, planos, catálogos, fotos).")
    st.caption("Formatos: PDF · PNG · JPG · WEBP")

    uploaded = st.file_uploader(
        "Archivos del producto",
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded and len(uploaded) > 4:
        st.warning("Máximo 4 archivos — se procesan los primeros 4.")
        uploaded = uploaded[:4]

    extract_btn = st.button(
        "🔍 Extraer información del documento",
        type="secondary",
        disabled=not uploaded or not api_key,
    )

    if extract_btn:
        files_data = [
            {
                "name": uf.name,
                "mime": MIME_MAP.get(uf.name.rsplit(".", 1)[-1].lower(), "application/octet-stream"),
                "data": uf.read(),
            }
            for uf in uploaded
        ]
        with st.spinner("Analizando documentos con visión IA…"):
            try:
                st.session_state.extraction = extract(files_data, company_context, provider, api_key)
            except Exception as e:
                st.error(f"Error en extracción: {e}")

    # Editable form
    if st.session_state.extraction:
        ex = st.session_state.extraction
        st.success("✓ Información extraída — revisa y corrige si es necesario antes de clasificar")

        c1, c2 = st.columns(2)
        with c1:
            product_name = st.text_input("📦 Nombre del producto", value=ex.get("product_name", ""), key="ex_name")
            material     = st.text_input("🧱 Material principal",  value=ex.get("material", ""),      key="ex_mat")
        with c2:
            application  = st.text_input("🔧 Aplicación / uso final",       value=ex.get("application", ""),  key="ex_app")
            tech_specs   = st.text_input("📐 Especificaciones técnicas",     value=ex.get("tech_specs", ""),   key="ex_spec")

        description_doc = st.text_area(
            "📝 Descripción arancelaria (editable — esta se usa para clasificar)",
            value=ex.get("suggested_description", ""),
            height=75,
            key="ex_desc",
        )

        if st.button("Clasificar →", type="primary", key="btn_classify_doc"):
            if not api_key:
                st.warning("Ingresa tu API Key en la barra lateral.")
            else:
                # Enrich query with all extracted fields
                enriched = (
                    f"{product_name}. Material: {material}. "
                    f"Uso: {application}. {tech_specs}. {description_doc}"
                ).strip()
                run_pipeline(enriched, description_doc or enriched)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Manual description
# ══════════════════════════════════════════════════════════════════════════════
with tab_text:
    query = st.text_area(
        "Descripción del producto",
        placeholder=(
            "Ej: Tanque de combustible en polietileno para autobús\n"
            "Ej: Racor recto 8×1/4 sistema contra incendios Fogmaker\n"
            "Ej: Tinta azul rojizo de baja intensidad PT125"
        ),
        height=90,
        key="manual_query",
    )
    if st.button("Clasificar →", type="primary", key="btn_classify_text"):
        if not query.strip():
            st.warning("Ingresa la descripción del producto.")
        elif not api_key:
            st.warning("Ingresa tu API Key en la barra lateral.")
        else:
            run_pipeline(query.strip(), query.strip())
