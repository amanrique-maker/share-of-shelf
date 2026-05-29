import streamlit as st
import anthropic
import base64
import json
import re
import unicodedata
import tempfile
import os
import cv2
import pandas as pd
import plotly.graph_objects as go

# ── Configuración ──────────────────────────────────────────────────────────────
NUESTRAS_MARCAS = {
    "advance", "ultima", "natures variety", "nature's variety",
    "natural trainer", "brekkies", "affinity"
}

COLORES_TECH = {"Dry": "#E8472C", "Wet": "#2196F3", "Snacks": "#FF9800", "Otros": "#BDBDBD"}
COLORES_PET  = {"Perro": "#5C6BC0", "Gato": "#26A69A", "Otros": "#BDBDBD"}
COLORES_SEG  = {
    "Maxi-Medium": "#E8472C", "Mini": "#FF9800",
    "Esterilizado": "#2196F3", "No Esterilizado": "#26A69A", "N/A": "#BDBDBD",
}

def normalizar(texto: str) -> str:
    return unicodedata.normalize("NFD", texto.lower()).encode("ascii", "ignore").decode("ascii")

def es_nuestra(marca: str) -> bool:
    """
    Marca nuestra si el nombre normalizado coincide exactamente
    o empieza por una de nuestras marcas seguida de espacio
    (ej: 'Ultima PRO', 'Advance Sensitive', 'Nature's Variety Instinct').
    """
    m_norm = normalizar(marca)
    return any(
        m_norm == normalizar(n) or m_norm.startswith(normalizar(n) + " ")
        for n in NUESTRAS_MARCAS
    )

def es_video(f) -> bool:
    return f.type.startswith("video/") or f.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))

# ── Prompt ─────────────────────────────────────────────────────────────────────
PROMPT = """Analiza estas imágenes de baldas de una tienda de alimentación para mascotas.
Son fotos consecutivas del MISMO lineal tomadas de izquierda a derecha. Analízalas como un conjunto único y devuelve UN SOLO JSON con todos los productos.

REGLA CRÍTICA — DEDUPLICACIÓN: Las fotos consecutivas se solapan en los bordes, por lo que el mismo producto puede aparecer en 2 fotos a la vez. Cada combinación única de Marca+Pet+Tecnología+Segmento+Formato debe aparecer EXACTAMENTE UNA VEZ en el JSON. Si ves el mismo producto en varias fotos, decide en cuál está más centrado/visible y úsalo solo en esa, ignorándolo en las demás. NUNCA repitas la misma fila.

IMPORTANTE: Sé lo más granular posible. Cada fila representa una combinación única de Marca + Pet + Tecnología + Segmento + Formato. No agrupes productos distintos en una sola fila.

Para cada grupo de productos visibles identifica:

1. MARCA (ej: Advance, Purina, Royal Canin…). Si no se distingue → "Otros".
2. PET: "Perro", "Gato" o "Otros".
3. TECNOLOGÍA — clasifica por el CONTENIDO, no por la forma del envase:
   - "Dry"    → SOLO bolsas o sacos que contengan pienso seco / croquetas
   - "Wet"    → cualquier envase con comida húmeda: latas metálicas, tarrinas, sobres/pouches, brick, tetra pak, cajas de cartón con comida húmeda, tarros de cristal
   - "Snacks" → premios, sticks, palitos, huesos masticables, golosinas, helados para mascotas
   ATENCIÓN: una caja de cartón NO es automáticamente Dry. Fíjate en las imágenes y texto del envase para determinar si es húmedo o snack.
4. SEGMENTO:
   - Perro → "Maxi-Medium" o "Mini"
   - Gato  → "Esterilizado" o "No Esterilizado"
   - Snacks / Otros → "N/A"
5. FORMATO: peso o tamaño (ej: "3 kg", "1.5 kg", "400 g"). Si no se ve → null.
6. FACINGS: unidades visibles de frente (incluye parcialmente tapadas).
7. ANCHO_RELATIVO: anchura proporcional que ocupa este grupo respecto al total de la balda (número entre 0 y 100). Estima visualmente. El conjunto de todos los valores debe sumar aproximadamente 100.
8. PRECIO: precio en la etiqueta de balda (decimal, ej: 24.99). Si no se ve → null.
9. PROMOCION: texto promocional visible (ej: "2x1", "2ª unidad 50%", "-20%"). Si no hay → null.

Devuelve ÚNICAMENTE un JSON válido, sin texto adicional:
{
  "productos": [
    {"marca": "Advance", "pet": "Perro", "tecnologia": "Dry", "segmento": "Maxi-Medium", "formato": "3 kg", "facings": 8, "ancho_relativo": 35.0, "precio": 24.99, "promocion": null},
    {"marca": "Ultima",  "pet": "Gato",  "tecnologia": "Dry", "segmento": "Esterilizado", "formato": "1.5 kg", "facings": 4, "ancho_relativo": 20.0, "precio": 12.50, "promocion": "2x1"}
  ],
  "notas": ""
}"""

# ── Función análisis (múltiples imágenes en una sola llamada) ──────────────────
def analizar_imagenes(imagenes: list[dict]) -> dict:
    """
    imagenes: lista de {"bytes": bytes, "media_type": str}
    Envía todas las imágenes en una sola llamada a Claude.
    """
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or st.session_state.get("_api_key", "")
    if not api_key:
        st.error("❌ Introduce tu API Key de Anthropic en la barra lateral izquierda.")
        st.stop()
    client = anthropic.Anthropic(api_key=api_key)

    content = []
    for img in imagenes:
        img_b64 = base64.standard_b64encode(img["bytes"]).decode("utf-8")
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": img["media_type"], "data": img_b64
        }})
    content.append({"type": "text", "text": PROMPT})

    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=8000,
        messages=[{"role": "user", "content": content}],
    )

    raw = msg.content[0].text.strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            raw = m.group(1).strip()

    st.session_state["debug_raw"] = raw

    result = json.loads(raw)
    result["_usage"] = {
        "input_tokens":  msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }
    return result

# ── Función análisis vídeo ─────────────────────────────────────────────────────
def extraer_frames(video_path: str, intervalo_s: int) -> list[bytes]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * intervalo_s)
    frames = []
    idx = 0
    while idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        _, buf = cv2.imencode(".jpg", frame)
        frames.append(buf.tobytes())
        idx += step
    cap.release()
    return frames

# ── Coste tokens ──────────────────────────────────────────────────────────────
PRECIO_INPUT  = 15.0 / 1_000_000
PRECIO_OUTPUT = 75.0 / 1_000_000

def mostrar_tokens(input_tokens: int, output_tokens: int, n_frames: int = 1):
    coste = input_tokens * PRECIO_INPUT + output_tokens * PRECIO_OUTPUT
    coste_eur = coste * 0.92
    label = f"{n_frames} frame{'s' if n_frames > 1 else ''}" if n_frames > 1 else "análisis"
    st.markdown(f"""
    <div style="background:#f0f7ff;border:1px solid #cce0ff;border-radius:10px;
                padding:12px 18px;margin-top:12px;font-size:0.85rem;color:#444;">
        🔢 <b>Tokens consumidos</b> ({label}) &nbsp;|&nbsp;
        ⬆️ Input: <b>{input_tokens:,}</b> &nbsp;·&nbsp;
        ⬇️ Output: <b>{output_tokens:,}</b> &nbsp;·&nbsp;
        💶 Coste: <b>~{coste_eur:.3f} €</b>
    </div>""", unsafe_allow_html=True)

# ── Helpers de gráficos ────────────────────────────────────────────────────────
def bar_chart(x, y, colores, customdata, ylabel="Share (%)", hover_extra="", height=320):
    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=colores,
        text=[f"{v:.1f}%" for v in y], textposition="outside",
        hovertemplate=f"<b>%{{x}}</b><br>{hover_extra}Share: %{{y:.1f}}%<extra></extra>",
        customdata=customdata,
    ))
    fig.update_layout(
        plot_bgcolor="white", showlegend=False,
        yaxis=dict(gridcolor="#f0f0f0", range=[0, max(y) * 1.25], title=ylabel),
        margin=dict(t=10, b=10), height=height,
    )
    return fig

def stacked_chart(pivot, colores_dict, ylabel="Facings", height=280):
    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(go.Bar(name=col, x=pivot.index, y=pivot[col],
                             marker_color=colores_dict.get(col, "#BDBDBD")))
    fig.update_layout(barmode="stack", plot_bgcolor="white",
                      yaxis=dict(gridcolor="#f0f0f0", title=ylabel),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(t=30, b=10), height=height)
    return fig

# ── Mostrar resultados ─────────────────────────────────────────────────────────
def mostrar_resultados(df: pd.DataFrame, notas: str, origen: str = ""):

    df["es_nuestra"] = df["marca"].apply(es_nuestra)

    total_facings = df["facings"].sum()
    df["sos_facings"] = (df["facings"] / total_facings * 100).round(1) if total_facings > 0 else 0

    total_ancho = df["ancho_relativo"].sum()
    df["sos_ancho"] = (df["ancho_relativo"] / total_ancho * 100).round(1) if total_ancho > 0 else 0

    df_marca = df.groupby(["marca", "es_nuestra"], as_index=False).agg(
        facings=("facings", "sum"),
        ancho=("ancho_relativo", "sum"),
    )
    df_marca["sos_facings"] = (df_marca["facings"] / total_facings * 100).round(1)
    df_marca["sos_ancho"]   = (df_marca["ancho"]   / total_ancho   * 100).round(1)
    df_marca = df_marca.sort_values("ancho", ascending=False).reset_index(drop=True)

    sos_nuestras_ancho   = df_marca[df_marca["es_nuestra"]]["sos_ancho"].sum()
    sos_nuestras_facings = df_marca[df_marca["es_nuestra"]]["sos_facings"].sum()
    n_marcas  = df_marca["marca"].nunique()
    n_promos  = df["promocion"].notna().sum()
    precios_v = df["precio"].dropna()

    # Rango de precios en lugar de media
    if precios_v.empty:
        precio_str  = "—"
        precio_label = "Precios"
    elif len(precios_v) == 1:
        precio_str  = f"{precios_v.iloc[0]:.2f}€"
        precio_label = "Precio"
    else:
        precio_str  = f"{precios_v.min():.2f}–{precios_v.max():.2f}€"
        precio_label = "Rango precios"

    # ── KPIs ──────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(f'<div class="metric-card"><div class="metric-value">{sos_nuestras_ancho:.1f}%</div><div class="metric-label">SoS por ancho</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="metric-card"><div class="metric-value">{sos_nuestras_facings:.1f}%</div><div class="metric-label">SoS por facings</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="metric-card"><div class="metric-value">{n_marcas}</div><div class="metric-label">Marcas</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="metric-card"><div class="metric-value">{n_promos}</div><div class="metric-label">En promoción</div></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.3rem">{precio_str}</div><div class="metric-label">{precio_label}</div></div>', unsafe_allow_html=True)

    if notas:
        st.info(f"📝 {notas}")
    if origen:
        st.caption(origen)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Por marca", "Por tecnología", "Por pet", "Por segmento",
        "Precios", "Promociones", "Detalle"
    ])

    colores_marca = ["#E8472C" if v else "#BDBDBD" for v in df_marca["es_nuestra"]]

    with tab1:
        modo_sos = st.radio("Calcular SoS por:", ["Ancho (recomendado)", "Facings"], horizontal=True, key="sos_modo")
        col_sos = "sos_ancho" if "Ancho" in modo_sos else "sos_facings"
        st.plotly_chart(bar_chart(
            df_marca["marca"], df_marca[col_sos], colores_marca,
            df_marca["facings"], height=340,
        ), use_container_width=True)

    with tab2:
        df_tech = df.groupby("tecnologia", as_index=False).agg(facings=("facings","sum"), ancho=("ancho_relativo","sum"))
        df_tech["sos"] = (df_tech["ancho"] / total_ancho * 100).round(1)
        df_tech = df_tech.sort_values("ancho", ascending=False)
        st.plotly_chart(bar_chart(df_tech["tecnologia"], df_tech["sos"],
                                  [COLORES_TECH.get(t, "#BDBDBD") for t in df_tech["tecnologia"]],
                                  df_tech["facings"]), use_container_width=True)
        st.markdown("**Desglose marca × tecnología**")
        pivot = df.pivot_table(index="marca", columns="tecnologia", values="ancho_relativo", aggfunc="sum", fill_value=0)
        st.plotly_chart(stacked_chart(pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index],
                                      COLORES_TECH, ylabel="Ancho relativo"), use_container_width=True)

    with tab3:
        df_pet = df.groupby("pet", as_index=False).agg(facings=("facings","sum"), ancho=("ancho_relativo","sum"))
        df_pet["sos"] = (df_pet["ancho"] / total_ancho * 100).round(1)
        df_pet = df_pet.sort_values("ancho", ascending=False)
        st.plotly_chart(bar_chart(df_pet["pet"], df_pet["sos"],
                                  [COLORES_PET.get(p, "#BDBDBD") for p in df_pet["pet"]],
                                  df_pet["facings"]), use_container_width=True)
        st.markdown("**Desglose marca × pet**")
        pivot_pet = df.pivot_table(index="marca", columns="pet", values="ancho_relativo", aggfunc="sum", fill_value=0)
        st.plotly_chart(stacked_chart(pivot_pet.loc[pivot_pet.sum(axis=1).sort_values(ascending=False).index],
                                      COLORES_PET, ylabel="Ancho relativo"), use_container_width=True)

    with tab4:
        df_seg = df[df["segmento"] != "N/A"].groupby("segmento", as_index=False).agg(
            facings=("facings","sum"), ancho=("ancho_relativo","sum"))
        df_seg["sos"] = (df_seg["ancho"] / total_ancho * 100).round(1)
        df_seg = df_seg.sort_values("ancho", ascending=False)
        if not df_seg.empty:
            st.plotly_chart(bar_chart(df_seg["segmento"], df_seg["sos"],
                                      [COLORES_SEG.get(s, "#BDBDBD") for s in df_seg["segmento"]],
                                      df_seg["facings"]), use_container_width=True)
            st.markdown("**Desglose marca × segmento**")
            df_nosna = df[df["segmento"] != "N/A"]
            pivot_seg = df_nosna.pivot_table(index="marca", columns="segmento", values="ancho_relativo", aggfunc="sum", fill_value=0)
            st.plotly_chart(stacked_chart(pivot_seg.loc[pivot_seg.sum(axis=1).sort_values(ascending=False).index],
                                          COLORES_SEG, ylabel="Ancho relativo"), use_container_width=True)
        else:
            st.info("No se detectaron segmentos.")

    with tab5:
        df_precio = df[df["precio"].notna()].copy()
        if df_precio.empty:
            st.info("No se detectaron precios en esta imagen.")
        else:
            # Precio por producto individual — sin promediar
            df_precio["producto_label"] = df_precio.apply(
                lambda r: f"{r['marca']}" + (f" · {r['formato']}" if pd.notna(r.get("formato")) and r.get("formato") else ""),
                axis=1,
            )
            df_ps = df_precio.sort_values("precio").reset_index(drop=True)
            colores_p = ["#E8472C" if v else "#BDBDBD" for v in df_ps["es_nuestra"]]
            fig_p = go.Figure(go.Bar(
                x=df_ps["producto_label"],
                y=df_ps["precio"],
                marker_color=colores_p,
                text=[f"{v:.2f}€" for v in df_ps["precio"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Precio: %{y:.2f}€<extra></extra>",
            ))
            fig_p.update_layout(
                title="Precio por producto (€/unidad)",
                plot_bgcolor="white", showlegend=False,
                yaxis=dict(gridcolor="#f0f0f0", range=[0, df_ps["precio"].max() * 1.3], title="€"),
                xaxis=dict(tickangle=-30),
                margin=dict(t=40, b=80), height=380,
            )
            st.plotly_chart(fig_p, use_container_width=True)

            st.markdown("**Tabla de precios**")
            df_ptab = df_precio[["marca","pet","tecnologia","formato","precio","promocion"]].copy()
            df_ptab.columns = ["Marca","Pet","Tecnología","Formato","Precio (€)","Promoción"]
            st.dataframe(df_ptab.sort_values("Precio (€)"), hide_index=True, use_container_width=True)

    with tab6:
        df_promo = df[df["promocion"].notna()].copy()
        if df_promo.empty:
            st.success("✅ No se detectaron productos en promoción.")
        else:
            st.warning(f"⚠️ {len(df_promo)} producto(s) en promoción detectados")
            df_ptab = df_promo[["marca","pet","tecnologia","segmento","formato","precio","promocion"]].copy()
            df_ptab.columns = ["Marca","Pet","Tecnología","Segmento","Formato","Precio (€)","Promoción"]
            st.dataframe(df_ptab, hide_index=True, use_container_width=True)

            ancho_promo    = df_promo["ancho_relativo"].sum()
            ancho_no_promo = total_ancho - ancho_promo
            fig_pie = go.Figure(go.Pie(
                labels=["En promoción", "Sin promoción"],
                values=[ancho_promo, ancho_no_promo],
                marker_colors=["#FF9800", "#BDBDBD"],
                hole=0.4,
            ))
            fig_pie.update_layout(title="SoS en promoción vs sin promoción", height=300, margin=dict(t=40,b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab7:
        df_show = df[["marca","pet","tecnologia","segmento","formato","facings","sos_ancho","sos_facings","precio","promocion"]].copy()
        df_show.columns = ["Marca","Pet","Tecnología","Segmento","Formato","Facings","SoS Ancho (%)","SoS Facings (%)","Precio (€)","Promoción"]
        st.dataframe(df_show.sort_values("SoS Ancho (%)", ascending=False), hide_index=True, use_container_width=True)
        csv = df_show.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", data=csv, file_name="share_of_shelf.csv",
                           mime="text/csv", use_container_width=True)

# ── Layout principal ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Share of Shelf Analyzer", page_icon="🛒", layout="wide")

st.markdown("""
<style>
    .metric-card { background:#f8f9fa; border-radius:12px; padding:16px;
                   text-align:center; border:1px solid #e9ecef; }
    .metric-value { font-size:1.8rem; font-weight:700; color:#E8472C; }
    .metric-label { font-size:0.8rem; color:#6c757d; margin-top:4px; }
</style>""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3514/3514447.png", width=60)
    st.title("Share of Shelf")
    st.divider()

    if not st.secrets.get("ANTHROPIC_API_KEY", ""):
        st.markdown("### 🔑 API Key de Anthropic")
        user_key = st.text_input(
            "Pega tu API Key aquí",
            type="password",
            placeholder="sk-ant-api03-...",
            help="Obtén tu key gratuita en console.anthropic.com",
            label_visibility="collapsed",
        )
        if user_key:
            st.session_state["_api_key"] = user_key
            st.success("✅ Key guardada para esta sesión")
        else:
            st.warning("Introduce tu API Key para analizar imágenes")
        st.markdown("[Obtener API Key →](https://console.anthropic.com)", unsafe_allow_html=False)
        st.divider()

    st.caption("Powered by Claude claude-opus-4-5 Vision · Affinity Petcare")

st.title("🛒 Share of Shelf Analyzer")
st.divider()

# ── Upload múltiple (fotos + vídeos combinados) ────────────────────────────────
st.subheader("📁 Sube fotos y/o vídeos")
st.caption("Puedes combinar varias fotos y vídeos de distintas baldas — se analizarán juntos.")

uploaded_files = st.file_uploader(
    "Selecciona archivos",
    type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    fotos  = [f for f in uploaded_files if not es_video(f)]
    videos = [f for f in uploaded_files if es_video(f)]

    if fotos:
        st.markdown(f"**📷 {len(fotos)} foto(s)**")
        cols = st.columns(min(len(fotos), 4))
        for i, f in enumerate(fotos):
            with cols[i % 4]:
                st.image(f, use_container_width=True, caption=f.name)

    if videos:
        st.markdown(f"**🎬 {len(videos)} vídeo(s)**")
        for v in videos:
            st.video(v)
            st.caption(v.name)
        intervalo = st.slider("Analizar un frame de vídeo cada… (segundos)", 1, 10, 3)
    else:
        intervalo = 3


    st.divider()
    analizar = st.button("🔍 Analizar todo", type="primary", use_container_width=True)
else:
    st.info("👆 Sube una o más fotos y/o vídeos para comenzar")
    analizar  = False
    intervalo = 3

# ── Análisis ───────────────────────────────────────────────────────────────────
if uploaded_files and analizar:

    # 1. Recopilar todas las imágenes (fotos + frames de vídeo)
    imagenes = []   # lista de {"bytes": ..., "media_type": ...}
    tmp_paths = []

    progress = st.progress(0, text="Preparando imágenes…")
    n_archivos = len(uploaded_files)

    for i_archivo, archivo in enumerate(uploaded_files):
        progress.progress(i_archivo / n_archivos, text=f"Cargando {archivo.name}…")

        if not es_video(archivo):
            imagenes.append({"bytes": archivo.getvalue(), "media_type": archivo.type or "image/jpeg"})
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(archivo.getvalue())
                tmp_paths.append(tmp.name)
            frames = extraer_frames(tmp_paths[-1], intervalo)
            for fb in frames:
                imagenes.append({"bytes": fb, "media_type": "image/jpeg"})

    progress.progress(1.0, text=f"Enviando {len(imagenes)} imagen(es) a Claude…")

    # 2. Una sola llamada con todas las imágenes
    try:
        data = analizar_imagenes(imagenes)
        todos_productos = data.get("productos", [])
        notas = data.get("notas", "")
        u = data.get("_usage", {})
    except Exception as e:
        st.error(f"❌ Error en el análisis: {e}")
        todos_productos = []
        notas = ""
        u = {}
    finally:
        for p in tmp_paths:
            try: os.unlink(p)
            except: pass

    progress.empty()

    if not todos_productos:
        st.error("No se detectaron productos.")
    else:
        df_raw = pd.DataFrame(todos_productos)
        for col in ["formato", "precio", "promocion", "ancho_relativo"]:
            if col not in df_raw.columns:
                df_raw[col] = None
        df_raw["ancho_relativo"] = pd.to_numeric(df_raw["ancho_relativo"], errors="coerce").fillna(1)
        df_raw["precio"]         = pd.to_numeric(df_raw["precio"],         errors="coerce")

        # Deduplicar por solape: misma combinación → quedarse con el MAX de facings/ancho
        df_agg = df_raw.groupby(
            ["marca", "pet", "tecnologia", "segmento", "formato"],
            as_index=False, dropna=False,
        ).agg(
            facings=("facings", "max"),
            ancho_relativo=("ancho_relativo", "max"),
            precio=("precio", "first"),
            promocion=("promocion", "first"),
        )

        origen = f"{len(uploaded_files)} archivo(s): {', '.join(f.name for f in uploaded_files)}"
        n_imgs = len(imagenes)

        st.success(f"✅ {n_imgs} imagen(es) analizadas en una sola llamada · {len(todos_productos)} productos detectados")
        st.divider()

        mostrar_resultados(df_agg, notas, origen=origen)
        mostrar_tokens(u.get("input_tokens", 0), u.get("output_tokens", 0), n_frames=n_imgs)

        with st.expander("🔍 Ver respuesta bruta del análisis (debug)"):
            st.code(st.session_state.get("debug_raw", ""), language="json")
