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
    return normalizar(marca) in {normalizar(m) for m in NUESTRAS_MARCAS}

# ── Prompt ─────────────────────────────────────────────────────────────────────
PROMPT = """Analiza esta imagen de baldas de una tienda de alimentación para mascotas.

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

# ── Función análisis imagen ────────────────────────────────────────────────────
def analizar_imagen(image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    client  = anthropic.Anthropic(api_key=api_key)
    img_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
            {"type": "text",  "text": PROMPT},
        ]}],
    )

    raw = msg.content[0].text.strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            raw = m.group(1).strip()

    # Debug: guardar respuesta en session_state para inspección
    st.session_state["debug_raw"] = raw

    return json.loads(raw)

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

    # ── SoS por facings
    total_facings = df["facings"].sum()
    df["sos_facings"] = (df["facings"] / total_facings * 100).round(1) if total_facings > 0 else 0

    # ── SoS por ancho relativo
    total_ancho = df["ancho_relativo"].sum()
    df["sos_ancho"] = (df["ancho_relativo"] / total_ancho * 100).round(1) if total_ancho > 0 else 0

    # Agrupado por marca para KPIs
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
    precio_medio = precios_v.mean() if not precios_v.empty else None

    # ── KPIs ──────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(f'<div class="metric-card"><div class="metric-value">{sos_nuestras_ancho:.1f}%</div><div class="metric-label">SoS por ancho</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="metric-card"><div class="metric-value">{sos_nuestras_facings:.1f}%</div><div class="metric-label">SoS por facings</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="metric-card"><div class="metric-value">{n_marcas}</div><div class="metric-label">Marcas</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="metric-card"><div class="metric-value">{n_promos}</div><div class="metric-label">En promoción</div></div>', unsafe_allow_html=True)
    precio_str = f"{precio_medio:.2f}€" if precio_medio else "—"
    k5.markdown(f'<div class="metric-card"><div class="metric-value">{precio_str}</div><div class="metric-label">Precio medio</div></div>', unsafe_allow_html=True)

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
        col_sos2 = col_sos  # sigue el mismo modo
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
            # Precio medio por marca
            df_pm = df_precio.groupby(["marca","es_nuestra"], as_index=False)["precio"].mean().round(2)
            df_pm = df_pm.sort_values("precio")
            colores_pm = ["#E8472C" if v else "#BDBDBD" for v in df_pm["es_nuestra"]]
            fig_p = go.Figure(go.Bar(
                x=df_pm["marca"], y=df_pm["precio"],
                marker_color=colores_pm,
                text=[f"{v:.2f}€" for v in df_pm["precio"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Precio medio: %{y:.2f}€<extra></extra>",
            ))
            fig_p.update_layout(
                title="Precio medio por marca (€/unidad)",
                plot_bgcolor="white", showlegend=False,
                yaxis=dict(gridcolor="#f0f0f0", range=[0, df_pm["precio"].max() * 1.25], title="€"),
                margin=dict(t=40, b=10), height=340,
            )
            st.plotly_chart(fig_p, use_container_width=True)

            # Tabla precio por producto
            st.markdown("**Precios por producto**")
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

            # SoS de productos en promo vs no promo
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

st.title("🛒 Share of Shelf Analyzer")
st.divider()

modo = st.radio("Modo de análisis", ["📷 Foto", "🎬 Vídeo"], horizontal=True)
st.divider()

# ════════════════════════════════════════════════════════════════
# MODO FOTO
# ════════════════════════════════════════════════════════════════
if modo == "📷 Foto":
    col_upload, col_results = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("📷 Imagen")
        uploaded = st.file_uploader("Selecciona una foto", type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed")
        if uploaded:
            st.image(uploaded, use_container_width=True)
            analizar = st.button("🔍 Analizar", type="primary", use_container_width=True)
        else:
            st.info("👆 Sube una foto para comenzar")
            analizar = False

    with col_results:
        st.subheader("📊 Resultados")
        if uploaded and analizar:
            with st.spinner("Analizando con IA…"):
                try:
                    data = analizar_imagen(uploaded.getvalue(), uploaded.type)
                    df   = pd.DataFrame(data["productos"])
                    # Asegurar columnas opcionales
                    for col in ["formato","precio","promocion","ancho_relativo"]:
                        if col not in df.columns:
                            df[col] = None
                    df["ancho_relativo"] = pd.to_numeric(df["ancho_relativo"], errors="coerce").fillna(1)
                    df["precio"]         = pd.to_numeric(df["precio"],         errors="coerce")
                    mostrar_resultados(df, data.get("notas", ""))

                    # Panel debug
                    with st.expander("🔍 Ver respuesta bruta de Claude (debug)"):
                        st.code(st.session_state.get("debug_raw", ""), language="json")

                except json.JSONDecodeError as e:
                    st.error("❌ Error al interpretar la respuesta de la IA.")
                    st.code(str(e))
                    with st.expander("🔍 Respuesta bruta"):
                        st.code(st.session_state.get("debug_raw", ""))
                except Exception as e:
                    st.error(f"❌ Error inesperado: {e}")
        elif not uploaded:
            st.markdown('<div style="text-align:center;padding:60px 20px;color:#adb5bd"><div style="font-size:3rem">📈</div><div>Los resultados aparecerán aquí</div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# MODO VÍDEO
# ════════════════════════════════════════════════════════════════
else:
    col_upload, col_results = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("🎬 Vídeo")
        video_file = st.file_uploader("Selecciona un vídeo", type=["mp4","mov","avi","mkv"],
                                      label_visibility="collapsed")
        if video_file:
            st.video(video_file)
            intervalo = st.slider("Analizar un frame cada… (segundos)", 1, 10, 3)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_file.getvalue())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            fps_vid    = cap.get(cv2.CAP_PROP_FPS) or 25
            n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duracion_s = n_frames / fps_vid
            cap.release()

            n_analizar = max(1, int(duracion_s / intervalo))
            st.info(f"⏱ Duración: **{duracion_s:.0f}s** · Se analizarán **{n_analizar} frames** · Tiempo estimado: ~{n_analizar * 8}s")
            analizar_video = st.button("🎬 Analizar vídeo", type="primary", use_container_width=True)
        else:
            st.info("👆 Sube un vídeo para comenzar")
            analizar_video = False
            tmp_path = None

    with col_results:
        st.subheader("📊 Resultados")

        if video_file and analizar_video and tmp_path:
            try:
                frames = extraer_frames(tmp_path, intervalo)
                todos_productos = []
                notas_todas     = []

                progress = st.progress(0, text="Analizando frames…")
                for i, frame_bytes in enumerate(frames):
                    progress.progress((i + 1) / len(frames), text=f"Frame {i+1} de {len(frames)}…")
                    try:
                        data = analizar_imagen(frame_bytes, "image/jpeg")
                        todos_productos.extend(data["productos"])
                        if data.get("notas"):
                            notas_todas.append(data["notas"])
                    except Exception:
                        pass

                progress.empty()
                os.unlink(tmp_path)

                if not todos_productos:
                    st.error("No se pudieron analizar los frames.")
                else:
                    df_total = pd.DataFrame(todos_productos)
                    for col in ["formato","precio","promocion","ancho_relativo"]:
                        if col not in df_total.columns:
                            df_total[col] = None
                    df_total["ancho_relativo"] = pd.to_numeric(df_total["ancho_relativo"], errors="coerce").fillna(1)
                    df_total["precio"]         = pd.to_numeric(df_total["precio"],         errors="coerce")

                    df_agg = df_total.groupby(["marca","pet","tecnologia","segmento","formato"], as_index=False, dropna=False).agg(
                        facings=("facings","sum"),
                        ancho_relativo=("ancho_relativo","sum"),
                        precio=("precio","mean"),
                        promocion=("promocion","first"),
                    )
                    notas = " · ".join(set(notas_todas)) if notas_todas else ""
                    st.success(f"✅ {len(frames)} frames analizados")
                    mostrar_resultados(df_agg, notas,
                                       origen=f"Resultado acumulado de {len(frames)} frames (intervalo: {intervalo}s)")
                    with st.expander("🔍 Ver respuesta bruta del último frame (debug)"):
                        st.code(st.session_state.get("debug_raw", ""), language="json")

            except Exception as e:
                st.error(f"❌ Error inesperado: {e}")

        elif not video_file:
            st.markdown('<div style="text-align:center;padding:60px 20px;color:#adb5bd"><div style="font-size:3rem">🎬</div><div>Los resultados aparecerán aquí</div></div>', unsafe_allow_html=True)
