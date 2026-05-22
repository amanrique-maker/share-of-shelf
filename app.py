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

# ── Prompt compartido ──────────────────────────────────────────────────────────
PROMPT = """Analiza esta imagen de baldas de una tienda de alimentación para mascotas.

Para cada grupo de productos visibles identifica:
1. MARCA (ej: Advance, Purina, Royal Canin…). Si no se distingue → "Otros".
2. PET: "Perro", "Gato" o "Otros".
3. TECNOLOGÍA: "Dry" (pienso seco), "Wet" (comida húmeda), "Snacks" (premios/sticks).
4. SEGMENTO:
   - Perro → "Maxi-Medium" o "Mini"
   - Gato  → "Esterilizado" o "No Esterilizado"
   - Snacks / Otros → "N/A"
5. FACINGS: unidades visibles de frente (incluye parcialmente tapadas).

Devuelve ÚNICAMENTE un JSON válido, sin texto adicional:
{
  "productos": [
    {"marca": "Advance", "pet": "Perro", "tecnologia": "Dry", "segmento": "Maxi-Medium", "facings": 8}
  ],
  "total_facings": 8,
  "notas": ""
}"""

# ── Función análisis imagen ────────────────────────────────────────────────────
def analizar_imagen(image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    client  = anthropic.Anthropic(api_key=api_key)
    img_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
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

# ── Función mostrar resultados (reutilizable) ──────────────────────────────────
def mostrar_resultados(df: pd.DataFrame, total: int, notas: str, origen: str = ""):
    df["es_nuestra"] = df["marca"].apply(es_nuestra)
    df["share"]      = (df["facings"] / total * 100).round(1)

    df_marca = df.groupby(["marca", "es_nuestra"], as_index=False)["facings"].sum()
    df_marca["share"] = (df_marca["facings"] / total * 100).round(1)
    df_marca = df_marca.sort_values("facings", ascending=False).reset_index(drop=True)

    sos_nuestras = df_marca[df_marca["es_nuestra"]]["share"].sum()
    n_marcas     = df_marca["marca"].nunique()

    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="metric-card"><div class="metric-value">{total}</div><div class="metric-label">Facings totales</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="metric-card"><div class="metric-value">{sos_nuestras:.1f}%</div><div class="metric-label">Nuestro SoS</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="metric-card"><div class="metric-value">{n_marcas}</div><div class="metric-label">Marcas detectadas</div></div>', unsafe_allow_html=True)

    if notas:
        st.info(f"📝 {notas}")
    if origen:
        st.caption(origen)

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Por marca", "Por tecnología", "Por pet", "Por segmento", "Detalle"])

    def bar_chart(x, y, colores, customdata, height=320):
        fig = go.Figure(go.Bar(
            x=x, y=y, marker_color=colores,
            text=[f"{v:.1f}%" for v in y], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Facings: %{customdata}<br>Share: %{y:.1f}%<extra></extra>",
            customdata=customdata,
        ))
        fig.update_layout(
            plot_bgcolor="white", showlegend=False,
            yaxis=dict(gridcolor="#f0f0f0", range=[0, max(y) * 1.25], title="Share (%)"),
            margin=dict(t=10, b=10), height=height,
        )
        return fig

    def stacked_chart(pivot, colores_dict, height=280):
        fig = go.Figure()
        for col in pivot.columns:
            fig.add_trace(go.Bar(name=col, x=pivot.index, y=pivot[col],
                                 marker_color=colores_dict.get(col, "#BDBDBD")))
        fig.update_layout(barmode="stack", plot_bgcolor="white",
                          yaxis=dict(gridcolor="#f0f0f0", title="Facings"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          margin=dict(t=30, b=10), height=height)
        return fig

    with tab1:
        st.plotly_chart(bar_chart(
            df_marca["marca"], df_marca["share"],
            ["#E8472C" if v else "#BDBDBD" for v in df_marca["es_nuestra"]],
            df_marca["facings"], height=340,
        ), use_container_width=True)

    with tab2:
        df_tech = df.groupby("tecnologia", as_index=False)["facings"].sum()
        df_tech["share"] = (df_tech["facings"] / total * 100).round(1)
        df_tech = df_tech.sort_values("facings", ascending=False)
        st.plotly_chart(bar_chart(df_tech["tecnologia"], df_tech["share"],
                                  [COLORES_TECH.get(t, "#BDBDBD") for t in df_tech["tecnologia"]],
                                  df_tech["facings"]), use_container_width=True)
        st.markdown("**Desglose marca × tecnología**")
        pivot = df.pivot_table(index="marca", columns="tecnologia", values="facings", aggfunc="sum", fill_value=0)
        st.plotly_chart(stacked_chart(pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index], COLORES_TECH), use_container_width=True)

    with tab3:
        df_pet = df.groupby("pet", as_index=False)["facings"].sum()
        df_pet["share"] = (df_pet["facings"] / total * 100).round(1)
        df_pet = df_pet.sort_values("facings", ascending=False)
        st.plotly_chart(bar_chart(df_pet["pet"], df_pet["share"],
                                  [COLORES_PET.get(p, "#BDBDBD") for p in df_pet["pet"]],
                                  df_pet["facings"]), use_container_width=True)
        st.markdown("**Desglose marca × pet**")
        pivot_pet = df.pivot_table(index="marca", columns="pet", values="facings", aggfunc="sum", fill_value=0)
        st.plotly_chart(stacked_chart(pivot_pet.loc[pivot_pet.sum(axis=1).sort_values(ascending=False).index], COLORES_PET), use_container_width=True)

    with tab4:
        df_seg = df[df["segmento"] != "N/A"].groupby("segmento", as_index=False)["facings"].sum()
        df_seg["share"] = (df_seg["facings"] / total * 100).round(1)
        df_seg = df_seg.sort_values("facings", ascending=False)
        if not df_seg.empty:
            st.plotly_chart(bar_chart(df_seg["segmento"], df_seg["share"],
                                      [COLORES_SEG.get(s, "#BDBDBD") for s in df_seg["segmento"]],
                                      df_seg["facings"]), use_container_width=True)
            st.markdown("**Desglose marca × segmento**")
            df_nosna = df[df["segmento"] != "N/A"]
            pivot_seg = df_nosna.pivot_table(index="marca", columns="segmento", values="facings", aggfunc="sum", fill_value=0)
            st.plotly_chart(stacked_chart(pivot_seg.loc[pivot_seg.sum(axis=1).sort_values(ascending=False).index], COLORES_SEG), use_container_width=True)
        else:
            st.info("No se detectaron segmentos en esta imagen.")

    with tab5:
        df_show = df[["marca", "pet", "tecnologia", "segmento", "facings", "share"]].copy()
        df_show.columns = ["Marca", "Pet", "Tecnología", "Segmento", "Facings", "Share (%)"]
        st.dataframe(df_show.sort_values("Facings", ascending=False), hide_index=True, use_container_width=True)
        csv = df_show.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", data=csv, file_name="share_of_shelf.csv",
                           mime="text/csv", use_container_width=True)

# ── Layout principal ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Share of Shelf Analyzer", page_icon="🛒", layout="wide")

st.markdown("""
<style>
    .metric-card { background:#f8f9fa; border-radius:12px; padding:20px;
                   text-align:center; border:1px solid #e9ecef; }
    .metric-value { font-size:2.2rem; font-weight:700; color:#E8472C; }
    .metric-label { font-size:0.9rem; color:#6c757d; margin-top:4px; }
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
                    mostrar_resultados(df, data["total_facings"], data.get("notas", ""))
                except json.JSONDecodeError as e:
                    st.error("❌ Error al interpretar la respuesta de la IA.")
                    st.code(str(e))
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

            # Guardar temporalmente para calcular duración
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

                progress = st.progress(0, text="Extrayendo y analizando frames…")

                for i, frame_bytes in enumerate(frames):
                    progress.progress((i + 1) / len(frames),
                                      text=f"Analizando frame {i+1} de {len(frames)}…")
                    try:
                        data = analizar_imagen(frame_bytes, "image/jpeg")
                        todos_productos.extend(data["productos"])
                        if data.get("notas"):
                            notas_todas.append(data["notas"])
                    except Exception:
                        pass  # Si un frame falla, continuamos

                progress.empty()
                os.unlink(tmp_path)

                if not todos_productos:
                    st.error("No se pudieron analizar los frames. Prueba con otro vídeo.")
                else:
                    # Agregar: suma de facings de todos los frames
                    df_total = pd.DataFrame(todos_productos)
                    df_agg   = df_total.groupby(["marca","pet","tecnologia","segmento"], as_index=False)["facings"].sum()
                    total    = int(df_agg["facings"].sum())
                    notas    = " · ".join(set(notas_todas)) if notas_todas else ""

                    st.success(f"✅ {len(frames)} frames analizados")
                    mostrar_resultados(df_agg, total, notas,
                                       origen=f"Resultado acumulado de {len(frames)} frames (intervalo: {intervalo}s)")

            except Exception as e:
                st.error(f"❌ Error inesperado: {e}")

        elif not video_file:
            st.markdown('<div style="text-align:center;padding:60px 20px;color:#adb5bd"><div style="font-size:3rem">🎬</div><div>Los resultados aparecerán aquí</div></div>', unsafe_allow_html=True)
