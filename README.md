# 🛒 Share of Shelf Analyzer

App para analizar fotos (y vídeos) de baldas de tienda y calcular el share of shelf por marca, tecnología, pet y segmento.

---

## ⚙️ Instalación (solo la primera vez)

### 1. Instalar Python
Si no lo tienes, descárgalo de [python.org](https://python.org) e instálalo.

### 2. Instalar dependencias
Abre una terminal (CMD o PowerShell) y ejecuta:
```
pip install streamlit anthropic plotly pillow opencv-python
```

### 3. Obtener tu API Key de Anthropic
1. Entra en [console.anthropic.com](https://console.anthropic.com)
2. Ve a **API Keys** → **Create API Key**
3. Copia la key (solo se muestra una vez)

### 4. Configurar tu API Key
Dentro de la carpeta `share_of_shelf`, crea la carpeta `.streamlit` y dentro el archivo `secrets.toml` con este contenido:
```
ANTHROPIC_API_KEY = "sk-ant-api03-TU_KEY_AQUI"
```

---

## 🚀 Cómo ejecutar la app

Haz doble clic en **`launch.bat`** (se abre el navegador automáticamente).

O desde la terminal:
```
streamlit run app.py
```

Luego abre el navegador en: **http://localhost:8501**

---

## 📋 Cómo usar la app

### Análisis de foto
1. Haz clic en **"Browse files"** y selecciona una foto de la balda (JPG, PNG, WEBP)
2. Pulsa el botón **"🔍 Analizar"**
3. Espera unos segundos — la IA analiza la imagen
4. Explora los resultados en las 5 pestañas:
   - **Por marca** — share of shelf por marca (en rojo las nuestras)
   - **Por tecnología** — Dry / Wet / Snacks
   - **Por pet** — Perro / Gato
   - **Por segmento** — Maxi-Medium / Mini / Esterilizado / No Esterilizado
   - **Detalle** — tabla completa + descarga CSV

### Análisis de vídeo
1. Selecciona un vídeo (MP4, MOV, AVI)
2. Ajusta cada cuántos segundos quieres analizar un frame
3. Pulsa **"🎬 Analizar vídeo"**
4. La app extrae frames automáticamente y los analiza uno a uno
5. Los resultados son el promedio de todos los frames analizados

---

## 🏷️ Nuestras marcas (se resaltan en rojo)
Advance · Ultima · Nature's Variety · Natural Trainer · Brekkies · Affinity

---

## ❓ Problemas frecuentes

| Problema | Solución |
|---|---|
| "Could not resolve authentication" | Falta la API key en `secrets.toml` |
| La app no abre | Asegúrate de que el servidor está corriendo (`launch.bat`) |
| Error al analizar | La imagen puede ser demasiado pequeña o borrosa |
