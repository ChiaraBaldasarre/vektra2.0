import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from modules.utils.visualization import generar_cubo_3d
from modules.vision.image_processing import (
    convertir_a_escala_grises, aplicar_desenfoque_gaussiano,
    detectar_bordes_canny, aplicar_clahe, aplicar_filtro_bilateral,
    aplicar_desenfoque_mediana, aplicar_nlmeans, detectar_umbrales_automaticos,
    mejorar_bordes_morfologicos, binarizar_adaptativo, segmentar_grabcut,
    detectar_bordes_multi_escala
)
from modules.vision.contours import (
    get_contours, remuestrear_contorno, suavizar_contorno_media_movil
)
from modules.geometry.extrusion import (
    normalize_points, extrude_polygon, create_plotly_mesh,
    sort_contour_points, ensure_closed_polygon
)


@st.cache_data
def reducir_resolucion(image_array, max_dim=800):
    """Reduce resolución para procesamiento rápido"""
    height, width = image_array.shape[:2]
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        return cv2.resize(image_array, (new_width, new_height))
    return image_array


def render_vectorizer():
    st.markdown("Carga una imagen JPG/PNG")
    st.markdown("---")

    # Carga de archivo
    uploaded_file = st.file_uploader(
        "Carga una imagen (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="vector_uploader"
    )

    if uploaded_file is not None:
        # Leer imagen
        image = Image.open(uploaded_file)
        image_array = np.array(image.convert('RGB'))

        # Sidebar - Parametros de procesamiento
        st.sidebar.subheader("⚙️ Parámetros de Detección Canny")

        kernel_size = st.sidebar.slider(
            "Tamaño del kernel (Desenfoque Gaussiano)",
            min_value=1, max_value=31, value=5, step=2,
            help="Aumentar para más suavizado"
        )

        threshold1 = st.sidebar.slider(
            "Umbral Bajo (Canny)",
            min_value=0, max_value=200, value=50,
            help="Umbral inferior para detección de bordes"
        )

        threshold2 = st.sidebar.slider(
            "Umbral Alto (Canny)",
            min_value=threshold1 + 10, max_value=500, value=150,
            help="Umbral superior para detección de bordes"
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("🎛️ Parámetros de Extrusión 3D")

        extrusion_height = st.sidebar.slider("Altura de extrusión", 0.1, 3.0, 1.0, 0.1)
        mesh_color = st.sidebar.color_picker("Color de la malla", "#1E90FF")
        mesh_opacity = st.sidebar.slider("Opacidad", 0.1, 1.0, 0.8)

        st.sidebar.markdown("---")
        st.sidebar.subheader("💡 Mejora de Detección")

        modo_deteccion = st.sidebar.selectbox(
            "Modo de detección",
            ["Canny Estándar", "Umbral Adaptativo", "Multi-Escala", "Segmentación Automática"],
            index=0
        )

        block_size_adaptive = 11
        c_adaptive = 2
        grabcut_iterations = 5

        if modo_deteccion == "Umbral Adaptativo":
            block_size_adaptive = st.sidebar.slider("Tamaño de bloque adaptativo", 3, 51, 11, 2)
            c_adaptive = st.sidebar.slider("Constante C", -10, 20, 2)

        if modo_deteccion == "Segmentación Automática":
            st.sidebar.info("💡 GrabCut separa automáticamente el objeto del fondo")
            grabcut_iterations = st.sidebar.slider("Iteraciones GrabCut", 1, 10, 5)

        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Calidad de Malla")

        mejora_preprocesado = st.sidebar.checkbox("Preprocesado avanzado (CLAHE)", True)
        metodo_denoise = st.sidebar.selectbox("Método de reducción de ruido",
                                              ["Bilateral (recomendado)", "Gaussiano", "Mediana", "NL-Means (lento)"],
                                              index=0)

        suavizar_contorno = st.sidebar.checkbox("Suavizar contorno", True)
        ventana_suavizado = st.sidebar.slider("Ventana de suavizado", 3, 15, 5, 2) if suavizar_contorno else 5

        remuestrear = st.sidebar.checkbox("Remuestrear contorno", True)
        num_puntos_malla = st.sidebar.slider("Puntos de la malla", 50, 500, 150, 10) if remuestrear else 0

        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Ajuste de Contorno")

        modo_contorno = st.sidebar.selectbox("Modo de contorno", ["Solo el más grande", "Todos los contornos (unidos)",
                                                                  "Todos los contornos (hull)"], index=0)
        morph_kernel_size = st.sidebar.slider("Tamaño kernel morfológico", 3, 21, 5, 2)
        min_area_contorno = st.sidebar.slider("Área mínima del contorno", 10, 5000, 100, 50)
        simplificacion_contorno = st.sidebar.select_slider("Simplificación del contorno",
                                                           options=["Ninguna", "Baja", "Media", "Alta"], value="Baja")

        invertir_bordes = st.sidebar.checkbox("Invertir bordes", False)
        trazar_bordes = st.sidebar.checkbox("Trazar bordes (mejor para dibujos)", True)

        if trazar_bordes:
            cierre_iteraciones = st.sidebar.slider("Iteraciones de cierre", 1, 20, 8)
            radio_ajuste = st.sidebar.slider("Radio de ajuste a bordes", 1, 30, 15)
        else:
            cierre_iteraciones = 5
            radio_ajuste = 10

        usar_hull = st.sidebar.checkbox("Usar Convex Hull", False)
        metodo_ordenamiento = st.sidebar.selectbox("Orden de puntos",
                                                   ["Original (recomendado)", "Angular (solo convexos)", "Optimizado"],
                                                   index=0)

        denoise_map = {
            "Bilateral (recomendado)": "bilateral", "Gaussiano": "gaussian",
            "Mediana": "median", "NL-Means (lento)": "nlmeans"
        }
        denoise_method = denoise_map.get(metodo_denoise, "bilateral")

        try:
            original = reducir_resolucion(image_array)
            gray = convertir_a_escala_grises(original)

            if mejora_preprocesado:
                enhanced = aplicar_clahe(gray, clip_limit=2.5)
            else:
                enhanced = gray

            if denoise_method == 'gaussian':
                denoised = aplicar_desenfoque_gaussiano(enhanced, kernel_size)
            elif denoise_method == 'bilateral':
                denoised = aplicar_filtro_bilateral(enhanced)
            elif denoise_method == 'median':
                denoised = aplicar_desenfoque_mediana(enhanced, kernel_size)
            elif denoise_method == 'nlmeans':
                denoised = aplicar_nlmeans(enhanced)
            else:
                denoised = enhanced

            blurred = denoised

            if modo_deteccion == "Canny Estándar":
                edges = detectar_bordes_canny(denoised, threshold1, threshold2)
                edges = mejorar_bordes_morfologicos(edges, kernel_size=3)

            elif modo_deteccion == "Umbral Adaptativo":
                binary = binarizar_adaptativo(denoised, block_size_adaptive, c_adaptive)
                if np.mean(binary) > 127: binary = 255 - binary
                edges = cv2.Canny(binary, 50, 150)
                edges = mejorar_bordes_morfologicos(edges, kernel_size=3)

            elif modo_deteccion == "Multi-Escala":
                edges = detectar_bordes_multi_escala(denoised, scales=[1.0, 0.5, 0.25])
                edges = mejorar_bordes_morfologicos(edges, kernel_size=3)

            elif modo_deteccion == "Segmentación Automática":
                mask = segmentar_grabcut(original, iterations=grabcut_iterations)
                edges = cv2.Canny(mask, 50, 150)
                edges = mejorar_bordes_morfologicos(edges, kernel_size=5)

            if invertir_bordes:
                edges = 255 - edges

            st.subheader("Resultados del Procesamiento")

            col_imgs = st.columns(4)
            col_imgs[0].image(original, caption="Original", width="stretch")
            col_imgs[1].image(gray, caption="Grises", width="stretch")
            col_imgs[2].image(blurred, caption="Preprocesado", width="stretch")
            col_imgs[3].image(edges, caption=f"Bordes ({modo_deteccion})", width="stretch")

            st.markdown("---")
            col_contour, col_3d = st.columns([1, 2])

            with col_contour:
                st.markdown("**🔍 Contorno Detectado**")

                kernel_morph = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
                simplify_map = {
                    "Ninguna": ("none", 0.0), "Baja": ("fixed", 0.0005),
                    "Media": ("adaptive", 0.001), "Alta": ("fixed", 0.005)
                }
                simplify_method, epsilon_factor = simplify_map.get(simplificacion_contorno, ("adaptive", 0.001))

                multi_contour_mode = "single"
                if modo_contorno == "Todos los contornos (unidos)":
                    multi_contour_mode = "union"
                elif modo_contorno == "Todos los contornos (hull)":
                    multi_contour_mode = "hull"

                contour_config = {
                    'simplify_method': simplify_method, 'epsilon_factor': epsilon_factor,
                    'min_area': min_area_contorno, 'morph_op': 'close',
                    'multi_contour_mode': multi_contour_mode, 'trace_edges': trazar_bordes,
                    'close_iterations': cierre_iteraciones, 'close_kernel': 3,
                    'adjust_to_edges': trazar_bordes, 'search_radius': radio_ajuste
                }
                contour_points = get_contours(edges, kernel_morph, contour_config)

                if len(contour_points) > 0:
                    if usar_hull:
                        hull = cv2.convexHull(contour_points.reshape(-1, 1, 2).astype(np.int32))
                        contour_points = hull.reshape(-1, 2).astype(np.float32)

                    processed_contour = contour_points.copy()

                    if suavizar_contorno and len(processed_contour) > ventana_suavizado:
                        processed_contour = suavizar_contorno_media_movil(processed_contour, window=ventana_suavizado)

                    if remuestrear and num_puntos_malla > 0:
                        processed_contour = remuestrear_contorno(processed_contour, num_puntos_malla)

                    contour_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(contour_display, [contour_points.reshape(-1, 1, 2).astype(np.int32)], -1,
                                     (255, 100, 100), 1)
                    cv2.drawContours(contour_display, [processed_contour.reshape(-1, 1, 2).astype(np.int32)], -1,
                                     (0, 255, 0), 2)

                    st.image(contour_display, width="stretch")

                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("Puntos originales", len(contour_points))
                    col_m2.metric("Puntos malla", len(processed_contour))
                else:
                    processed_contour = np.array([])
                    st.warning("No se detectaron contornos. Ajusta los umbrales.")

            with col_3d:
                st.markdown("**🔷 Modelo 3D Extrusionado**")

                if len(processed_contour) >= 3:
                    sort_method_map = {
                        "Original (recomendado)": "original",
                        "Angular (solo convexos)": "angular",
                        "Optimizado": "optimized"
                    }
                    sort_method = sort_method_map.get(metodo_ordenamiento, "original")

                    ordered = sort_contour_points(processed_contour, method=sort_method)
                    ordered = ensure_closed_polygon(ordered)
                    norm = normalize_points(ordered, 2.0)

                    vertices, faces = extrude_polygon(norm, height=extrusion_height, triangulate=True)
                    mesh_data = create_plotly_mesh(vertices, faces, color=mesh_color, opacity=mesh_opacity)

                    if mesh_data:
                        fig = go.Figure(data=[go.Mesh3d(**mesh_data)])
                        fig.update_layout(
                            scene=dict(aspectmode='data', xaxis=dict(showgrid=True), yaxis=dict(showgrid=True),
                                       zaxis=dict(showgrid=True)),
                            margin=dict(l=0, r=0, b=0, t=0), height=450
                        )
                        st.plotly_chart(fig, width="stretch")
                        st.caption(f"📊 Malla: {len(vertices)} vértices, {len(faces)} caras")
                    else:
                        st.warning("No se pudo crear la malla 3D")
                else:
                    st.info("Cargando cubo de demostración...")
                    st.plotly_chart(generar_cubo_3d(), width="stretch")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("👆 Carga una imagen para comenzar")