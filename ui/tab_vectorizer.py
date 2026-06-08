# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from PIL import Image
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import streamlit as st
import traceback

from modules.utils.visualization import generar_cubo_3d
from modules.vision.image_processing import (
    KERNEL_MORPH,
    convertir_a_escala_grises, aplicar_desenfoque_gaussiano,
    detectar_bordes_canny, aplicar_clahe, aplicar_filtro_bilateral,
    aplicar_desenfoque_mediana, aplicar_nlmeans, detectar_umbrales_automaticos,
    binarizar_adaptativo, segmentar_grabcut,
    detectar_bordes_multi_escala
)
from modules.vision.contours import (
    get_contours, remuestrear_contorno, suavizar_contorno_media_movil
)
# Arquitectura POO integrada para el procesamiento analítico
from modules.geometry.extrusion import Extruder, ExtrusionConfig
from modules.vision.svg_parser import SVGParser


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
    # Título limpio y formal para el entregable del Capstone
    st.title("Procesamiento de Geometría y Extrusión 3D")
    st.markdown("Carga una imagen JPG/PNG o un archivo vectorial SVG")
    st.markdown("---")

    # Carga de archivo
    uploaded_file = st.file_uploader(
        "Carga una imagen (JPG/PNG/SVG)",
        type=["jpg", "jpeg", "png", "svg"],
        key="vector_uploader"
    )

    if uploaded_file is not None:
        is_svg = uploaded_file.name.lower().endswith('.svg')

        # --- PARÁMETROS COMUNES COMUNICADOS POR COLUMNAS DENTRO DE LA UI ---
        st.subheader("Parámetros de Extrusión 3D")
        col1, col2 = st.columns(2)
        with col1:
            extrusion_height = st.slider("Altura de extrusión", 0.1, 3.0, 1.0, 0.1)
            mesh_opacity = st.slider("Opacidad", 0.1, 1.0, 0.8, key="opacity_svg")
        with col2:
            color_mode = st.selectbox("Esquema de color", ["Monocromático", "Paleta de colores"], index=1)
            if color_mode == "Monocromático":
                mesh_color = st.color_picker("Color de la malla", "#1E90FF")
            else:
                mesh_color = "#1E90FF"

        st.markdown("---")
        suavizar_caras_solicitado = st.toggle(
            "Suavizado de Cara (Subdivisión)",
            value=False,
            help="Habilita la subdivisión geométrica por niveles en Z requerida en el entregable."
        )

        colors_list = ["#1E90FF", "#FF4500", "#32CD32", "#FFD700", "#FF1493", "#8A2BE2", "#00FFFF"]

        if is_svg:
            st.markdown("---")
            mostrar_contorno_exterior = st.toggle(
                "Mostrar Contorno Exterior",
                value=False,
                help="Agrega un segundo modelo 3D usando solo el path más grande del SVG."
            )

            st.markdown("---")
            bezier_res = st.slider("Resolución curvas Bézier", 5, 100, 30, help="Número de puntos por segmento de curva Bézier")

            # --- SVG Compatibility Guide ---
            with st.expander("Guía de compatibilidad SVG", expanded=False):
                st.markdown("""
| Tipo de SVG | Compatibilidad | Resultado 3D |
|---|---|---|
| Silueta / contorno único | Excelente | Sólido fiel a la forma |
| Logo simple (1-3 paths) | Muy bueno | Capas bien definidas |
| Ícono con capas decorativas | Usar toggle | Activa *Mostrar Contorno Exterior* |
""")

            try:
                parser = SVGParser(bezier_resolution=bezier_res)
                elements = parser.parse_file(uploaded_file)

                if not elements:
                    st.warning("El archivo SVG no contiene elementos geométricos válidos compatibles.")
                else:
                    all_pts = [elem['points'] for elem in elements if len(elem['points']) > 0]

                    if all_pts:
                        all_pts_concat = np.concatenate(all_pts, axis=0)
                        dims = all_pts_concat.max(axis=0) - all_pts_concat.min(axis=0)
                        scale_factor = 2.0 / (max(dims[0], dims[1]) if max(dims[0], dims[1]) != 0 else 1.0)
                    else:
                        scale_factor = 1.0

                    def compute_polygon_area(pts):
                        n = len(pts)
                        if n < 3: return 0.0
                        return 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1)))

                    def subsample_path(pts, max_points=500):
                        if len(pts) <= max_points: return pts
                        indices = np.round(np.linspace(0, len(pts) - 1, max_points)).astype(int)
                        return pts[indices]

                    st.subheader("Visualización del Vector 2D (Wireframe)")
                    fig2d = go.Figure()

                    for i, elem in enumerate(elements):
                        pts = elem['points']
                        if len(pts) > 0:
                            pts_plot = np.vstack([pts, pts[0]]) if elem['closed'] else pts
                            fig2d.add_trace(go.Scatter(
                                x=pts_plot[:, 0].tolist(), y=pts_plot[:, 1].tolist(),
                                mode='lines+markers', line=dict(width=2, color=colors_list[i % len(colors_list)]),
                                marker=dict(size=4), name=f"{elem['type']} #{i+1}"
                            ))

                    fig2d.update_layout(yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1), height=400)
                    st.plotly_chart(fig2d, use_container_width=True)

                    st.markdown("---")
                    st.subheader("Modelo 3D Extrusionado")

                    fig3d = go.Figure()
                    total_vertices, total_faces = 0, 0
                    mallas_creadas = []

                    for i, elem in enumerate(elements):
                        pts = elem['points']
                        if len(pts) >= (3 if elem['closed'] else 2):
                            scaled_pts = pts * scale_factor
                            scaled_pts[:, 1] = -scaled_pts[:, 1]
                            scaled_pts = subsample_path(scaled_pts)

                            # Uso correcto del motor POO Extruder
                            config_elem = ExtrusionConfig(
                                height=extrusion_height,
                                opacity=mesh_opacity,
                                color=mesh_color if color_mode == "Monocromático" else colors_list[i % len(colors_list)],
                                closed=elem['closed'],
                                suavizado_de_cara=suavizar_caras_solicitado,
                                subdivisions=2 if suavizar_caras_solicitado else 1
                            )
                            extruder = Extruder(config=config_elem)
                            vertices, faces = extruder.process(scaled_pts)

                            if len(vertices) > 0 and len(faces) > 0:
                                total_vertices += len(vertices)
                                total_faces += len(faces)
                                mallas_creadas.append((extruder, vertices, faces))

                                mesh_data = extruder.create_plotly_mesh(vertices, faces)
                                if mesh_data:
                                    fig3d.add_trace(go.Mesh3d(**mesh_data))

                    if total_vertices > 0:
                        fig3d.update_layout(scene=dict(aspectmode='data'), height=500, margin=dict(l=0, r=0, b=0, t=0))
                        st.plotly_chart(fig3d, use_container_width=True)
                        st.caption(f"Malla Compuesta: {total_vertices} vértices, {total_faces} caras")

                        # MÉTRICAS INTEGRADAS EN EL PIPELINE SVG
                        st.markdown("#### Datos Físicos del Modelo Compuesto")
                        col_vol, col_area = st.columns(2)
                        vol_total = sum(ext.calculate_volume(v, f) for ext, v, f in mallas_creadas)
                        area_total = sum(ext.calculate_surface_area(v, f) for ext, v, f in mallas_creadas)
                        col_vol.metric("Volumen Total Estimado", f"{vol_total:.4f} u³")
                        col_area.metric("Área Superficial Total", f"{area_total:.4f} u²")
                    else:
                        st.warning("No se pudo extruir ninguno de los trazados del SVG.")

                    if mostrar_contorno_exterior:
                        st.markdown("---")
                        st.subheader("Contorno Exterior — Path más grande")
                        closed_elems = [e for e in elements if e['closed'] and len(e['points']) >= 3]
                        source_list = closed_elems if closed_elems else elements
                        largest_elem = max(source_list, key=lambda e: compute_polygon_area(e['points']))
                        st.caption(f"Path seleccionado: {len(largest_elem['points'])} puntos")

                        pts_ext = largest_elem['points'].copy() * scale_factor
                        pts_ext[:, 1] = -pts_ext[:, 1]
                        pts_ext = subsample_path(pts_ext)

                        config_ext = ExtrusionConfig(
                            height=extrusion_height,
                            color=mesh_color if color_mode == "Monocromático" else "#1E90FF",
                            opacity=mesh_opacity,
                            closed=largest_elem['closed'],
                            suavizado_de_cara=suavizar_caras_solicitado,
                            subdivisions=2 if suavizar_caras_solicitado else 1
                        )
                        extruder_ext = Extruder(config=config_ext)
                        verts_ext, faces_ext = extruder_ext.process(pts_ext)

                        if len(verts_ext) > 0 and len(faces_ext) > 0:
                            mesh_ext = extruder_ext.create_plotly_mesh(verts_ext, faces_ext)
                            if mesh_ext:
                                fig_ext = go.Figure(data=[go.Mesh3d(**mesh_ext)])
                                fig_ext.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), height=500)
                                st.plotly_chart(fig_ext, use_container_width=True)
                        else:
                            st.warning("No se pudo extruir el contorno exterior.")

            except Exception as e:
                st.error(f"Error procesando SVG: {e}")
                st.code(traceback.format_exc())

        else:
            # Classic PNG/JPG Pipeline
            image = Image.open(uploaded_file)
            image_array = np.array(image.convert('RGB'))

            st.subheader("Parámetros de Detección Canny")
            col1, col2 = st.columns(2)
            with col1:
                kernel_size = st.slider("Tamaño del kernel", 1, 31, 5, 2)
                threshold1 = st.slider("Umbral Bajo", 0, 200, 50)
            with col2:
                threshold2 = st.slider("Umbral Alto", threshold1 + 10, 500, 150)
                extrusion_height = st.slider("Altura extrusión", 0.1, 3.0, 1.0, 0.1)

            mesh_color = st.color_picker("Color de la malla", "#1E90FF")
            mesh_opacity = st.slider("Opacidad", 0.1, 1.0, 0.8, key="opacity_img")

            st.markdown("---")

            with st.expander("Configuración avanzada (opcional)"):
                modo_deteccion = st.selectbox(
                    "Modo de detección",
                    ["Canny Estándar", "Umbral Adaptativo", "Multi-Escala", "Segmentación Automática"],
                    index=0
                )

                block_size_adaptive, c_adaptive, grabcut_iterations = 11, 2, 5
                if modo_deteccion == "Umbral Adaptativo":
                    block_size_adaptive = st.slider("Tamaño de bloque adaptativo", 3, 51, 11, 2)
                    c_adaptive = st.slider("Constante C", -10, 20, 2)
                elif modo_deteccion == "Segmentación Automática":
                    st.info("GrabCut separa automáticamente el objeto del fondo")
                    grabcut_iterations = st.slider("Iteraciones GrabCut", 1, 10, 5)

                mejora_preprocesado = st.checkbox("Preprocesado avanzado (CLAHE)", True)
                metodo_denoise = st.selectbox("Método de reducción de ruido",
                                              ["Bilateral (recomendado)", "Gaussiano", "Mediana", "NL-Means (lento)"],
                                              index=0)

                suavizar_contorno = st.checkbox("Suavizar contorno", True)
                ventana_suavizado = st.slider("Ventana de suavizado", 3, 15, 5, 2) if suavizar_contorno else 5

                remuestrear = st.checkbox("Remuestrear contorno", True)
                num_puntos_malla = st.slider("Puntos de la malla", 50, 500, 150, 10) if remuestrear else 0

                modo_contorno = st.selectbox("Modo de contorno", ["Solo el más grande", "Todos los contornos (unidos)", "Todos los contornos (hull)"], index=0)
                morph_kernel_size = st.slider("Tamaño kernel morfológico", 3, 21, 5, 2)
                min_area_contorno = st.slider("Área mínima del contorno", 10, 5000, 100, 50)
                simplificacion_contorno = st.select_slider("Simplificación del contorno", options=["Ninguna", "Baja", "Media", "Alta"], value="Baja")

                invertir_bordes = st.checkbox("Invertir bordes", False)
                trazar_bordes = st.checkbox("Trazar bordes (mejor para dibujos)", True)

                if trazar_bordes:
                    cierre_iteraciones = st.slider("Iteraciones de cierre", 1, 20, 8)
                    radio_ajuste = st.slider("Radio de ajuste a bordes", 1, 30, 15)
                else:
                    cierre_iteraciones = 5
                    radio_ajuste = 10

                usar_hull = st.checkbox("Usar Convex Hull", False)
                metodo_ordenamiento = st.selectbox("Orden de puntos", ["Original (recomendado)", "Angular (solo convexos)", "Optimizado"], index=0)

            denoise_map = {"Bilateral (recomendado)": "bilateral", "Gaussiano": "gaussian", "Mediana": "median"}
            denoise_method = denoise_map.get(metodo_denoise, "bilateral")

            try:
                original = reducir_resolucion(image_array)
                gray = convertir_a_escala_grises(original)
                enhanced = aplicar_clahe(gray, clip_limit=2.5) if mejora_preprocesado else gray

                if denoise_method == 'gaussian': denoised = aplicar_desenfoque_gaussiano(enhanced, kernel_size)
                elif denoise_method == 'bilateral': denoised = aplicar_filtro_bilateral(enhanced)
                elif denoise_method == 'median': denoised = aplicar_desenfoque_mediana(enhanced, kernel_size)
                else: denoised = enhanced

                if modo_deteccion == "Canny Estándar":
                    edges = detectar_bordes_canny(denoised, threshold1, threshold2)
                elif modo_deteccion == "Umbral Adaptativo":
                    binary = binarizar_adaptativo(denoised, block_size_adaptive, c_adaptive)
                    if np.mean(binary) > 127: binary = 255 - binary
                    edges = cv2.Canny(binary, 50, 150)
                elif modo_deteccion == "Multi-Escala":
                    edges = detectar_bordes_multi_escala(denoised, scales=[1.0, 0.5, 0.25])
                else:
                    mask = segmentar_grabcut(original, iterations=grabcut_iterations)
                    edges = cv2.Canny(mask, 50, 150)

                if invertir_bordes: edges = 255 - edges

                st.subheader("Resultados del Procesamiento")
                col_imgs = st.columns(4)
                col_imgs[0].image(original, caption="Original", use_container_width=True)
                col_imgs[1].image(gray, caption="Grises", use_container_width=True)
                col_imgs[2].image(denoised, caption="Preprocesado", use_container_width=True)
                col_imgs[3].image(edges, caption=f"Bordes ({modo_deteccion})", use_container_width=True)

                st.markdown("---")
                col_contour, col_3d = st.columns([1, 2])

                with col_contour:
                    st.markdown("**Contorno Detectado**")
                    kernel_morph = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
                    simplify_map = {"Ninguna": ("none", 0.0), "Baja": ("fixed", 0.0005), "Media": ("adaptive", 0.001), "Alta": ("fixed", 0.005)}
                    simplify_method, epsilon_factor = simplify_map.get(simplificacion_contorno, ("adaptive", 0.001))

                    multi_contour_mode = "union" if modo_contorno == "Todos los contornos (unidos)" else "single"
                    contour_config = {
                        'simplify_method': simplify_method, 'epsilon_factor': epsilon_factor, 'min_area': min_area_contorno,
                        'morph_op': 'close', 'multi_contour_mode': multi_contour_mode, 'trace_edges': trazar_bordes,
                        'close_iterations': cierre_iteraciones, 'close_kernel': 3, 'adjust_to_edges': trazar_bordes, 'search_radius': radio_ajuste
                    }
                    contour_points = get_contours(edges, kernel_morph, contour_config)

                    if len(contour_points) > 0:
                        if usar_hull:
                            contour_points = cv2.convexHull(contour_points.reshape(-1, 1, 2).astype(np.int32)).reshape(-1, 2).astype(np.float32)

                        processed_contour = contour_points.copy()
                        if suavizar_contorno and len(processed_contour) > ventana_suavizado:
                            processed_contour = suavizar_contorno_media_movil(processed_contour, window=ventana_suavizado)
                        if remuestrear and num_puntos_malla > 0:
                            processed_contour = remuestrear_contorno(processed_contour, num_puntos_malla)

                        contour_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                        cv2.drawContours(contour_display, [contour_points.reshape(-1, 1, 2).astype(np.int32)], -1, (255, 100, 100), 1)
                        cv2.drawContours(contour_display, [processed_contour.reshape(-1, 1, 2).astype(np.int32)], -1, (0, 255, 0), 2)
                        st.image(contour_display, use_container_width=True)

                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("Puntos originales", len(contour_points))
                        col_m2.metric("Puntos malla", len(processed_contour))
                    else:
                        processed_contour = np.array([])
                        st.warning("No se detectaron contornos.")

                with col_3d:
                    st.markdown("**Modelo 3D Extrusionado**")
                    if len(processed_contour) >= 3:
                        sort_method_map = {"Original (recomendado)": "original", "Angular (solo convexos)": "angular", "Optimizado": "optimized"}

                        config_img = ExtrusionConfig(
                            height=extrusion_height,
                            color=mesh_color,
                            opacity=mesh_opacity,
                            sort_method=sort_method_map.get(metodo_ordenamiento, "original"),
                            suavizado_de_cara=suavizar_caras_solicitado,
                            subdivisions=3 if suavizar_caras_solicitado else 1
                        )
                        extruder = Extruder(config=config_img)
                        norm = extruder.normalize_points(processed_contour)
                        vertices, faces = extruder.process(norm)
                        mesh_data = extruder.create_plotly_mesh(vertices, faces)

                        if mesh_data:
                            fig = go.Figure(data=[go.Mesh3d(**mesh_data)])
                            fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), height=450)
                            st.plotly_chart(fig, use_container_width=True)

                            # MÉTRICAS INTEGRADAS EN EL PIPELINE DE IMÁGENES
                            st.markdown("##### Métricas Geométricas del Sólido")
                            c1, c2 = st.columns(2)
                            c1.metric("Volumen Estimado", f"{extruder.calculate_volume(vertices, faces):.4f} u³")
                            c2.metric("Área Superficial Total", f"{extruder.calculate_surface_area(vertices, faces):.4f} u²")
                        else:
                            st.warning("No se pudo crear la malla 3D")
                    else:
                        st.info("Cargando cubo de demostración...")
                        st.plotly_chart(generar_cubo_3d(), use_container_width=True)

            except Exception as e:
                st.error(f"Error en procesamiento de imagen: {e}")
    else:
        st.info("Carga una imagen para comenzar")