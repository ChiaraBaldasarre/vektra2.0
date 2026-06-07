# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from PIL import Image
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import streamlit as st

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
# Importamos la nueva arquitectura POO integrada
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

        # --- PANEL LATERAL COMÚN DE EXTRUSIÓN 3D ---
        st.sidebar.subheader("Parametros de Extrusion 3D")
        extrusion_height = st.sidebar.slider("Altura de extrusión", 0.1, 3.0, 1.0, 0.1)
        mesh_opacity = st.sidebar.slider("Opacidad", 0.1, 1.0, 0.8)
        color_mode = st.sidebar.selectbox("Esquema de color", ["Monocromático", "Paleta de colores"], index=1)
        mesh_color = st.sidebar.color_picker("Color de la malla", "#1E90FF") if color_mode == "Monocromático" else "#1E90FF"

        st.sidebar.markdown("---")
        st.sidebar.subheader("Modo Estructural")
        suavizar_caras_solicitado = st.sidebar.toggle(
            "Suavizado de Cara (Subdivisión)",
            value=False,
            help="Habilita la subdivisión geométrica por niveles en Z requerida en el ticket #4."
        )

        colors_list = ["#1E90FF", "#FF4500", "#32CD32", "#FFD700", "#FF1493", "#8A2BE2", "#00FFFF"]

        if is_svg:
            # SVG Pipeline UI Sidebar
            mostrar_contorno_exterior = st.sidebar.toggle(
                "Mostrar Contorno Exterior",
                value=False,
                help="Agrega un segundo modelo 3D usando solo el path más grande del SVG."
            )

            st.sidebar.markdown("---")
            st.sidebar.subheader("Parametros de Curvas SVG")
            bezier_res = st.sidebar.slider("Resolución curvas Bézier", 5, 100, 30, help="Puntos por segmento Bézier")

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
                        min_coords = all_pts_concat.min(axis=0)
                        max_coords = all_pts_concat.max(axis=0)
                        dims = max_coords - min_coords
                        max_dim = max(dims[0], dims[1])
                        scale_factor = 2.0 / (max_dim if max_dim != 0 else 1.0)
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

                    # 2D Preview Wireframe
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

                    # ---- 3D Extrusion Pipeline ----
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

                            # Construcción de la Configuración del motor POO para el elemento
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

                        # RENDERIZADO DE NUESTRAS MÉTRICAS AVANZADAS
                        st.markdown("#### Datos Físicos del Modelo Compuesto")
                        col_vol, col_area = st.columns(2)
                        vol_total = sum(ext.calculate_volume(v, f) for ext, v, f in mallas_creadas)
                        area_total = sum(ext.calculate_surface_area(v, f) for ext, v, f in mallas_creadas)
                        col_vol.metric("Volumen Total Estimado", f"{vol_total:.4f} u³")
                        col_area.metric("Área Superficial Total", f"{area_total:.4f} u²")
                    else:
                        st.warning("No se pudo extruir ninguno de los trazados del SVG.")

            except Exception as e:
                import traceback
                st.error(f"Error procesando SVG: {e}")
                st.code(traceback.format_exc())

        else:
            # Classic PNG/JPG Pipeline
            image = Image.open(uploaded_file)
            image_array = np.array(image.convert('RGB'))

            st.sidebar.subheader("Parametros de Deteccion Canny")
            kernel_size = st.sidebar.slider("Kernel (Desenfoque)", 1, 31, 5, step=2)
            threshold1 = st.sidebar.slider("Umbral Bajo (Canny)", 0, 200, 50)
            threshold2 = st.sidebar.slider("Umbral Alto (Canny)", threshold1 + 10, 500, 150)

            st.sidebar.markdown("---")
            modo_deteccion = st.sidebar.selectbox("Modo de detección", ["Canny Estándar", "Umbral Adaptativo", "Multi-Escala", "Segmentación Automática"])

            block_size_adaptive, c_adaptive, grabcut_iterations = 11, 2, 5
            if modo_deteccion == "Umbral Adaptativo":
                block_size_adaptive = st.sidebar.slider("Tamaño bloque adaptativo", 3, 51, 11, 2)
                c_adaptive = st.sidebar.slider("Constante C", -10, 20, 2)
            elif modo_deteccion == "Segmentación Automática":
                grabcut_iterations = st.sidebar.slider("Iteraciones GrabCut", 1, 10, 5)

            mejora_preprocesado = st.sidebar.checkbox("Preprocesado avanzado (CLAHE)", True)
            metodo_denoise = st.sidebar.selectbox("Reducción de ruido", ["Bilateral (recomendado)", "Gaussiano", "Mediana"])
            suavizar_contorno = st.sidebar.checkbox("Suavizar contorno", True)
            ventana_suavizado = st.sidebar.slider("Ventana de suavizado", 3, 15, 5, 2) if suavizar_contorno else 5
            remuestrear = st.sidebar.checkbox("Remuestrear contorno", True)
            num_puntos_malla = st.sidebar.slider("Puntos de la malla", 50, 500, 150, 10) if remuestrear else 0

            st.sidebar.markdown("---")
            modo_contorno = st.sidebar.selectbox("Modo de contorno", ["Solo el más grande", "Todos los contornos (unidos)"], index=0)
            morph_kernel_size = st.sidebar.slider("Kernel morfológico", 3, 21, 5, 2)
            min_area_contorno = st.sidebar.slider("Área mínima contorno", 10, 5000, 100, 50)
            simplificacion_contorno = st.sidebar.select_slider("Simplificación", options=["Ninguna", "Baja", "Media", "Alta"], value="Baja")
            invertir_bordes = st.sidebar.checkbox("Invertir bordes", False)
            trazar_bordes = st.sidebar.checkbox("Trazar bordes", True)
            cierre_iteraciones = st.sidebar.slider("Iteraciones de cierre", 1, 20, 8) if trazar_bordes else 5
            radio_ajuste = st.sidebar.slider("Radio de ajuste", 1, 30, 15) if trazar_bordes else 10
            usar_hull = st.sidebar.checkbox("Usar Convex Hull", False)
            metodo_ordenamiento = st.sidebar.selectbox("Orden de puntos", ["Original (recomendado)", "Angular (solo convexos)", "Optimizado"])

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
                    edges = mejorar_bordes_morfologicos(edges, kernel_size=3)
                elif modo_deteccion == "Umbral Adaptativo":
                    binary = binarizar_adaptativo(denoised, block_size_adaptive, c_adaptive)
                    if np.mean(binary) > 127: binary = 255 - binary
                    edges = cv2.Canny(binary, 50, 150)
                    edges = mejorar_bordes_morfologicos(edges, kernel_size=3)
                elif modo_deteccion == "Multi-Escala":
                    edges = detectar_bordes_multi_escala(denoised, scales=[1.0, 0.5, 0.25])
                    edges = mejorar_bordes_morfologicos(edges, kernel_size=3)
                else:
                    mask = segmentar_grabcut(original, iterations=grabcut_iterations)
                    edges = cv2.Canny(mask, 50, 150)
                    edges = mejorar_bordes_morfologicos(edges, kernel_size=5)

                if invertir_bordes: edges = 255 - edges

                st.subheader("Resultados del Procesamiento")
                col_imgs = st.columns(4)
                col_imgs[0].image(original, caption="Original", width="stretch")
                col_imgs[1].image(gray, caption="Grises", width="stretch")
                col_imgs[2].image(denoised, caption="Preprocesado", width="stretch")
                col_imgs[3].image(edges, caption=f"Bordes", width="stretch")

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
                        st.image(contour_display, width="stretch")
                    else:
                        processed_contour = np.array([])
                        st.warning("No se detectaron contornos.")

                with col_3d:
                    st.markdown("**Modelo 3D Extrusionado**")
                    if len(processed_contour) >= 3:
                        sort_method_map = {"Original (recomendado)": "original", "Angular (solo convexos)": "angular", "Optimizado": "optimized"}

                        # Pipeline utilizando la nueva clase Extruder
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
                            fig.update_layout(scene=dict(aspectmode='data'), height=450, margin=dict(l=0, r=0, b=0, t=0))
                            st.plotly_chart(fig, use_container_width=True)

                            # Métricas en pantalla para imágenes
                            st.markdown("##### Métricas Geométricas del Sólido")
                            c1, c2 = st.columns(2)
                            c1.metric("Volumen", f"{extruder.calculate_volume(vertices, faces):.4f} u³")
                            c2.metric("Área Superficial", f"{extruder.calculate_surface_area(vertices, faces):.4f} u²")
                        else:
                            st.warning("No se pudo crear la malla 3D")
                    else:
                        st.info("Cargando cubo de demostración...")
                        st.plotly_chart(generar_cubo_3d(), use_container_width=True)

            except Exception as e:
                st.error(f"Error en procesamiento de imagen: {e}")
    else:
        st.info("👆 Carga una imagen para comenzar")