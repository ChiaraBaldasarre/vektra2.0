
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
from modules.geometry.extrusion import (
    normalize_points, extrude_polygon, create_plotly_mesh,
    sort_contour_points, ensure_closed_polygon
)
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
        # Check if file is SVG
        is_svg = uploaded_file.name.lower().endswith('.svg')

        if is_svg:
            # SVG Pipeline UI Sidebar
            st.sidebar.subheader("Parametros de Extrusion 3D")
            extrusion_height = st.sidebar.slider("Altura de extrusión", 0.1, 3.0, 1.0, 0.1)
            mesh_opacity = st.sidebar.slider("Opacidad", 0.1, 1.0, 0.8)

            color_mode = st.sidebar.selectbox("Esquema de color", ["Monocromático", "Paleta de colores"], index=1)

            if color_mode == "Monocromático":
                mesh_color = st.sidebar.color_picker("Color de la malla", "#1E90FF")
            else:
                mesh_color = "#1E90FF"  # Default

            st.sidebar.markdown("---")
            st.sidebar.subheader("Modo de Extrusión")
            mostrar_contorno_exterior = st.sidebar.toggle(
                "Mostrar Contorno Exterior",
                value=False,
                help="Agrega un segundo modelo 3D usando solo el path más grande del SVG. "
                     "Útil para íconos con capas decorativas."
            )

            st.sidebar.markdown("---")
            st.sidebar.subheader("Parametros de Curvas SVG")
            bezier_res = st.sidebar.slider("Resolución curvas Bézier", 5, 100, 30, help="Número de puntos por segmento de curva Bézier")

            # --- SVG Compatibility Guide (minimal) ---
            with st.expander("Guía de compatibilidad SVG", expanded=False):
                st.markdown("""
| Tipo de SVG | Compatibilidad | Resultado 3D |
|---|---|---|
| Silueta / contorno único | Excelente | Sólido fiel a la forma |
| Logo simple (1-3 paths) | Muy bueno | Capas bien definidas |
| Ícono con capas decorativas | Usar toggle | Activa *Mostrar Contorno Exterior* |
| Ilustración técnica (plano) | Bueno | Múltiples sólidos por sección |
| Fotografía vectorizada | Parcial | Muchos paths, resultado complejo |
| SVG con texto o referencias internas | No soportado | Elementos ignorados |
""")

            try:
                # Parse SVG
                parser = SVGParser(bezier_resolution=bezier_res)
                elements = parser.parse_file(uploaded_file)

                if not elements:
                    st.warning("El archivo SVG no contiene elementos geométricos válidos compatibles (rect, circle, polygon, path).")
                else:
                    # Find bounds of all elements to auto-scale proportionately
                    all_pts = []
                    for elem in elements:
                        if len(elem['points']) > 0:
                            all_pts.append(elem['points'])

                    if all_pts:
                        all_pts_concat = np.concatenate(all_pts, axis=0)
                        min_coords = all_pts_concat.min(axis=0)
                        max_coords = all_pts_concat.max(axis=0)
                        dims = max_coords - min_coords
                        max_dim = max(dims[0], dims[1])
                        if max_dim == 0:
                            max_dim = 1.0
                        scale_factor = 2.0 / max_dim
                    else:
                        scale_factor = 1.0

                    # --- Helper: absolute polygon area via Shoelace ---
                    def compute_polygon_area(pts):
                        """Absolute area of a 2D polygon via Shoelace formula."""
                        n = len(pts)
                        if n < 3:
                            return 0.0
                        x = pts[:, 0]
                        y = pts[:, 1]
                        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

                    # --- Robust extrusion with fallback to lateral-only ---
                    def extrude_robust(pts_2d, height, closed):
                        """
                        Tries full triangulated extrusion first.
                        Falls back to lateral-faces-only (no caps) when Delaunay
                        fails on complex or self-intersecting polygons.
                        """
                        verts, faces = extrude_polygon(pts_2d, height=height, triangulate=True, closed=closed)
                        if len(verts) > 0 and len(faces) > 0:
                            return verts, faces
                        return extrude_polygon(pts_2d, height=height, triangulate=False, closed=closed)

                    # --- Uniform subsampling for dense paths ---
                    def subsample_path(pts, max_points=500):
                        """Uniformly subsample a path to at most max_points points.
                        Preserves shape fidelity while keeping extrusion fast."""
                        n = len(pts)
                        if n <= max_points:
                            return pts
                        indices = np.round(np.linspace(0, n - 1, max_points)).astype(int)
                        return pts[indices]


                    # 2D Preview Wireframe
                    st.subheader("Visualización del Vector 2D (Wireframe)")
                    fig2d = go.Figure()
                    colors_list = ["#1E90FF", "#FF4500", "#32CD32", "#FFD700", "#FF1493", "#8A2BE2", "#00FFFF"]
                    
                    for i, elem in enumerate(elements):
                        pts = elem['points']
                        if len(pts) > 0:
                            if elem['closed']:
                                pts_plot = np.vstack([pts, pts[0]])
                            else:
                                pts_plot = pts
                            
                            color_idx = i % len(colors_list)
                            fig2d.add_trace(go.Scatter(
                                x=pts_plot[:, 0].tolist(),
                                y=pts_plot[:, 1].tolist(),
                                mode='lines+markers',
                                line=dict(width=2, color=colors_list[color_idx]),
                                marker=dict(size=4, color=colors_list[color_idx]),
                                name=f"{elem['type']} #{i+1} ({'cerrado' if elem['closed'] else 'abierto'})"
                            ))
                            
                    fig2d.update_layout(
                        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
                        xaxis=dict(constrain="domain"),
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig2d, use_container_width=True)
                    
                    # ---- 3D Extrusion: todos los paths ----
                    st.markdown("---")
                    st.subheader("Modelo 3D Extrusionado")

                    fig3d = go.Figure()
                    total_vertices = 0
                    total_faces = 0

                    for i, elem in enumerate(elements):
                        pts = elem['points']
                        if len(pts) >= (3 if elem['closed'] else 2):
                            scaled_pts = pts * scale_factor
                            scaled_pts[:, 1] = -scaled_pts[:, 1]
                            scaled_pts = subsample_path(scaled_pts)

                            vertices, faces = extrude_robust(scaled_pts, height=extrusion_height, closed=elem['closed'])

                            if len(vertices) > 0 and len(faces) > 0:
                                total_vertices += len(vertices)
                                total_faces += len(faces)
                                color = mesh_color if color_mode == "Monocromático" else colors_list[i % len(colors_list)]
                                mesh_data = create_plotly_mesh(vertices, faces, color=color, opacity=mesh_opacity)
                                if mesh_data:
                                    fig3d.add_trace(go.Mesh3d(**mesh_data))

                    if total_vertices > 0:
                        fig3d.update_layout(
                            scene=dict(aspectmode='data', xaxis=dict(showgrid=True),
                                       yaxis=dict(showgrid=True), zaxis=dict(showgrid=True)),
                            margin=dict(l=0, r=0, b=0, t=0), height=500
                        )
                        st.plotly_chart(fig3d, use_container_width=True)
                        st.caption(f"Malla Compuesta: {total_vertices} vértices, {total_faces} caras — {len(elements)} paths")
                    else:
                        st.warning("No se pudo extruir ninguno de los trazados del SVG.")

                    # ---- Contorno Exterior (solo si toggle activo) ----
                    if mostrar_contorno_exterior:
                        st.markdown("---")
                        st.subheader("Contorno Exterior — Path más grande")

                        closed_elems = [e for e in elements if e['closed'] and len(e['points']) >= 3]
                        source_list = closed_elems if closed_elems else elements
                        largest_elem = max(source_list, key=lambda e: compute_polygon_area(e['points']))
                        st.caption(f"Path seleccionado: {len(largest_elem['points'])} puntos (el mayor de {len(elements)} paths)")

                        pts_ext = largest_elem['points'].copy() * scale_factor
                        pts_ext[:, 1] = -pts_ext[:, 1]
                        pts_ext = subsample_path(pts_ext)
                        verts_ext, faces_ext = extrude_robust(pts_ext, height=extrusion_height, closed=largest_elem['closed'])


                        if len(verts_ext) > 0 and len(faces_ext) > 0:
                            color_ext = mesh_color if color_mode == "Monocromático" else "#1E90FF"
                            mesh_ext = create_plotly_mesh(verts_ext, faces_ext, color=color_ext, opacity=mesh_opacity)
                            if mesh_ext:
                                fig_ext = go.Figure(data=[go.Mesh3d(**mesh_ext)])
                                fig_ext.update_layout(
                                    scene=dict(aspectmode='data', xaxis=dict(showgrid=True),
                                               yaxis=dict(showgrid=True), zaxis=dict(showgrid=True)),
                                    margin=dict(l=0, r=0, b=0, t=0), height=500
                                )
                                st.plotly_chart(fig_ext, use_container_width=True)
                        else:
                            st.warning("No se pudo extruir el contorno exterior.")

            except Exception as e:
                import traceback
                st.error(f"Error procesando SVG: {e}")
                st.code(traceback.format_exc())

        else:
            # Classic PNG/JPG Pipeline
            # Leer imagen
            image = Image.open(uploaded_file)
            image_array = np.array(image.convert('RGB'))

            # Sidebar - Parametros de procesamiento
            st.sidebar.subheader("Parametros de Deteccion Canny")

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
            st.sidebar.subheader("Parametros de Extrusión 3D")

            extrusion_height = st.sidebar.slider("Altura de extrusión", 0.1, 3.0, 1.0, 0.1)
            mesh_color = st.sidebar.color_picker("Color de la malla", "#1E90FF")
            mesh_opacity = st.sidebar.slider("Opacidad", 0.1, 1.0, 0.8)

            st.sidebar.markdown("---")
            st.sidebar.subheader("Mejora de Deteccion")

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
                st.sidebar.info("GrabCut separa automáticamente el objeto del fondo")
                grabcut_iterations = st.sidebar.slider("Iteraciones GrabCut", 1, 10, 5)

            st.sidebar.markdown("---")
            st.sidebar.subheader("Calidad de Malla")

            mejora_preprocesado = st.sidebar.checkbox("Preprocesado avanzado (CLAHE)", True)
            metodo_denoise = st.sidebar.selectbox("Método de reducción de ruido",
                                                  ["Bilateral (recomendado)", "Gaussiano", "Mediana", "NL-Means (lento)"],
                                                  index=0)

            suavizar_contorno = st.sidebar.checkbox("Suavizar contorno", True)
            ventana_suavizado = st.sidebar.slider("Ventana de suavizado", 3, 15, 5, 2) if suavizar_contorno else 5

            remuestrear = st.sidebar.checkbox("Remuestrear contorno", True)
            num_puntos_malla = st.sidebar.slider("Puntos de la malla", 50, 500, 150, 10) if remuestrear else 0

            st.sidebar.markdown("---")
            st.sidebar.subheader("Ajuste de Contorno")

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
                    st.markdown("**Contorno Detectado**")

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
                    st.markdown("**Modelo 3D Extrusionado**")

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
                            st.caption(f"Malla: {len(vertices)} vértices, {len(faces)} caras")
                        else:
                            st.warning("No se pudo crear la malla 3D")
                    else:
                        st.info("Cargando cubo de demostración...")
                        st.plotly_chart(generar_cubo_3d(), width="stretch")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("👆 Carga una imagen para comenzar")