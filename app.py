import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from modules.visualization import generar_cubo_3d
from modules.command_parser import parse_commands
from modules.image_processing import (
    convertir_a_escala_grises,
    aplicar_desenfoque_gaussiano,
    detectar_bordes_canny,
    aplicar_clahe,
    aplicar_filtro_bilateral,
    aplicar_desenfoque_mediana,
    aplicar_nlmeans,
    detectar_umbrales_automaticos,
    mejorar_bordes_morfologicos,
    binarizar_adaptativo,
    segmentar_grabcut,
    detectar_bordes_multi_escala
)
from modules.contours import (
    get_contours, remuestrear_contorno, suavizar_contorno_media_movil,
    refinar_contorno_subpixel, calcular_curvatura, detectar_puntos_criticos,
    procesar_contorno_robusto
)
from modules.extrusion import normalize_points, extrude_polygon, create_plotly_mesh, sort_contour_points, ensure_closed_polygon
from modules.primitives import get_cube, get_pyramid, get_sphere, get_cylinder, get_cone, get_prisma

from modules.parametric import (
    generar_paraboloide, generar_silla_montar, generar_onda_seno,
    generar_helice, generar_espiral_conica, generar_mobius,
    generar_klein_bottle, generar_toro_anudado, generar_funcion_z,
    generar_superficie_custom, crear_mesh_plotly as crear_mesh_parametrico,
    generar_cilindro, generar_cono_parametrico, generar_toro,
    generar_pseudoesfera, generar_enneper, generar_catalan,
    generar_hiperboloide, generar_helicoide, generar_vela,
    generar_romboidal, generar_catenoide, generar_ondulatoria_parametrica,
    generar_nudo_trebol, generar_nudo_figura_ocho, generar_espiral_toroidal,
    generar_hipocicloide, generar_epicicloide,
    SUPERFICIES, EJEMPLOS_FORMULAS, FORMULAS_SUPERFICIES,
    EJEMPLOS_Z_FXY, EJEMPLOS_PARAMETRICAS
)

# Configuración de página
st.set_page_config(page_title="Vektra", layout="wide")

# Cache para procesamiento rápido
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


def procesar_imagen_avanzado(image_array, config):
    """
    Pipeline avanzado de procesamiento con opciones configurables.
    
    Args:
        image_array: Imagen RGB
        config: Dict con configuración de procesamiento
        
    Returns:
        Dict con resultados de cada etapa
    """
    results = {}
    
    # Reducir resolución para velocidad
    image_resized = reducir_resolucion(image_array)
    results['original'] = image_resized
    
    # 1. Conversión a escala de grises
    gray = convertir_a_escala_grises(image_resized)
    results['gray'] = gray
    
    # 2. Mejora de contraste (CLAHE)
    if config.get('enhance_contrast', True):
        enhanced = aplicar_clahe(gray, clip_limit=config.get('clahe_clip', 2.0))
        results['enhanced'] = enhanced
    else:
        enhanced = gray
    
    # 3. Reducción de ruido según método seleccionado
    denoise_method = config.get('denoise_method', 'bilateral')
    kernel_size = config.get('kernel_size', 5)
    
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
    
    results['denoised'] = denoised
    
    # 4. Calcular umbrales
    if config.get('auto_threshold', True):
        t1, t2 = detectar_umbrales_automaticos(denoised)
    else:
        t1 = config.get('threshold1', 50)
        t2 = config.get('threshold2', 150)
    
    results['thresholds'] = (t1, t2)
    
    # 5. Detección de bordes
    edges = detectar_bordes_canny(denoised, t1, t2)
    
    # 6. Mejora morfológica de bordes
    kernel = np.ones((3, 3), np.uint8)
    edges_improved = mejorar_bordes_morfologicos(edges, kernel_size=3)
    results['edges'] = edges_improved
    
    return results


# Mantener función legacy para compatibilidad
def procesar_imagen(image_array, kernel_size, threshold1, threshold2):
    """Pipeline básico de procesamiento (legacy)"""
    image_resized = reducir_resolucion(image_array)
    gray = convertir_a_escala_grises(image_resized)
    blurred = aplicar_desenfoque_gaussiano(gray, kernel_size)
    edges = detectar_bordes_canny(blurred, threshold1, threshold2)
    return image_resized, gray, blurred, edges

# ============ INTERFAZ PRINCIPAL ============
st.title("Motor de Vectorización de Imágenes y Renderizado Procedural")
tab_vector, tab_primitives, tab_code, tab_parametric = st.tabs(["🖼️ Vectorizador de Imagen", "📐 Generador de Figuras Primitivas", "⌨️ Editor por Código","📐 Superficies Matemáticas"])

with tab_vector:
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
            min_value=1,
            max_value=31,
            value=5,
            step=2,
            help="Aumentar para más suavizado"
        )

        threshold1 = st.sidebar.slider(
            "Umbral Bajo (Canny)",
            min_value=0,
            max_value=200,
            value=50,
            help="Umbral inferior para detección de bordes"
        )

        threshold2 = st.sidebar.slider(
            "Umbral Alto (Canny)",
            min_value=threshold1 + 10,
            max_value=500,
            value=150,
            help="Umbral superior para detección de bordes"
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("🎛️ Parámetros de Extrusión 3D")

        extrusion_height = st.sidebar.slider(
            "Altura de extrusión",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Profundidad del modelo 3D"
        )

        mesh_color = st.sidebar.color_picker(
            "Color de la malla",
            value="#1E90FF"  # Dodger blue
        )

        mesh_opacity = st.sidebar.slider(
            "Opacidad",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("💡 Mejora de Detección")
        
        modo_deteccion = st.sidebar.selectbox(
            "Modo de detección",
            ["Canny Estándar", "Umbral Adaptativo", "Multi-Escala", "Segmentación Automática"],
            index=0,
            help="Prueba diferentes modos si la imagen no tiene bordes claros"
        )
        
        # Valores por defecto para variables condicionales
        block_size_adaptive = 11
        c_adaptive = 2
        grabcut_iterations = 5
        
        if modo_deteccion == "Umbral Adaptativo":
            block_size_adaptive = st.sidebar.slider(
                "Tamaño de bloque adaptativo",
                min_value=3,
                max_value=51,
                value=11,
                step=2,
                help="Tamaño del vecindario para umbral local"
            )
            c_adaptive = st.sidebar.slider(
                "Constante C",
                min_value=-10,
                max_value=20,
                value=2,
                help="Valor a restar del umbral calculado"
            )
        
        if modo_deteccion == "Segmentación Automática":
            st.sidebar.info("💡 GrabCut separa automáticamente el objeto del fondo")
            grabcut_iterations = st.sidebar.slider(
                "Iteraciones GrabCut",
                min_value=1,
                max_value=10,
                value=5,
                help="Más iteraciones = mejor segmentación pero más lento"
            )

        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Calidad de Malla")

        mejora_preprocesado = st.sidebar.checkbox(
            "Preprocesado avanzado (CLAHE)",
            value=True,
            help="Mejora el contraste para detectar mejor los bordes"
        )

        metodo_denoise = st.sidebar.selectbox(
            "Método de reducción de ruido",
            ["Bilateral (recomendado)", "Gaussiano", "Mediana", "NL-Means (lento)"],
            index=0,
            help="Bilateral preserva bordes, NL-Means es el más preciso pero lento"
        )

        suavizar_contorno = st.sidebar.checkbox(
            "Suavizar contorno",
            value=True,
            help="Aplica suavizado para eliminar ruido del contorno"
        )

        ventana_suavizado = st.sidebar.slider(
            "Ventana de suavizado",
            min_value=3,
            max_value=15,
            value=5,
            step=2,
            help="Mayor valor = contorno más suave"
        ) if suavizar_contorno else 5

        remuestrear = st.sidebar.checkbox(
            "Remuestrear contorno",
            value=True,
            help="Distribuye los puntos uniformemente para mejor malla"
        )

        num_puntos_malla = st.sidebar.slider(
            "Puntos de la malla",
            min_value=50,
            max_value=500,
            value=150,
            step=10,
            help="Más puntos = mayor detalle pero más procesamiento"
        ) if remuestrear else 0

        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Ajuste de Contorno")
        
        modo_contorno = st.sidebar.selectbox(
            "Modo de contorno",
            ["Solo el más grande", "Todos los contornos (separados)" , "Todos los contornos (unidos)", "Todos los contornos (hull)"],
            index=0,
            help="Elige cómo manejar múltiples objetos en la imagen"
        )
        
        morph_kernel_size = st.sidebar.slider(
            "Tamaño kernel morfológico",
            min_value=3,
            max_value=21,
            value=5,
            step=2,
            help="Mayor = cierra más huecos en el contorno"
        )
        
        min_area_contorno = st.sidebar.slider(
            "Área mínima del contorno",
            min_value=10,
            max_value=5000,
            value=100,
            step=50,
            help="Ignora contornos más pequeños que este valor"
        )
        
        simplificacion_contorno = st.sidebar.select_slider(
            "Simplificación del contorno",
            options=["Ninguna", "Baja", "Media", "Alta"],
            value="Baja",
            help="Reduce la cantidad de puntos del contorno"
        )
        
        invertir_bordes = st.sidebar.checkbox(
            "Invertir bordes",
            value=False,
            help="Útil si el objeto es más oscuro que el fondo"
        )
        
        trazar_bordes = st.sidebar.checkbox(
            "Trazar bordes (mejor para dibujos)",
            value=True,
            help="Usa cierre adaptativo para seguir mejor los bordes detectados"
        )
        
        if trazar_bordes:
            cierre_iteraciones = st.sidebar.slider(
                "Iteraciones de cierre",
                min_value=1,
                max_value=20,
                value=8,
                help="Más iteraciones = conecta bordes más separados"
            )
            radio_ajuste = st.sidebar.slider(
                "Radio de ajuste a bordes",
                min_value=1,
                max_value=30,
                value=15,
                help="Radio de búsqueda para ajustar el contorno a los bordes originales"
            )
        else:
            cierre_iteraciones = 5
            radio_ajuste = 10
        
        usar_hull = st.sidebar.checkbox(
            "Usar Convex Hull",
            value=False,
            help="Envuelve el contorno en su casco convexo (ignora concavidades)"
        )
        
        metodo_ordenamiento = st.sidebar.selectbox(
            "Orden de puntos",
            ["Original (recomendado)", "Angular (solo convexos)", "Optimizado"],
            index=0,
            help="Cómo ordenar los puntos del contorno para la malla 3D"
        )

        # Mapear método de denoise
        denoise_map = {
            "Bilateral (recomendado)": "bilateral",
            "Gaussiano": "gaussian",
            "Mediana": "median",
            "NL-Means (lento)": "nlmeans"
        }
        denoise_method = denoise_map.get(metodo_denoise, "bilateral")

        # Procesar imagen con pipeline avanzado
        try:
            # Reducir resolución primero
            original = reducir_resolucion(image_array)
            gray = convertir_a_escala_grises(original)
            
            # Preprocesamiento
            if mejora_preprocesado:
                enhanced = aplicar_clahe(gray, clip_limit=2.5)
            else:
                enhanced = gray
            
            # Reducción de ruido
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
            
            # Detección de bordes según el modo seleccionado
            if modo_deteccion == "Canny Estándar":
                edges = detectar_bordes_canny(denoised, threshold1, threshold2)
                edges = mejorar_bordes_morfologicos(edges, kernel_size=3)
                
            elif modo_deteccion == "Umbral Adaptativo":
                # Umbralización adaptativa - mejor para imágenes con iluminación desigual
                binary = binarizar_adaptativo(denoised, block_size_adaptive, c_adaptive)
                # Invertir si es necesario (fondo blanco)
                if np.mean(binary) > 127:
                    binary = 255 - binary
                # Detectar bordes del binario
                edges = cv2.Canny(binary, 50, 150)
                edges = mejorar_bordes_morfologicos(edges, kernel_size=3)
                
            elif modo_deteccion == "Multi-Escala":
                # Detección en múltiples escalas - mejor para detalles variados
                edges = detectar_bordes_multi_escala(denoised, scales=[1.0, 0.5, 0.25])
                edges = mejorar_bordes_morfologicos(edges, kernel_size=3)
                
            elif modo_deteccion == "Segmentación Automática":
                # GrabCut para separar objeto del fondo
                mask = segmentar_grabcut(original, iterations=grabcut_iterations)
                # Detectar bordes de la máscara
                edges = cv2.Canny(mask, 50, 150)
                edges = mejorar_bordes_morfologicos(edges, kernel_size=5)

            # Invertir bordes si está seleccionado

            kernel_morph = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
            if trazar_bordes:
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_morph, iterations=cierre_iteraciones)
            else:
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_morph)

            # 2. Forzar el Área Mínima borrando la basura pequeña de la imagen principal
            contours_raw, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_area = np.zeros_like(edges)
            for cnt in contours_raw:
                if cv2.contourArea(cnt) >= min_area_contorno:
                    # Rellenar con blanco solo las formas que cumplen el tamaño mínimo
                    cv2.drawContours(mask_area, [cnt], -1, 255, -1)

            # Borrar de la imagen original todo lo que no esté en la máscara (las basuras)
            edges = cv2.bitwise_and(edges, mask_area)

            # --- FIN DE CORRECCIÓN ---

            if invertir_bordes:
                edges = 255 - edges

            # Mostrar resultados
            st.subheader("Resultados del Procesamiento")

            col_imgs = st.columns(4)
            col_imgs[0].image(original, caption="Original", use_container_width=True)
            col_imgs[1].image(gray, caption="Grises", use_container_width=True)
            col_imgs[2].image(blurred, caption="Preprocesado", use_container_width=True)
            col_imgs[3].image(edges, caption=f"Bordes ({modo_deteccion})", use_container_width=True)

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
                if modo_contorno == "Todos los contornos (unidos)": multi_contour_mode = "union"
                elif modo_contorno == "Todos los contornos (hull)": multi_contour_mode = "hull"
                elif modo_contorno == "Todos los contornos (separados)": multi_contour_mode = "separate"

                processed_contours_list = []
                contour_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

                if multi_contour_mode == "separate":


                    edges_procesados = edges.copy()
                    if trazar_bordes:
                        edges_procesados = cv2.morphologyEx(
                            edges_procesados,
                            cv2.MORPH_CLOSE,
                            kernel_morph,
                            iterations=cierre_iteraciones
                        )
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cv2.contourArea(cnt) >= min_area_contorno:
                            # 1. Simplificar
                            c_pts = procesar_contorno_robusto(cnt)
                            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
                            approx = cv2.approxPolyDP(cnt, epsilon, True)
                            c_pts = approx.reshape(-1, 2).astype(np.float32)

                            # 2. Suavizado
                            if suavizar_contorno and len(c_pts) > ventana_suavizado:
                                c_pts = suavizar_contorno_media_movil(c_pts, window=ventana_suavizado)
                                c_pts = suavizar_contorno_media_movil(c_pts, window=ventana_suavizado)

                            # 3. Remuestreo
                            if remuestrear and num_puntos_malla > 0:
                                c_pts = remuestrear_contorno(c_pts, num_puntos_malla)

                            # 4. Escudo de seguridad de NumPy
                            if c_pts is not None and len(c_pts) >= 3:
                                c_pts = c_pts.reshape(-1, 2) # Forzar exactamente 2 dimensiones (X,Y)
                                processed_contours_list.append(c_pts)
                                cv2.drawContours(contour_display, [c_pts.reshape(-1, 1, 2).astype(np.int32)], -1, (0, 255, 0), 2)

                    st.image(contour_display, use_container_width=True)
                    st.metric("Objetos independientes", len(processed_contours_list))

                else:
                    contour_config = {
                        'simplify_method': simplify_method, 'epsilon_factor': epsilon_factor,
                        'min_area': min_area_contorno, 'morph_op': 'close',
                        'multi_contour_mode': multi_contour_mode, 'trace_edges': trazar_bordes,
                        'close_iterations': cierre_iteraciones, 'close_kernel': 3,
                        'adjust_to_edges': trazar_bordes, 'search_radius': radio_ajuste
                    }
                    contour_points = get_contours(edges, kernel_morph, contour_config)

                    if contour_points is not None and len(contour_points) > 0:
                        # 1. Estandarizar la variable principal
                        pts = contour_points.reshape(-1, 2).astype(np.float32)

                        # 2. Aplicar Convex Hull de forma secuencial
                        if usar_hull:
                            hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.int32))
                            pts = hull.reshape(-1, 2).astype(np.float32)

                        # 3. Aplicar Suavizado
                        if suavizar_contorno and len(pts) > ventana_suavizado:
                            pts = suavizar_contorno_media_movil(pts, window=ventana_suavizado)

                        # 4. Aplicar Remuestreo
                        if remuestrear and num_puntos_malla > 0:
                            pts = remuestrear_contorno(pts, num_puntos_malla)

                        # 5. Escudo de seguridad antes de pasar a la extrusión 3D
                        if pts is not None and len(pts) >= 3:
                            pts = pts.reshape(-1, 2) # Forzar exactamente 2 dimensiones (X,Y)
                            processed_contours_list.append(pts)

                            # Dibujar en la UI (Azul = detección base, Verde = malla final ajustada)
                            cv2.drawContours(contour_display, [contour_points.reshape(-1, 1, 2).astype(np.int32)], -1, (255, 100, 100), 1)
                            cv2.drawContours(contour_display, [pts.reshape(-1, 1, 2).astype(np.int32)], -1, (0, 255, 0), 2)
                            st.image(contour_display, use_container_width=True)
                        else:
                            st.warning("⚠️ Los ajustes eliminaron la geometría. Reduce la simplificación o desactiva el remuestreo.")
                    else:
                        st.warning("No se detectaron contornos. Ajusta los umbrales.")

            with col_3d:

                st.markdown("**🔷 Modelo 3D Extrusionado**")
                mapa_orden = {
                    "Original (recomendado)" : "original",
                    "Angular (solo convexos)": "angular",
                    "Optimizado":"optimized"
                }

                metodo_orden_interno = mapa_orden.get(metodo_ordenamiento, "original")
                if len(processed_contours_list) > 0:
                    fig = go.Figure()
                    total_vertices = 0
                    total_faces = 0

                    # 1. Calcular el centro global de todas las formas para mantener su posición relativa
                    all_pts = np.vstack(processed_contours_list)
                    min_coords = np.min(all_pts, axis=0)
                    max_coords = np.max(all_pts, axis=0)
                    scale = 2.0 / max(1e-5, np.max(max_coords - min_coords))
                    center_global = (min_coords + max_coords) / 2.0

                    # Función interna para extruir polígonos complejos con soporte para agujeros
                    def extruir_con_agujeros(outer, holes_list, height=0.5):
                        vertices = []
                        faces = []

                        # Asegurar polígono cerrado y ordenado para el contorno exterior
                        outer = sort_contour_points(outer, method="original")
                        outer = ensure_closed_polygon(outer)
                        outer_norm = (outer - center_global) * scale
                        n_outer = len(outer_norm)

                        # Procesar y normalizar los agujeros
                        norm_holes = []
                        for h in holes_list:
                            h_sorted = sort_contour_points(h, method=metodo_orden_interno)
                            h_closed = ensure_closed_polygon(h_sorted)
                            norm_holes.append((h_closed - center_global) * scale)

                        # --- A. Paredes del contorno exterior ---
                        for p in outer_norm:
                            vertices.append([p[0], p[1], 0.0])       # Base
                        for p in outer_norm:
                            vertices.append([p[0], p[1], height])    # Tapa
                        for i in range(n_outer):
                            j = (i + 1) % n_outer
                            faces.append([i, j, i + n_outer])
                            faces.append([j, j + n_outer, i + n_outer])

                        vertex_offset = 2 * n_outer

                        # --- B. Paredes de los agujeros internos ---
                        for h_norm in norm_holes:
                            n_h = len(h_norm)
                            for p in h_norm:
                                vertices.append([p[0], p[1], 0.0])   # Base del agujero
                            for p in h_norm:
                                vertices.append([p[0], p[1], height]) # Tapa del agujero
                            for i in range(n_h):
                                j = (i + 1) % n_h
                                base_i = vertex_offset + i
                                base_j = vertex_offset + j
                                top_i = vertex_offset + i + n_h
                                top_j = vertex_offset + j + n_h
                                # Invertimos el orden de las caras para que miren hacia adentro del hueco
                                faces.append([base_i, top_i, base_j])
                                faces.append([base_j, top_i, top_j])
                            vertex_offset += 2 * n_h

                        # --- C. Tapas con sustracción de agujeros usando Delaunay ---
                        all_lists = [outer_norm] + norm_holes
                        flat_pts = np.vstack(all_lists)

                        # Mapear índices planos a los índices de nuestros vértices 3D
                        base_map = []
                        top_map = []
                        for i in range(n_outer):
                            base_map.append(i)
                            top_map.append(i + n_outer)

                        curr_offset = 2 * n_outer
                        for h_norm in norm_holes:
                            n_h = len(h_norm)
                            for i in range(n_h):
                                base_map.append(curr_offset + i)
                                top_map.append(curr_offset + i + n_h)
                            curr_offset += 2 * n_h

                        # Calcular la triangulación conjunta de Delaunay
                        from scipy.spatial import Delaunay
                        tri = Delaunay(flat_pts)

                        for triangle in tri.simplices:
                            # Calcular el centro matemático (centroide) del triángulo
                            centroid = np.mean(flat_pts[triangle], axis=0)

                            # Validar que esté ADENTRO del exterior pero AFUERA de todos los agujeros
                            inside_outer = cv2.pointPolygonTest(outer_norm.astype(np.float32), (float(centroid[0]), float(centroid[1])), False) >= 0
                            if inside_outer:
                                inside_hole = False
                                for h_norm in norm_holes:
                                    if cv2.pointPolygonTest(h_norm.astype(np.float32), (float(centroid[0]), float(centroid[1])), False) > 0:
                                        inside_hole = True
                                        break
                                # Si cumple la condición de donut, agregamos el triángulo a la base y la tapa
                                if not inside_hole:
                                    faces.append([base_map[triangle[0]], base_map[triangle[2]], base_map[triangle[1]]])
                                    faces.append([top_map[triangle[0]], top_map[triangle[1]], top_map[triangle[2]]])

                        return np.array(vertices), faces

                    # 2. Agrupación espacial jerárquica automática (Padres vs Agujeros)
                    valid_contours = [pc for pc in processed_contours_list if len(pc) >= 3]
                    # Ordenar por área de mayor a menor para procesar figuras envolventes primero
                    sorted_contours = sorted(valid_contours, key=lambda c: cv2.contourArea(c.astype(np.float32)), reverse=True)

                    outer_shapes = []
                    assigned = np.zeros(len(sorted_contours), dtype=bool)

                    for i, c_ext in enumerate(sorted_contours):
                        if assigned[i]:
                            continue
                        holes = []
                        for j in range(i + 1, len(sorted_contours)):
                            if assigned[j]:
                                continue
                            # Probar si el primer punto del contorno j está dentro del contorno i
                            test_pt = sorted_contours[j][0]
                            is_inside = cv2.pointPolygonTest(c_ext.astype(np.float32), (float(test_pt[0]), float(test_pt[1])), False) >= 0
                            if is_inside:
                                holes.append(sorted_contours[j])
                                assigned[j] = True
                        outer_shapes.append((c_ext, holes))
                        assigned[i] = True

                    # 3. Generar las mallas 3D finales
                    for outer, holes in outer_shapes:
                        vertices, faces = extruir_con_agujeros(outer, holes, height=extrusion_height)
                        mesh_data = create_plotly_mesh(vertices, faces, color=mesh_color, opacity=mesh_opacity)

                        if mesh_data:
                            fig.add_trace(go.Mesh3d(**mesh_data))
                            total_vertices += len(vertices)
                            total_faces += len(faces)

                    if len(fig.data) > 0:
                        fig.update_layout(
                            scene=dict(
                                aspectmode='data',
                                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                                zaxis=dict(showgrid=True, gridcolor='lightgray'),
                            ),
                            margin=dict(l=0, r=0, b=0, t=0),
                            height=450
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"📊 Malla Compleja: {total_vertices} vértices, {total_faces} caras en {len(fig.data)} figuras estructuradas")
                    else:
                        st.warning("No se pudo crear la malla 3D")
                else:
                    st.info("Cargando cubo de demostración...")
                    st.plotly_chart(generar_cubo_3d(), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("👆 Carga una imagen para comenzar")


with tab_primitives:
    st.subheader("Generador de Primitivas Geométricas 3D")
    col_ui, col_viz = st.columns([1, 2])

    with col_ui:
        st.markdown("**🎨 Configuración**")

        tipo_figura = st.selectbox(
            "Selecciona la figura",
            ["Cubo", "Esfera", "Cilindro", "Pirámide", "Cono", "Prisma"],
            key="sel_prim"
        )

        color_p = st.color_picker("Color de la figura", "#00CED1", key="col_prim")
        alpha_p = st.slider("Opacidad", 0.1, 1.0, 0.8, key="alpha_prim")
        st.info("Figuras generadas proceduralmente en el origen (0,0,0).")

    with col_viz:
        st.markdown("**🔷 Previsualización 3D**")
        v, i, j, k = None, None, None, None
        m_args = {'color': color_p, 'opacity': alpha_p, 'flatshading': True}

        if tipo_figura == "Cubo": v, i, j, k = get_cube()
        elif tipo_figura == "Pirámide": v, i, j, k = get_pyramid()
        elif tipo_figura == "Esfera": v, i, j, k = get_sphere(); m_args['alphahull'] = 0
        elif tipo_figura == "Cilindro": v, i, j, k = get_cylinder(); m_args['alphahull'] = 0
        elif tipo_figura == "Cono": v, i, j, k = get_cone(); m_args['alphahull'] = 0
        elif tipo_figura == "Prisma":
            v, i, j, k = get_prisma(n=6)
            m_args['alphahull'] = 0

        if v is not None:
            fig_p = go.Figure(data=[go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=i, j=j, k=k, **m_args)])
            fig_p.update_layout(scene=dict(aspectmode='cube'), margin=dict(l=0, r=0, b=0, t=0), height=500)
            st.plotly_chart(fig_p, use_container_width=True)

with tab_code:
    st.subheader("⌨️ Editor de Código para Figuras 3D")

    example_code = """
    # Ejemplo de comandos
        cube
        sphere
        prisma n=8
        cylinder
        """

    code = st.text_area(
        "Escribí comandos para generar figuras 3D:",
        value=example_code,
        height=300
    )

    # Color picker para las figuras
    col1, col2 = st.columns([1, 4])
    with col1:
        figura_color = st.color_picker(
            "Color de las figuras",
            value="#00CED1",
            key="code_editor_color"
        )
    
    with col2:
        st.write("")  # Para alinear verticalmente

    if st.button("▶ Ejecutar código"):
        try:
            figures = parse_commands(code)
            # Guardar figuras en session_state
            st.session_state.code_figures = figures
            st.session_state.code_executed = True
        except Exception as e:
            st.error(f"❌ Error en el código: {e}")
            st.session_state.code_executed = False
    
    # Mostrar figuras guardadas con color dinámico
    if hasattr(st.session_state, 'code_executed') and st.session_state.code_executed:
        for v, i, j, k in st.session_state.code_figures:
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=v[:,0], y=v[:,1], z=v[:,2],
                    i=i, j=j, k=k,
                    opacity=0.9,
                    color=figura_color,
                    flatshading=False
                )
            ])
            fig.update_layout(
                scene=dict(aspectmode='cube'),
                margin=dict(l=0, r=0, b=0, t=0),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Haz clic en 'Ejecutar código' para generar las figuras")
    
with tab_parametric:
    st.subheader("📐 Superficies Paramétricas")
    st.markdown("Genera superficies 3D usando **fórmulas matemáticas de vectores**")
    
    col_formula, col_vista = st.columns([1, 2])
    
    with col_formula:
        st.markdown("**🔢 Configuración**")
        
        # Modo de entrada
        modo_superficie = st.radio(
            "Modo de generación",
            options=["Superficies predefinidas", "Fórmula z = f(x,y)", "Superficie paramétrica"],
            horizontal=True
        )
        
        if modo_superficie == "Superficies predefinidas":
            # Selector de superficie predefinida con etiquetas descriptivas
            # Diccionario de emojis para cada superficie
            emojis_superficies = {
                "paraboloide": "🔵",
                "silla_montar": "🐴",
                "onda_seno": "🌊",
                "cilindro": "📦",
                "cono": "🔺",
                "toro": "🍩",
                "pseudoesfera": "🌐",
                "enneper": "🌈",
                "catalan": "🎨",
                "hiperboloide": "⚡",
                "helicoide": "🌀",
                "vela": "⛵",
                "romboidal": "◇",
                "catenoide": "⛓️",
                "ondulatoria": "〰️",
                "helice": "🧬",
                "espiral_conica": "🌀",
                "nudo_trebol": "🍀",
                "nudo_figura_ocho": "8️⃣",
                "espiral_toroidal": "🌪️",
                "hipocicloide": "🔄",
                "epicicloide": "⭕",
                "mobius": "♾️",
                "klein": "🍶",
                "toro_anudado": "🔗"
            }
            
            # Crear labels dinámicamente con emojis y fórmulas
            labels_superficies = {
                sup: f"{emojis_superficies.get(sup, '')} {sup.replace('_', ' ').title()} ({FORMULAS_SUPERFICIES.get(sup, 'N/A')})"
                for sup in sorted(SUPERFICIES.keys())
            }
            
            superficie_sel = st.selectbox(
                "Selecciona una superficie",
                options=sorted(SUPERFICIES.keys()),
                format_func=lambda x: labels_superficies.get(x, x.replace("_", " ").title())
            )
            
            # Parámetros según la superficie
            param_resolution = st.slider("Resolución", 20, 100, 50, key="param_res")
            param_color = st.color_picker("Color", value="#9b59b6", key="param_color")
            param_opacity = st.slider("Opacidad", 0.1, 1.0, 0.85, key="param_opacity")
            
            # Mostrar fórmula de la superficie seleccionada
            with st.expander("📐 Ver fórmula"):
                st.code(EJEMPLOS_FORMULAS, language="python")
            
            # Generar superficie de forma dinámica
            try:
                # Parámetros específicos por tipo de superficie
                if superficie_sel == "paraboloide":
                    param_a = st.slider("Escala A", 0.5, 2.0, 1.0)
                    param_h = st.slider("Altura máx", 1.0, 4.0, 2.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](a=param_a, height=param_h, resolution=param_resolution)
                
                elif superficie_sel == "silla_montar":
                    param_size = st.slider("Tamaño", 1.0, 4.0, 2.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](size=param_size, resolution=param_resolution)
                
                elif superficie_sel == "onda_seno":
                    param_amp = st.slider("Amplitud", 0.5, 2.0, 1.0)
                    param_freq = st.slider("Frecuencia", 0.5, 3.0, 1.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](amplitud=param_amp, frecuencia=param_freq, resolution=param_resolution)
                
                elif superficie_sel == "helice":
                    param_vueltas = st.slider("Vueltas", 1, 6, 3)
                    param_paso = st.slider("Paso", 0.2, 1.0, 0.5)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](vueltas=param_vueltas, paso=param_paso, resolution=param_resolution)
                
                elif superficie_sel == "espiral_conica":
                    param_vueltas = st.slider("Vueltas", 2, 8, 4)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](vueltas=param_vueltas, resolution=param_resolution)
                
                elif superficie_sel == "mobius":
                    param_ancho = st.slider("Ancho banda", 0.2, 0.8, 0.4)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](ancho=param_ancho, resolution=param_resolution)
                
                elif superficie_sel == "toro":
                    param_r_mayor = st.slider("Radio mayor", 0.5, 2.0, 1.0)
                    param_r_menor = st.slider("Radio menor", 0.2, 0.8, 0.4)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](radio_mayor=param_r_mayor, radio_menor=param_r_menor, resolution=param_resolution)
                
                elif superficie_sel == "toro_anudado":
                    param_p = st.slider("Parámetro p", 2, 5, 2)
                    param_q = st.slider("Parámetro q", 2, 7, 3)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](p=param_p, q=param_q, resolution=param_resolution)
                
                elif superficie_sel == "cono":
                    param_radio = st.slider("Radio base", 0.5, 2.0, 1.0)
                    param_altura = st.slider("Altura", 1.0, 4.0, 2.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](radio_base=param_radio, altura=param_altura, resolution=param_resolution)
                
                elif superficie_sel == "cilindro":
                    param_radio = st.slider("Radio", 0.5, 2.0, 1.0)
                    param_altura = st.slider("Altura", 1.0, 4.0, 2.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](radio=param_radio, altura=param_altura, resolution=param_resolution)
                
                elif superficie_sel == "pseudoesfera":
                    param_altura = st.slider("Altura", 0.5, 3.0, 2.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](altura=param_altura, resolution=param_resolution)
                
                elif superficie_sel == "hiperboloide":
                    param_a = st.slider("Escala a", 0.5, 2.0, 1.0)
                    param_b = st.slider("Escala b", 0.5, 2.0, 1.0)
                    param_c = st.slider("Escala c", 0.5, 2.0, 1.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](a=param_a, b=param_b, c=param_c, resolution=param_resolution)
                
                elif superficie_sel == "helicoide":
                    param_paso = st.slider("Paso", 0.2, 1.0, 0.3)
                    param_vueltas = st.slider("Vueltas", 1, 5, 3)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](paso=param_paso, vueltas=param_vueltas, resolution=param_resolution)
                
                elif superficie_sel == "catenoide":
                    param_radio = st.slider("Radio", 0.5, 2.0, 1.0)
                    param_altura = st.slider("Altura", 1.0, 4.0, 3.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](radio=param_radio, altura=param_altura, resolution=param_resolution)
                
                elif superficie_sel == "ondulatoria":
                    param_amp = st.slider("Amplitud", 0.5, 2.0, 1.0)
                    param_n = st.slider("Frecuencia n", 1, 5, 2)
                    param_m = st.slider("Frecuencia m", 1, 5, 3)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](amplitud=param_amp, freq_n=param_n, freq_m=param_m, resolution=param_resolution)
                
                elif superficie_sel == "hipocicloide":
                    param_R = st.slider("Radio mayor (R)", 3.0, 8.0, 5.0)
                    param_r = st.slider("Radio menor (r)", 1.0, 4.0, 3.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](R=param_R, r=param_r, resolution=param_resolution)
                
                elif superficie_sel == "epicicloide":
                    param_R = st.slider("Radio mayor (R)", 3.0, 8.0, 5.0)
                    param_r = st.slider("Radio menor (r)", 1.0, 4.0, 2.0)
                    param_k = st.slider("Frecuencia (k)", 1, 5, 3)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](R=param_R, r=param_r, k=param_k, resolution=param_resolution)
                
                elif superficie_sel == "enneper":
                    param_size = st.slider("Tamaño", 0.5, 3.0, 1.5)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](size=param_size, resolution=param_resolution)
                
                elif superficie_sel == "klein":
                    st.info("💡 Botella de Klein: superficie no orientable sin borde")
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](resolution=param_resolution)
                
                elif superficie_sel == "nudo_trebol":
                    st.info("💡 Nudo de trébol: curva 3D cerrada")
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](resolution=param_resolution * 5)
                
                elif superficie_sel == "nudo_figura_ocho":
                    st.info("💡 Nudo figura-ocho: curva 3D cerrada")
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](resolution=param_resolution * 5)
                
                elif superficie_sel == "espiral_toroidal":
                    st.info("💡 Espiral toroidal: curva sobre un toro")
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](resolution=param_resolution * 5)
                
                else:
                    # Para superficies sin parámetros especiales, llamar con solo resolution
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](resolution=param_resolution)
                    
            except Exception as e:
                st.error(f"Error generando superficie: {e}")
                vertices_param, faces_param = np.array([]), []
        
        elif modo_superficie == "Fórmula z = f(x,y)":
            st.markdown("**Ingresa tu fórmula:**")
            
            # Ejemplos
            with st.expander("📚 Ver ejemplos de fórmulas"):
                col_selec, col_code = st.columns([1, 2])
                
                with col_selec:
                    st.markdown("**Ejemplos disponibles:**")
                    ejemplo_seleccionado = st.selectbox(
                        "Elige un ejemplo",
                        options=list(EJEMPLOS_Z_FXY.keys()),
                        key="ejemplo_z_select"
                    )
                
                with col_code:
                    st.markdown(f"**Fórmula:** {ejemplo_seleccionado}")
                    st.code(EJEMPLOS_Z_FXY[ejemplo_seleccionado], language="python")
                
                # Botón para usar el ejemplo
                if st.button("📌 Usar este ejemplo", key="usar_ejemplo_z"):
                    st.session_state.usar_formula_ejemplo = ejemplo_seleccionado
            
            # Usar fórmula del ejemplo si se seleccionó
            formula_default = EJEMPLOS_Z_FXY.get(
                st.session_state.get("usar_formula_ejemplo", ""),
                "sin(sqrt(x**2 + y**2))"
            )
            
            formula_z = st.text_input(
                "z = f(x, y)",
                value=formula_default,
                help="Usa: sin, cos, tan, exp, log, sqrt, abs, pi, e"
            )
            
            col_range1, col_range2 = st.columns(2)
            with col_range1:
                x_min = st.number_input("X mín", value=-3.0)
                y_min = st.number_input("Y mín", value=-3.0)
            with col_range2:
                x_max = st.number_input("X máx", value=3.0)
                y_max = st.number_input("Y máx", value=3.0)
            
            param_resolution = st.slider("Resolución", 20, 80, 50, key="formula_res")
            param_color = st.color_picker("Color", value="#3498db", key="formula_color")
            param_opacity = st.slider("Opacidad", 0.1, 1.0, 0.85, key="formula_opacity")
            
            try:
                vertices_param, faces_param = generar_funcion_z(
                    formula_z,
                    x_min=x_min, x_max=x_max,
                    y_min=y_min, y_max=y_max,
                    resolution=param_resolution
                )
            except Exception as e:
                st.error(f"Error en fórmula: {e}")
                vertices_param, faces_param = np.array([]), []
        
        else:  # Superficie paramétrica completa
            st.markdown("**Vector r(u,v) = (x, y, z):**")
            
            with st.expander("📚 Ver ejemplos paramétricos"):
                col_selec, col_code = st.columns([1, 2])
                
                with col_selec:
                    st.markdown("**Ejemplos disponibles:**")
                    ejemplo_param = st.selectbox(
                        "Elige un ejemplo",
                        options=list(EJEMPLOS_PARAMETRICAS.keys()),
                        key="ejemplo_param_select"
                    )
                    
                    data = EJEMPLOS_PARAMETRICAS[ejemplo_param]
                    st.markdown(f"**Descripción:** {data['descripcion']}")
                
                with col_code:
                    st.markdown(f"**Fórmulas paramétricas:**")
                    codigo_param = f"""# {ejemplo_param}
x: {data['x']}
y: {data['y']}
z: {data['z']}"""
                    st.code(codigo_param, language="python")
                
                # Botón para usar el ejemplo
                if st.button("📌 Usar este ejemplo", key="usar_ejemplo_param"):
                    st.session_state.usar_param_ejemplo = ejemplo_param
            
            # Usar parámetros del ejemplo si se seleccionó
            ejemplo_activo = st.session_state.get("usar_param_ejemplo", "Toro")
            ejemplo_data = EJEMPLOS_PARAMETRICAS.get(ejemplo_activo, EJEMPLOS_PARAMETRICAS["Toro"])
            
            eq_x = st.text_input("x(u,v) =", value=ejemplo_data["x"])
            eq_y = st.text_input("y(u,v) =", value=ejemplo_data["y"])
            eq_z = st.text_input("z(u,v) =", value=ejemplo_data["z"])
            
            st.markdown("**Rangos de parámetros:**")
            col_u, col_v = st.columns(2)
            with col_u:
                u_min = st.number_input("u mín", value=float(ejemplo_data["u_rango"][0]))
                u_max = st.number_input("u máx", value=float(ejemplo_data["u_rango"][1]))
            with col_v:
                v_min = st.number_input("v mín", value=float(ejemplo_data["v_rango"][0]))
                v_max = st.number_input("v máx", value=float(ejemplo_data["v_rango"][1]))
            
            param_resolution = st.slider("Resolución", 20, 80, 40, key="custom_res")
            param_color = st.color_picker("Color", value="#e74c3c", key="custom_color")
            param_opacity = st.slider("Opacidad", 0.1, 1.0, 0.85, key="custom_opacity")
            
            try:
                vertices_param, faces_param = generar_superficie_custom(
                    eq_x, eq_y, eq_z,
                    u_min=u_min, u_max=u_max,
                    v_min=v_min, v_max=v_max,
                    resolution=param_resolution
                )
            except Exception as e:
                st.error(f"Error en ecuaciones: {e}")
                vertices_param, faces_param = np.array([]), []
    
    with col_vista:
        st.markdown("**🔷 Visualización 3D**")
        
        if len(vertices_param) > 0 and len(faces_param) > 0:
            mesh_param = crear_mesh_parametrico(
                vertices_param, faces_param,
                color=param_color, opacity=param_opacity,
                name="Superficie"
            )
            
            fig_param = go.Figure(data=[go.Mesh3d(**mesh_param)])
            
            # Agregar colorscale basado en Z
            fig_param.data[0].update(
                intensity=vertices_param[:, 2],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Z")
            )
            
            fig_param.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                height=500,
                title=f"Superficie Matemática"
            )
            
            st.plotly_chart(fig_param, use_container_width=True)
            
            # Estadísticas
            stats1, stats2, stats3 = st.columns(3)
            with stats1:
                st.metric("Vértices", len(vertices_param))
            with stats2:
                st.metric("Caras", len(faces_param))
            with stats3:
                z_range = vertices_param[:, 2].max() - vertices_param[:, 2].min()
                st.metric("Rango Z", f"{z_range:.2f}")
            
            # Información adicional
            with st.expander("📊 Información de la superficie"):
                st.markdown(f"""
                **Estadísticas:**
                - X: [{vertices_param[:, 0].min():.2f}, {vertices_param[:, 0].max():.2f}]
                - Y: [{vertices_param[:, 1].min():.2f}, {vertices_param[:, 1].max():.2f}]
                - Z: [{vertices_param[:, 2].min():.2f}, {vertices_param[:, 2].max():.2f}]
                - Resolución: {param_resolution}×{param_resolution}
                """)
        else:
            st.warning("No se pudo generar la superficie. Verifica los parámetros.")
