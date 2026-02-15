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
    mejorar_bordes_morfologicos
)
from modules.contours import get_contours, remuestrear_contorno, suavizar_contorno_media_movil
from modules.extrusion import normalize_points, extrude_polygon, create_plotly_mesh, sort_contour_points
from modules.primitives import get_cube, get_pyramid, get_sphere, get_cylinder, get_cone, get_prisma

from modules.parametric import (
    generar_paraboloide, generar_silla_montar, generar_onda_seno,
    generar_helice, generar_espiral_conica, generar_mobius,
    generar_klein_bottle, generar_toro_anudado, generar_funcion_z,
    generar_superficie_custom, crear_mesh_plotly as crear_mesh_parametrico,
    SUPERFICIES, EJEMPLOS_FORMULAS
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

        # Crear kernel para operaciones morfológicas
        # kernel_contour = np.ones((5, 5), np.uint8)

        # Procesar imagen
        try:
            original, gray, blurred, edges = procesar_imagen(
                image_array, kernel_size, threshold1, threshold2
            )

            # Mostrar resultados
            st.subheader("Resultados del Procesamiento")

            col_imgs = st.columns(4)
            col_imgs[0].image(original, caption="Original", use_container_width=True)
            col_imgs[1].image(gray, caption="Grises", use_container_width=True)
            col_imgs[2].image(blurred, caption="Blur", use_container_width=True)
            col_imgs[3].image(edges, caption="Canny", use_container_width=True)

            st.markdown("---")
            col_contour, col_3d = st.columns([1, 2])

            with col_contour:

                st.markdown("**🔍 Contorno Detectado**")
                contour_points = get_contours(edges, np.ones((5, 5), np.uint8))

                if len(contour_points) > 0:
                    contour_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(contour_display, [contour_points.reshape(-1, 1, 2).astype(np.int32)], -1, (0, 255, 0), 2)
                    st.image(contour_display, use_container_width=True)
                    st.metric("Puntos", len(contour_points))

                else:
                    st.warning("No se detectaron contornos.")

            with col_3d:

                st.markdown("**🔷 Modelo 3D Extrusionado**")

                if len(contour_points) >= 3:
                    ordered = sort_contour_points(contour_points)
                    norm = normalize_points(ordered, 2.0)
                    vertices, faces = extrude_polygon(norm, height=extrusion_height, triangulate=True)
                    mesh_data = create_plotly_mesh(vertices, faces, color=mesh_color, opacity=mesh_opacity)
                    fig = go.Figure(data=[go.Mesh3d(**mesh_data)])
                    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), height=400)
                    st.plotly_chart(fig, use_container_width=True)

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

    if st.button("▶ Ejecutar código"):
        try:
            figures = parse_commands(code)

            for v, i, j, k in figures:
                fig = go.Figure(data=[
                    go.Mesh3d(
                        x=v[:,0], y=v[:,1], z=v[:,2],
                        i=i, j=j, k=k,
                        opacity=0.9,
                        color="lightblue",
                        flatshading=False
                    )
                ])
                fig.update_layout(
                    scene=dict(aspectmode='cube'),
                    margin=dict(l=0, r=0, b=0, t=0),
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error en el código: {e}")

    else:
        st.info("👆 Carga una imagen para comenzar el procesamiento")
    
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
            # Selector de superficie predefinida
            superficie_sel = st.selectbox(
                "Selecciona una superficie",
                options=[
                    "paraboloide", "silla_montar", "onda_seno", 
                    "helice", "espiral_conica", "mobius", 
                    "klein", "toro_anudado"
                ],
                format_func=lambda x: {
                    "paraboloide": "🔵 Paraboloide (z = x² + y²)",
                    "silla_montar": "🐴 Silla de Montar (z = x² - y²)",
                    "onda_seno": "🌊 Onda Seno (z = sin(x)cos(y))",
                    "helice": "🧬 Hélice (espiral 3D)",
                    "espiral_conica": "🌀 Espiral Cónica",
                    "mobius": "♾️ Banda de Möbius",
                    "klein": "🍶 Botella de Klein",
                    "toro_anudado": "🔗 Nudo Toroidal"
                }.get(x, x)
            )
            
            # Parámetros según la superficie
            param_resolution = st.slider("Resolución", 20, 100, 50, key="param_res")
            param_color = st.color_picker("Color", value="#9b59b6", key="param_color")
            param_opacity = st.slider("Opacidad", 0.1, 1.0, 0.85, key="param_opacity")
            
            # Generar superficie
            try:
                if superficie_sel == "paraboloide":
                    param_a = st.slider("Escala A", 0.5, 2.0, 1.0)
                    param_h = st.slider("Altura máx", 1.0, 4.0, 2.0)
                    vertices_param, faces_param = generar_paraboloide(a=param_a, height=param_h, resolution=param_resolution)
                
                elif superficie_sel == "silla_montar":
                    param_size = st.slider("Tamaño", 1.0, 4.0, 2.0)
                    vertices_param, faces_param = generar_silla_montar(size=param_size, resolution=param_resolution)
                
                elif superficie_sel == "onda_seno":
                    param_amp = st.slider("Amplitud", 0.5, 2.0, 1.0)
                    param_freq = st.slider("Frecuencia", 0.5, 3.0, 1.0)
                    vertices_param, faces_param = generar_onda_seno(amplitud=param_amp, frecuencia=param_freq, resolution=param_resolution)
                
                elif superficie_sel == "helice":
                    param_vueltas = st.slider("Vueltas", 1, 6, 3)
                    param_paso = st.slider("Paso", 0.2, 1.0, 0.5)
                    vertices_param, faces_param = generar_helice(vueltas=param_vueltas, paso=param_paso, resolution=param_resolution)
                
                elif superficie_sel == "espiral_conica":
                    param_vueltas = st.slider("Vueltas", 2, 8, 4)
                    vertices_param, faces_param = generar_espiral_conica(vueltas=param_vueltas, resolution=param_resolution)
                
                elif superficie_sel == "mobius":
                    param_ancho = st.slider("Ancho banda", 0.2, 0.8, 0.4)
                    vertices_param, faces_param = generar_mobius(ancho=param_ancho, resolution=param_resolution)
                
                elif superficie_sel == "klein":
                    vertices_param, faces_param = generar_klein_bottle(resolution=param_resolution)
                
                elif superficie_sel == "toro_anudado":
                    param_p = st.slider("Parámetro p", 2, 5, 2)
                    param_q = st.slider("Parámetro q", 2, 7, 3)
                    vertices_param, faces_param = generar_toro_anudado(p=param_p, q=param_q, resolution=param_resolution)
                else:
                    vertices_param, faces_param = generar_paraboloide()
                    
            except Exception as e:
                st.error(f"Error generando superficie: {e}")
                vertices_param, faces_param = np.array([]), []
        
        elif modo_superficie == "Fórmula z = f(x,y)":
            st.markdown("**Ingresa tu fórmula:**")
            
            # Ejemplos
            with st.expander("📚 Ver ejemplos de fórmulas"):
                st.code("""
# Paraboloide
x**2 + y**2

# Silla de montar
x**2 - y**2

# Ondas
sin(x) * cos(y)

# Gaussiana
exp(-(x**2 + y**2))

# Ondas radiales
sin(sqrt(x**2 + y**2))

# Crestas
sin(x) + sin(y)
""", language="python")
            
            formula_z = st.text_input(
                "z = f(x, y)",
                value="sin(sqrt(x**2 + y**2))",
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
                st.code("""
# Esfera (radio 1)
x: cos(u) * sin(v)
y: sin(u) * sin(v)
z: cos(v)
u: [0, 2π], v: [0, π]

# Toro
x: (2 + cos(v)) * cos(u)
y: (2 + cos(v)) * sin(u)
z: sin(v)
u: [0, 2π], v: [0, 2π]

# Cono
x: u * cos(v)
y: u * sin(v)
z: u
u: [0, 2], v: [0, 2π]
""", language="python")
            
            eq_x = st.text_input("x(u,v) =", value="(2 + cos(v)) * cos(u)")
            eq_y = st.text_input("y(u,v) =", value="(2 + cos(v)) * sin(u)")
            eq_z = st.text_input("z(u,v) =", value="sin(v)")
            
            st.markdown("**Rangos de parámetros:**")
            col_u, col_v = st.columns(2)
            with col_u:
                u_min = st.number_input("u mín", value=0.0)
                u_max = st.number_input("u máx", value=6.28)  # 2π
            with col_v:
                v_min = st.number_input("v mín", value=0.0)
                v_max = st.number_input("v máx", value=6.28)  # 2π
            
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
