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
    detectar_bordes_canny
)
from modules.contours import get_contours
from modules.extrusion import normalize_points, extrude_polygon, create_plotly_mesh, sort_contour_points
from modules.primitives import get_cube, get_pyramid, get_sphere, get_cylinder, get_cone, get_prisma

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


def procesar_imagen(image_array, kernel_size, threshold1, threshold2):
    """Pipeline completo de procesamiento"""
    # Reducir resolucion para velocidad
    image_resized = reducir_resolucion(image_array)
    
    # Conversion a escala de grises (función importada del módulo)
    gray = convertir_a_escala_grises(image_resized)
    
    # Desenfoque gaussiano (función importada del módulo)
    blurred = aplicar_desenfoque_gaussiano(gray, kernel_size)
    
    # Deteccion de bordes Canny (función importada del módulo)
    edges = detectar_bordes_canny(blurred, threshold1, threshold2)
    
    return image_resized, gray, blurred, edges

# ============ INTERFAZ PRINCIPAL ============
st.title("Motor de Vectorización de Imágenes y Renderizado Procedural")
tab_vector, tab_primitives, tab_code = st.tabs(["🖼️ Vectorizador de Imagen", "📐 Generador de Figuras Primitivas", "⌨️ Editor por Código"])

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
