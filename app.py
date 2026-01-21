import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from modules.visualization import generar_cubo_3d
from modules.image_processing import (
    convertir_a_escala_grises,
    aplicar_desenfoque_gaussiano,
    detectar_bordes_canny
)
from modules.contours import get_contours
from modules.extrusion import normalize_points, extrude_polygon, create_plotly_mesh, sort_contour_points

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
st.markdown("Carga una imagen JPG/PNG")
st.markdown("---")

# Carga de archivo
uploaded_file = st.file_uploader(
    "Carga una imagen (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
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
        step=0.05
    )

    # Crear kernel para operaciones morfológicas
    kernel_contour = np.ones((5, 5), np.uint8)
    
    # Procesar imagen
    try:
        original, gray, blurred, edges = procesar_imagen(
            image_array, kernel_size, threshold1, threshold2
        )
        
        # Mostrar resultados
        st.subheader("Resultados del Procesamiento")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**📷 Imagen Original**")
            st.image(original, width='stretch')
        
        with col2:
            st.markdown("**🔍 Escala de Grises**")
            st.image(gray, width='stretch', clamp=True)
        
        with col3:
            st.markdown("**🌫️ Desenfoque Gaussiano**")
            st.image(blurred, width='stretch', clamp=True)
        
        with col4:
            st.markdown("**✨ Bordes Detectados (Canny)**")
            st.image(edges, width='stretch', clamp=True)

        st.markdown("---")
        st.subheader("📐 Extracción de Contornos y Modelado 3D")

        # Columna para contornos y 3D
        col_contour, col_3d = st.columns([1, 2])

        with col_contour:
            st.markdown("**🔍 Contorno Detectado**")

            # Mostrar imagen con contorno dibujado
            if len(edges.shape) == 2:  # Si es imagen binaria
                contour_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            else:
                contour_display = edges.copy()

            # Extraer contornos
            try:
                contour_points = get_contours(edges, kernel_contour)

                # Dibujar contorno
                if len(contour_points) > 0:
                    # Convertir a formato para cv2.drawContours
                    contour_for_draw = contour_points.reshape(-1, 1, 2).astype(np.int32)
                    cv2.drawContours(contour_display, [contour_for_draw], -1, (0, 255, 0), 2)

                    # Mostrar imagen con contorno
                    st.image(contour_display, width='stretch')

                    # Estadísticas del contorno
                    st.markdown("**📊 Estadísticas del Contorno**")
                    st.metric("Puntos", len(contour_points))
                    st.metric("Área", f"{cv2.contourArea(contour_for_draw):.0f} px²")

                    # Mostrar puntos como tabla
                    with st.expander("Ver puntos del contorno"):
                        st.dataframe(
                            contour_points,
                            column_config={
                                "0": st.column_config.NumberColumn("X", format="%d px"),
                                "1": st.column_config.NumberColumn("Y", format="%d px")
                            },
                            hide_index=True
                        )
                else:
                    st.warning("No se encontraron contornos válidos")
                    contour_points = np.array([])

            except Exception as contour_error:
                st.error(f"Error en extracción de contornos: {str(contour_error)}")
                contour_points = np.array([])

        with col_3d:
            st.markdown("**🔷 Modelo 3D Extrusionado**")

            if len(contour_points) >= 3:  # Mínimo para polígono
                try:
                    # Normalizar puntos
                    ordered_points = sort_contour_points(contour_points)
                    normalized_points = normalize_points(ordered_points, target_size=2.0)

                    # Extruir a 3D
                    vertices, faces = extrude_polygon(normalized_points, height=extrusion_height)

                    # Crear malla Plotly
                    mesh_data = create_plotly_mesh(
                        vertices, 
                        faces, 
                        color=mesh_color,
                        opacity=mesh_opacity
                    )

                    # Crear figura 3D
                    fig = go.Figure(data=[
                        go.Mesh3d(**mesh_data)
                    ])

                    # Configurar vista 3D
                    fig.update_layout(
                        scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y', 
                            zaxis_title='Z',
                            aspectmode='data',
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=1.0)
                            )
                        ),
                        margin=dict(l=0, r=0, b=0, t=0),
                        height=400
                    )

                    # Mostrar modelo 3D
                    st.plotly_chart(fig, width='stretch')

                    # Controles de vista 3D
                    view_col1, view_col2, view_col3 = st.columns([2, 1, 1])

                    with view_col1:
                        view_options = ["Perspectiva", "Superior", "Frontal", "Lateral"]
                        selected_view = st.selectbox("Ángulo de vista", view_options, key="view_select")

                    with view_col2:
                        if st.button("Aplicar", key="apply_view"):
                            views = {
                                "Perspectiva": dict(x=1.5, y=1.5, z=1.0),
                                "Superior": dict(x=0, y=0, z=2.0),
                                "Frontal": dict(x=0, y=2.0, z=0),
                                "Lateral": dict(x=2.0, y=0, z=0)
                            }
                            fig.update_layout(scene_camera=dict(eye=views[selected_view]))
                            st.plotly_chart(fig, width='stretch', key="updated_view")

                    with view_col3:
                        if st.button("Reiniciar", key="reset_view"):
                            fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)))
                            st.plotly_chart(fig, width='stretch', key="reset_view_chart")

                    # Estadísticas 3D
                    st.markdown("**📈 Estadísticas 3D**")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Vértices", len(vertices))
                    with stats_col2:
                        st.metric("Caras", len(faces))
                    with stats_col3:
                        st.metric("Altura", f"{extrusion_height:.2f}")

                except Exception as extrusion_error:
                    st.error(f"Error en modelado 3D: {str(extrusion_error)}")
                    st.info("Mostrando cubo de demostración...")
                    fig_demo = generar_cubo_3d()
                    st.plotly_chart(fig_demo, width='stretch')
            else:
                st.warning("Se necesitan al menos 3 puntos para crear un modelo 3D")
                st.info("Mostrando cubo de demostración...")
                fig_demo = generar_cubo_3d()
                st.plotly_chart(fig_demo, width='stretch')

    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {str(e)}")

else:
    st.info("👆 Carga una imagen para comenzar el procesamiento")