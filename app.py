import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from modules.visualization import generar_cubo_3d
from modules.image_processing import (
    convertir_a_escala_grises,
    aplicar_desenfoque_gaussiano,
    detectar_bordes_canny
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
            st.image(original, use_container_width=True)
        
        with col2:
            st.markdown("**🔍 Escala de Grises**")
            st.image(gray, use_container_width=True, clamp=True)
        
        with col3:
            st.markdown("**🌫️ Desenfoque Gaussiano**")
            st.image(blurred, use_container_width=True, clamp=True)
        
        with col4:
            st.markdown("**✨ Bordes Detectados (Canny)**")
            st.image(edges, use_container_width=True, clamp=True)
        
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {str(e)}")

else:
    st.info("👆 Carga una imagen para comenzar el procesamiento")
