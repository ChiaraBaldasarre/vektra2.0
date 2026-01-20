"""
Módulo de procesamiento de imagenes para deteccion de bordes.
Implementa:
1. Conversion a Escala de Grises
2. Desenfoque Gaussiano
3. Deteccion de Bordes Canny
"""

import cv2


def convertir_a_escala_grises(image_array):
    """
    Convierte una imagen RGB a escala de grises.
    
    Args:
        image_array: Array de imagen en RGB
        
    Returns:
        Array de imagen en escala de grises
    """
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)


def aplicar_desenfoque_gaussiano(gray_image, kernel_size):
    """
    Aplica un desenfoque gaussiano para reducir ruido.
    
    Args:
        gray_image: Imagen en escala de grises
        kernel_size: Tamaño del kernel (debe ser impar)
        
    Returns:
        Imagen desenfocada
    """
    # Asegurar que el tamaño del kernel sea impar
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)


def detectar_bordes_canny(blurred_image, threshold1, threshold2):
    """
    Detecta bordes usando el algoritmo Canny.
    
    Args:
        blurred_image: Imagen desenfocada
        threshold1: Umbral inferior (0-200)
        threshold2: Umbral superior (variable-500)
        
    Returns:
        Imagen binaria con bordes detectados
    """
    return cv2.Canny(blurred_image, threshold1, threshold2)

