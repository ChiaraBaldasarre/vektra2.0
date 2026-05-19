"""
Módulo de procesamiento de imagenes para deteccion de bordes.
Implementa técnicas avanzadas de preprocesamiento y detección.
"""

import cv2
import numpy as np

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


# ============ FUNCIONES AVANZADAS DE PREPROCESAMIENTO ============

def aplicar_clahe(gray_image, clip_limit=2.0, tile_size=8):
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Mejora el contraste local para detectar mejor los bordes.
    
    Args:
        gray_image: Imagen en escala de grises
        clip_limit: Límite de contraste (2.0-4.0 típico)
        tile_size: Tamaño de la cuadrícula
        
    Returns:
        Imagen con contraste mejorado
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray_image)


def aplicar_filtro_bilateral(gray_image, d=9, sigma_color=75, sigma_space=75):
    """
    Aplica filtro bilateral: reduce ruido preservando bordes.
    Mejor que Gaussiano para mantener bordes nítidos.
    
    Args:
        gray_image: Imagen en escala de grises
        d: Diámetro del vecindario
        sigma_color: Filtro sigma en espacio de color
        sigma_space: Filtro sigma en espacio de coordenadas
        
    Returns:
        Imagen filtrada
    """
    return cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space)


def aplicar_desenfoque_mediana(gray_image, kernel_size=5):
    """
    Aplica desenfoque de mediana.
    Excelente para eliminar ruido "sal y pimienta".
    
    Args:
        gray_image: Imagen en escala de grises
        kernel_size: Tamaño del kernel (impar)
        
    Returns:
        Imagen filtrada
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(gray_image, kernel_size)


def detectar_umbrales_automaticos(gray_image):
    """
    Calcula umbrales óptimos para Canny basados en estadísticas de la imagen.
    Usa el método de Otsu y percentiles.
    
    Args:
        gray_image: Imagen en escala de grises
        
    Returns:
        Tuple (threshold1, threshold2) óptimos
    """
    # Método 1: Basado en mediana (regla de 0.33)
    median = np.median(gray_image)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    
    return lower, upper


def detectar_umbrales_otsu(gray_image):
    """
    Usa el método de Otsu para encontrar umbral óptimo.
    
    Args:
        gray_image: Imagen en escala de grises
        
    Returns:
        Tuple (threshold1, threshold2) basados en Otsu
    """
    # Aplicar Otsu
    otsu_thresh, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Usar Otsu como base para Canny
    lower = int(otsu_thresh * 0.5)
    upper = int(otsu_thresh)
    
    return lower, upper


def preprocesamiento_avanzado(image_array, config=None):
    """
    Pipeline de preprocesamiento avanzado configurable.
    
    Args:
        image_array: Imagen RGB
        config: Dict con configuración:
            - denoise_method: 'gaussian', 'bilateral', 'median', 'nlmeans'
            - enhance_contrast: bool
            - clahe_clip: float
            - auto_threshold: bool
            - kernel_size: int
            
    Returns:
        Dict con resultados del pipeline
    """
    if config is None:
        config = {}
    
    results = {}
    
    # 1. Convertir a escala de grises
    gray = convertir_a_escala_grises(image_array)
    results['gray'] = gray
    
    # 2. Mejora de contraste (opcional)
    if config.get('enhance_contrast', True):
        clip_limit = config.get('clahe_clip', 2.0)
        gray = aplicar_clahe(gray, clip_limit=clip_limit)
        results['enhanced'] = gray
    
    # 3. Reducción de ruido
    denoise_method = config.get('denoise_method', 'bilateral')
    kernel_size = config.get('kernel_size', 5)
    
    if denoise_method == 'gaussian':
        denoised = aplicar_desenfoque_gaussiano(gray, kernel_size)
    elif denoise_method == 'bilateral':
        denoised = aplicar_filtro_bilateral(gray)
    elif denoise_method == 'median':
        denoised = aplicar_desenfoque_mediana(gray, kernel_size)
    elif denoise_method == 'nlmeans':
        denoised = aplicar_nlmeans(gray)
    else:
        denoised = gray
    
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
    results['edges'] = edges
    
    return results


def aplicar_nlmeans(gray_image, h=10, template_window=7, search_window=21):
    """
    Aplica Non-Local Means Denoising.
    El mejor método para preservar detalles mientras elimina ruido.
    
    Args:
        gray_image: Imagen en escala de grises
        h: Fuerza del filtro (mayor = más suavizado)
        template_window: Tamaño del template
        search_window: Tamaño del área de búsqueda
        
    Returns:
        Imagen denoised
    """
    return cv2.fastNlMeansDenoising(gray_image, None, h, template_window, search_window)


def detectar_bordes_multi_escala(gray_image, scales=[1.0, 0.5, 0.25]):
    """
    Detecta bordes en múltiples escalas y combina resultados.
    Mejora la detección en imágenes con detalles de diferentes tamaños.
    
    Args:
        gray_image: Imagen en escala de grises
        scales: Lista de escalas a usar
        
    Returns:
        Imagen de bordes combinada
    """
    h, w = gray_image.shape
    combined = np.zeros_like(gray_image)
    
    for scale in scales:
        # Redimensionar
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h < 10 or new_w < 10:
            continue
            
        scaled = cv2.resize(gray_image, (new_w, new_h))
        
        # Detectar bordes
        t1, t2 = detectar_umbrales_automaticos(scaled)
        edges = cv2.Canny(scaled, t1, t2)
        
        # Redimensionar de vuelta
        edges_full = cv2.resize(edges, (w, h))
        
        # Combinar (OR lógico)
        combined = cv2.bitwise_or(combined, edges_full)
    
    return combined


def mejorar_bordes_morfologicos(edges, kernel_size=3, iterations=1):
    """
    Mejora los bordes usando operaciones morfológicas.
    
    Args:
        edges: Imagen binaria de bordes
        kernel_size: Tamaño del kernel morfológico
        iterations: Número de iteraciones
        
    Returns:
        Bordes mejorados
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Dilatación para conectar bordes rotos
    dilated = cv2.dilate(edges, kernel, iterations=iterations)
    
    # Erosión para adelgazar
    eroded = cv2.erode(dilated, kernel, iterations=iterations)
    
    # Cierre para rellenar huecos pequeños
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
    
    return closed


def binarizar_adaptativo(gray_image, block_size=11, c=2):
    """
    Binarización adaptativa para imágenes con iluminación desigual.
    
    Args:
        gray_image: Imagen en escala de grises
        block_size: Tamaño del bloque (impar)
        c: Constante a restar
        
    Returns:
        Imagen binarizada
    """
    if block_size % 2 == 0:
        block_size += 1
    
    return cv2.adaptiveThreshold(
        gray_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )


def detectar_esquinas_harris(gray_image, block_size=2, ksize=3, k=0.04):
    """
    Detecta esquinas usando el detector de Harris.
    Útil para encontrar puntos clave del contorno.
    
    Args:
        gray_image: Imagen en escala de grises
        block_size: Tamaño del vecindario
        ksize: Apertura del operador Sobel
        k: Parámetro libre de Harris
        
    Returns:
        Imagen con respuesta de esquinas
    """
    gray_float = np.float32(gray_image)
    corners = cv2.cornerHarris(gray_float, block_size, ksize, k)
    
    # Dilatar para marcar mejor las esquinas
    corners = cv2.dilate(corners, None)
    
    return corners


def segmentar_grabcut(image_rgb, rect=None, iterations=5):
    """
    Segmenta el objeto principal usando GrabCut.
    Útil para separar el objeto del fondo antes de detectar bordes.
    
    Args:
        image_rgb: Imagen RGB
        rect: Rectángulo inicial (x, y, w, h) o None para auto
        iterations: Número de iteraciones
        
    Returns:
        Máscara binaria del objeto
    """
    h, w = image_rgb.shape[:2]
    
    # Si no se proporciona rect, usar el 90% central
    if rect is None:
        margin_x, margin_y = int(w * 0.05), int(h * 0.05)
        rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
    
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(image_rgb, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        
        # Crear máscara binaria (0, 2 = fondo, 1, 3 = primer plano)
        mask_binary = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        
        return mask_binary
    except:
        # Si falla, devolver máscara completa
        return np.ones((h, w), np.uint8) * 255
