"""
Módulo de procesamiento de imagenes para deteccion de bordes.
Implementa técnicas avanzadas de preprocesamiento y detección.

Contrato de datos:
    - Entrada estándar:  np.ndarray uint8, shape (H, W) escala de grises
                         o (H, W, 3) RGB según cada función.
    - Bordes de salida:  np.ndarray uint8, shape (H, W), valores {0, 255}.
    - Umbrales Canny:    int en el rango de magnitud de gradiente [0, 1400]
                         (NOT intensidad de píxel). Se calculan sobre la
                         imagen de gradiente Sobel, no sobre intensidades.

Kernels centralizados (únicos defaults por propósito):
    KERNEL_DENOISE  = 5   — desenfoque gaussiano / mediana
    KERNEL_MORPH    = 3   — cierre morfológico único aplicado en get_contours
"""

import cv2
import numpy as np

KERNEL_DENOISE: int = 5
KERNEL_MORPH: int = 3


def convertir_a_escala_grises(image_array: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen RGB a escala de grises.

    Args:
        image_array: np.ndarray uint8 (H, W, 3) RGB

    Returns:
        np.ndarray uint8 (H, W)
    """
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)


def aplicar_desenfoque_gaussiano(gray_image: np.ndarray, kernel_size: int = KERNEL_DENOISE) -> np.ndarray:
    """
    Aplica un desenfoque gaussiano para reducir ruido.

    Args:
        gray_image: np.ndarray uint8 (H, W)
        kernel_size: int impar — default KERNEL_DENOISE

    Returns:
        np.ndarray uint8 (H, W)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)


def detectar_bordes_canny(blurred_image: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    """
    Detecta bordes usando el algoritmo Canny.

    Args:
        blurred_image: np.ndarray uint8 (H, W) ya desenfocada
        threshold1: umbral inferior de gradiente (0-1400)
        threshold2: umbral superior de gradiente (0-1400)

    Returns:
        np.ndarray uint8 (H, W) con valores {0, 255}
    """
    return cv2.Canny(blurred_image, threshold1, threshold2)


def aplicar_clahe(gray_image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Mejora el contraste local para detectar mejor los bordes.

    Args:
        gray_image: np.ndarray uint8 (H, W)
        clip_limit: límite de contraste, típico 2.0-4.0
        tile_size: tamaño de la cuadrícula

    Returns:
        np.ndarray uint8 (H, W) con contraste mejorado
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray_image)


def aplicar_filtro_bilateral(gray_image: np.ndarray, d: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    """
    Aplica filtro bilateral: reduce ruido preservando bordes.

    Args:
        gray_image: np.ndarray uint8 (H, W)
        d: diámetro del vecindario
        sigma_color: sigma en espacio de color
        sigma_space: sigma en espacio de coordenadas

    Returns:
        np.ndarray uint8 (H, W)
    """
    return cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space)


def aplicar_desenfoque_mediana(gray_image: np.ndarray, kernel_size: int = KERNEL_DENOISE) -> np.ndarray:
    """
    Aplica desenfoque de mediana. Excelente para ruido sal-y-pimienta.

    Args:
        gray_image: np.ndarray uint8 (H, W)
        kernel_size: int impar — default KERNEL_DENOISE

    Returns:
        np.ndarray uint8 (H, W)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(gray_image, kernel_size)


def aplicar_nlmeans(gray_image: np.ndarray, h: int = 10, template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """
    Aplica Non-Local Means Denoising.
    Mejor preservación de detalles a costa de mayor tiempo de cómputo.

    Args:
        gray_image: np.ndarray uint8 (H, W)
        h: fuerza del filtro
        template_window: tamaño del template
        search_window: tamaño del área de búsqueda

    Returns:
        np.ndarray uint8 (H, W)
    """
    return cv2.fastNlMeansDenoising(gray_image, None, h, template_window, search_window)


def detectar_umbrales_automaticos(gray_image: np.ndarray) -> tuple[int, int]:
    """
    Calcula umbrales óptimos para Canny sobre la magnitud del gradiente Sobel.

    Trabaja sobre el espacio correcto (gradientes, rango 0-1400) en lugar de
    intensidades de píxel (rango 0-255), evitando el error conceptual de
    aplicar estadísticas de intensidad a umbrales de gradiente.

    Args:
        gray_image: np.ndarray uint8 (H, W)

    Returns:
        Tuple[int, int] (threshold1, threshold2) en rango de gradiente
    """
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    median = float(np.median(magnitude[magnitude > 0])) if np.any(magnitude > 0) else 1.0
    sigma = 0.33
    lower = int(max(0.0, (1.0 - sigma) * median))
    upper = int(min(1400.0, (1.0 + sigma) * median))

    return lower, upper


def preprocesamiento_avanzado(image_array: np.ndarray, config: dict | None = None) -> dict:
    """
    Pipeline de preprocesamiento avanzado configurable.
    Integra todos los pasos en un único punto de entrada para el pipeline principal.

    Args:
        image_array: np.ndarray uint8 (H, W, 3) RGB
        config: dict con claves opcionales:
            denoise_method  : 'gaussian' | 'bilateral' | 'median' | 'nlmeans'
            enhance_contrast: bool  — activa CLAHE
            clahe_clip      : float — clip_limit para CLAHE
            auto_threshold  : bool  — usa detectar_umbrales_automaticos
            threshold1      : int   — umbral manual inferior
            threshold2      : int   — umbral manual superior
            kernel_size     : int   — kernel para denoise (default KERNEL_DENOISE)

    Returns:
        dict con claves:
            'gray'       : np.ndarray uint8 (H, W)
            'enhanced'   : np.ndarray uint8 (H, W)  — solo si enhance_contrast=True
            'denoised'   : np.ndarray uint8 (H, W)
            'thresholds' : Tuple[int, int]
            'edges'      : np.ndarray uint8 (H, W) {0, 255}
    """
    if config is None:
        config = {}

    results: dict = {}

    gray = convertir_a_escala_grises(image_array)
    results['gray'] = gray

    if config.get('enhance_contrast', True):
        gray = aplicar_clahe(gray, clip_limit=config.get('clahe_clip', 2.0))
        results['enhanced'] = gray

    denoise_method = config.get('denoise_method', 'bilateral')
    kernel_size = config.get('kernel_size', KERNEL_DENOISE)

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

    if config.get('auto_threshold', True):
        t1, t2 = detectar_umbrales_automaticos(denoised)
    else:
        t1 = config.get('threshold1', 50)
        t2 = config.get('threshold2', 150)

    results['thresholds'] = (t1, t2)
    results['edges'] = detectar_bordes_canny(denoised, t1, t2)

    return results


def detectar_bordes_multi_escala(gray_image: np.ndarray, scales: list[float] | None = None) -> np.ndarray:
    """
    Detecta bordes en múltiples escalas y combina resultados con OR lógico.
    Mejora la detección en imágenes con detalles de tamaños variados.

    Los umbrales se calculan sobre la magnitud del gradiente Sobel en cada
    escala para mantener consistencia con el espacio de gradiente.

    Args:
        gray_image: np.ndarray uint8 (H, W)
        scales: lista de factores de escala — default [1.0, 0.5, 0.25]

    Returns:
        np.ndarray uint8 (H, W) con valores {0, 255}
    """
    if scales is None:
        scales = [1.0, 0.5, 0.25]

    h, w = gray_image.shape
    combined = np.zeros_like(gray_image)

    for scale in scales:
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h < 10 or new_w < 10:
            continue

        scaled = cv2.resize(gray_image, (new_w, new_h))
        t1, t2 = detectar_umbrales_automaticos(scaled)
        edges = cv2.Canny(scaled, t1, t2)
        edges_full = cv2.resize(edges, (w, h))
        combined = cv2.bitwise_or(combined, edges_full)

    return combined


def binarizar_adaptativo(gray_image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    Binarización adaptativa para imágenes con iluminación desigual.

    Args:
        gray_image: np.ndarray uint8 (H, W)
        block_size: tamaño del bloque (impar)
        c: constante a restar al umbral calculado

    Returns:
        np.ndarray uint8 (H, W) con valores {0, 255}
    """
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )


def segmentar_grabcut(image_rgb: np.ndarray, rect: tuple | None = None, iterations: int = 5) -> np.ndarray:
    """
    Segmenta el objeto principal usando GrabCut.
    para separar el objeto del fondo antes de detectar bordes.

    Args:
        image_rgb: np.ndarray uint8 (H, W, 3) RGB
        rect: (x, y, w, h) — None usa el 90% central
        iterations: número de iteraciones GrabCut

    Returns:
        np.ndarray uint8 (H, W) máscara binaria {0, 255}
    """
    h, w = image_rgb.shape[:2]

    if rect is None:
        margin_x, margin_y = int(w * 0.05), int(h * 0.05)
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image_rgb, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        return np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    except Exception:
        return np.ones((h, w), np.uint8) * 255