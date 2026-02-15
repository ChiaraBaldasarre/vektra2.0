"""
Módulo avanzado de extracción de contornos.
Implementa múltiples algoritmos para obtener contornos precisos.
"""

import cv2
import numpy as np


def get_contours(edges, kernel, config=None):
    """
    Extrae el contorno principal de una imagen de bordes.
    
    Args:
        edges: Imagen binaria de bordes (Canny)
        kernel: Kernel para operaciones morfológicas
        config: Dict con configuración avanzada:
            - simplify_method: 'adaptive', 'fixed', 'none'
            - epsilon_factor: float para simplificación
            - min_area: área mínima del contorno
            - smooth_contour: bool
            
    Returns:
        Array de puntos (n, 2) en el orden del contorno
    """
    if config is None:
        config = {}
    
    # 1. Operaciones morfológicas para mejorar contornos
    closed_edges = aplicar_morfologia(edges, kernel, config.get('morph_op', 'close'))
    
    # 2. Encontrar contornos
    contours, hierarchy = cv2.findContours(
        closed_edges,
        cv2.RETR_EXTERNAL,      # Solo contorno externo
        cv2.CHAIN_APPROX_NONE   # Mantener TODOS los puntos
    )
    
    if not contours:
        return np.array([])
    
    # 3. Filtrar contornos por área mínima
    min_area = config.get('min_area', 100)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not contours:
        return np.array([])
    
    # 4. Seleccionar contorno principal
    main_contour = max(contours, key=cv2.contourArea)
    
    # 5. Simplificación adaptativa
    simplify_method = config.get('simplify_method', 'adaptive')
    epsilon_factor = config.get('epsilon_factor', 0.001)
    
    if simplify_method == 'adaptive':
        approx = simplificar_contorno_adaptativo(main_contour, epsilon_factor)
    elif simplify_method == 'fixed':
        perimeter = cv2.arcLength(main_contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
    else:
        approx = main_contour
    
    # 6. Convertir a array 2D
    points = approx.reshape(-1, 2)
    
    return points


def aplicar_morfologia(edges, kernel, operation='close'):
    """
    Aplica operaciones morfológicas para mejorar los bordes.
    
    Args:
        edges: Imagen binaria
        kernel: Kernel morfológico
        operation: 'close', 'open', 'dilate', 'erode', 'gradient'
        
    Returns:
        Imagen procesada
    """
    if operation == 'close':
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    elif operation == 'open':
        return cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    elif operation == 'dilate':
        return cv2.dilate(edges, kernel, iterations=1)
    elif operation == 'erode':
        return cv2.erode(edges, kernel, iterations=1)
    elif operation == 'gradient':
        return cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    else:
        return edges


def simplificar_contorno_adaptativo(contour, base_epsilon=0.001):
    """
    Simplifica el contorno de forma adaptativa según su complejidad.
    
    Args:
        contour: Contorno de OpenCV
        base_epsilon: Factor base de simplificación
        
    Returns:
        Contorno simplificado
    """
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    
    # Calcular complejidad (relación perímetro/área)
    if area > 0:
        complexity = perimeter / np.sqrt(area)
    else:
        complexity = 1
    
    # Ajustar epsilon según complejidad
    # Formas complejas (alta relación P/A) necesitan menos simplificación
    if complexity > 20:  # Forma muy compleja
        epsilon = base_epsilon * 0.1 * perimeter
    elif complexity > 10:  # Forma compleja
        epsilon = base_epsilon * 0.5 * perimeter
    else:  # Forma simple
        epsilon = base_epsilon * perimeter
    
    return cv2.approxPolyDP(contour, epsilon, True)


def get_contours_multiples(edges, kernel, max_contours=5, min_area=100):
    """
    Extrae múltiples contornos de la imagen.
    
    Args:
        edges: Imagen binaria de bordes
        kernel: Kernel morfológico
        max_contours: Número máximo de contornos a devolver
        min_area: Área mínima para considerar un contorno
        
    Returns:
        Lista de arrays de puntos
    """
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(
        closed_edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    
    if not contours:
        return []
    
    # Filtrar por área y ordenar por tamaño
    valid_contours = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= min_area]
    valid_contours.sort(key=lambda x: x[1], reverse=True)
    
    # Tomar los N más grandes
    results = []
    for contour, _ in valid_contours[:max_contours]:
        approx = simplificar_contorno_adaptativo(contour)
        points = approx.reshape(-1, 2)
        results.append(points)
    
    return results


def get_contours_jerarquicos(edges, kernel):
    """
    Extrae contornos con su jerarquía (para formas con huecos).
    
    Args:
        edges: Imagen binaria de bordes
        kernel: Kernel morfológico
        
    Returns:
        Tuple (contornos_externos, contornos_internos)
    """
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, hierarchy = cv2.findContours(
        closed_edges,
        cv2.RETR_TREE,  # Jerarquía completa
        cv2.CHAIN_APPROX_NONE
    )
    
    if not contours or hierarchy is None:
        return [], []
    
    hierarchy = hierarchy[0]
    
    externos = []
    internos = []
    
    for i, (contour, h) in enumerate(zip(contours, hierarchy)):
        # h = [next, prev, first_child, parent]
        parent = h[3]
        
        if parent == -1:  # Sin padre = contorno externo
            approx = simplificar_contorno_adaptativo(contour)
            externos.append(approx.reshape(-1, 2))
        else:  # Con padre = contorno interno (hueco)
            approx = simplificar_contorno_adaptativo(contour)
            internos.append(approx.reshape(-1, 2))
    
    return externos, internos


def refinar_contorno_subpixel(gray_image, contour, win_size=5):
    """
    Refina las posiciones del contorno a nivel subpíxel.
    Mejora significativa en la precisión.
    
    Args:
        gray_image: Imagen en escala de grises
        contour: Contorno a refinar
        win_size: Tamaño de la ventana de refinamiento
        
    Returns:
        Contorno refinado
    """
    # Convertir contorno a puntos flotantes
    points = contour.reshape(-1, 2).astype(np.float32)
    
    # Criterio de terminación
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    try:
        # Refinar usando cornerSubPix
        refined = cv2.cornerSubPix(
            gray_image, 
            points, 
            (win_size, win_size), 
            (-1, -1), 
            criteria
        )
        return refined
    except:
        return points


def calcular_curvatura(points, window=5):
    """
    Calcula la curvatura en cada punto del contorno.
    Útil para identificar puntos importantes.
    
    Args:
        points: Array de puntos (n, 2)
        window: Tamaño de la ventana para calcular curvatura
        
    Returns:
        Array de curvaturas
    """
    n = len(points)
    curvatures = np.zeros(n)
    
    for i in range(n):
        # Puntos vecinos
        prev_idx = (i - window) % n
        next_idx = (i + window) % n
        
        p_prev = points[prev_idx]
        p_curr = points[i]
        p_next = points[next_idx]
        
        # Vectores
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        # Ángulo entre vectores
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        
        angle = np.arctan2(cross, dot)
        curvatures[i] = abs(angle)
    
    return curvatures


def detectar_puntos_criticos(points, curvature_threshold=0.3):
    """
    Detecta puntos críticos (esquinas, curvas pronunciadas) del contorno.
    
    Args:
        points: Array de puntos
        curvature_threshold: Umbral de curvatura para considerar crítico
        
    Returns:
        Índices de puntos críticos
    """
    curvatures = calcular_curvatura(points)
    
    # Encontrar máximos locales de curvatura
    critical_indices = []
    
    for i in range(len(curvatures)):
        if curvatures[i] > curvature_threshold:
            # Verificar si es máximo local
            prev_idx = (i - 1) % len(curvatures)
            next_idx = (i + 1) % len(curvatures)
            
            if curvatures[i] >= curvatures[prev_idx] and curvatures[i] >= curvatures[next_idx]:
                critical_indices.append(i)
    
    return critical_indices


def remuestrear_contorno(points, num_points):
    """
    Remuestrea el contorno para tener un número uniforme de puntos.
    Mantiene mejor la forma que simplemente tomar cada N puntos.
    
    Args:
        points: Array de puntos original
        num_points: Número deseado de puntos
        
    Returns:
        Array de puntos remuestreados
    """
    if len(points) < 3:
        return points
    
    # Calcular longitud acumulada
    diff = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))
    cumulative = np.zeros(len(points))
    cumulative[1:] = np.cumsum(distances)
    
    total_length = cumulative[-1]
    
    if total_length == 0:
        return points[:num_points] if len(points) >= num_points else points
    
    # Puntos uniformemente espaciados
    target_distances = np.linspace(0, total_length, num_points, endpoint=False)
    
    # Interpolar
    new_points = np.zeros((num_points, 2))
    
    for i, target in enumerate(target_distances):
        # Encontrar segmento
        idx = np.searchsorted(cumulative, target, side='right') - 1
        idx = max(0, min(idx, len(points) - 2))
        
        # Interpolar en el segmento
        segment_start = cumulative[idx]
        segment_end = cumulative[idx + 1]
        
        if segment_end > segment_start:
            t = (target - segment_start) / (segment_end - segment_start)
        else:
            t = 0
        
        new_points[i] = points[idx] + t * (points[idx + 1] - points[idx])
    
    return new_points


def suavizar_contorno_media_movil(points, window=3):
    """
    Suaviza el contorno usando media móvil.
    Método simple pero efectivo para eliminar ruido.
    
    Args:
        points: Array de puntos
        window: Tamaño de la ventana
        
    Returns:
        Puntos suavizados
    """
    n = len(points)
    smoothed = np.zeros_like(points, dtype=np.float64)
    
    for i in range(n):
        # Calcular media de puntos vecinos
        indices = [(i + j) % n for j in range(-window//2, window//2 + 1)]
        smoothed[i] = np.mean(points[indices], axis=0)
    
    return smoothed


# ===== FUNCIÓN DE DEBUG =====

def debug_contour_order(points, image_edges, kernel):
    """
    Función para visualizar el orden de puntos (solo para debug).
    """
    if len(points) == 0:
        return None
    
    if len(image_edges.shape) == 2:
        display_img = cv2.cvtColor(image_edges, cv2.COLOR_GRAY2BGR)
    else:
        display_img = image_edges.copy()
    
    for i, (x, y) in enumerate(points):
        cv2.circle(display_img, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.putText(display_img, str(i), (int(x)+5, int(y)-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    for i in range(len(points) - 1):
        pt1 = tuple(points[i].astype(int))
        pt2 = tuple(points[i+1].astype(int))
        cv2.line(display_img, pt1, pt2, (0, 255, 0), 1)
    
    if len(points) > 2:
        pt1 = tuple(points[-1].astype(int))
        pt2 = tuple(points[0].astype(int))
        cv2.line(display_img, pt1, pt2, (0, 255, 0), 1)
    
    return display_img