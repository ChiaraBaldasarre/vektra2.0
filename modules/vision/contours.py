"""
Módulo avanzado de extracción de contornos.
Implementa múltiples algoritmos para obtener contornos precisos.
"""

import cv2
import numpy as np

# --- 1. FUNCIONES NUEVAS DE LA HISTORIA DE USUARIO ---

def asegurar_contorno_cerrado(contour, tolerancia=2.0):
    cnt = contour.reshape(-1, 2)
    dist = np.linalg.norm(cnt[0] - cnt[-1])
    if dist > tolerancia:
        return np.vstack([cnt, cnt[0]])
    return cnt

def simplificar_contorno_optimo(contour, factor=0.002):
    epsilon = factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def corregir_orientacion(contour, sentido_horario=True):
    area = cv2.contourArea(contour, oriented=True)
    if (sentido_horario and area > 0) or (not sentido_horario and area < 0):
        return contour[::-1]
    return contour

def procesar_contorno_robusto(contour):
    """Pipeline maestro de limpieza"""
    c = simplificar_contorno_optimo(contour)
    c = asegurar_contorno_cerrado(c)
    c = corregir_orientacion(c)
    return c.astype(np.float32)

def combinar_contornos_ordenados(contours, method='connected'):
    """
    Combina múltiples contornos en uno solo de forma ordenada.
    
    Args:
        contours: Lista de contornos
        method: 'connected' (une por puntos cercanos) o 'concat' (simple concatenación)
        
    Returns:
        Contorno combinado como array numpy
    """
    if len(contours) == 1:
        return contours[0]
    
    if method == 'concat':
        # Simple concatenación
        all_points = np.vstack(contours)
        return all_points
    
    # Método 'connected': conectar contornos por puntos más cercanos
    # Ordenar contornos por área (del más grande al más pequeño)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    combined = sorted_contours[0].reshape(-1, 2).tolist()
    remaining = [c.reshape(-1, 2) for c in sorted_contours[1:]]
    
    while remaining:
        # Encontrar el contorno más cercano al final del combinado actual
        last_point = np.array(combined[-1])
        
        min_dist = float('inf')
        best_idx = 0
        best_start_idx = 0
        
        for i, contour in enumerate(remaining):
            # Buscar el punto más cercano en este contorno
            distances = np.sqrt(np.sum((contour - last_point) ** 2, axis=1))
            min_contour_dist = np.min(distances)
            
            if min_contour_dist < min_dist:
                min_dist = min_contour_dist
                best_idx = i
                best_start_idx = np.argmin(distances)
        
        # Agregar el contorno más cercano, empezando desde el punto más cercano
        next_contour = remaining.pop(best_idx)
        
        # Reordenar el contorno para empezar desde el punto más cercano
        reordered = np.roll(next_contour, -best_start_idx, axis=0)
        combined.extend(reordered.tolist())
    
    return np.array(combined).reshape(-1, 1, 2).astype(np.int32)


def cerrar_bordes_adaptativo(edges, max_iterations=10, kernel_size=3):
    """
    Cierra los bordes de forma adaptativa usando dilatación iterativa.
    Esto ayuda a conectar bordes que están ligeramente separados.
    
    Args:
        edges: Imagen binaria de bordes
        max_iterations: Máximo de iteraciones de dilatación
        kernel_size: Tamaño del kernel de dilatación
        
    Returns:
        Imagen con bordes cerrados
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = edges.copy()
    
    for i in range(max_iterations):
        # Dilatar para conectar bordes cercanos
        dilated = cv2.dilate(closed, kernel, iterations=1)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Verificar si el contorno más grande tiene un área razonable
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            perimeter = cv2.arcLength(max_contour, True)
            
            # Si el perímetro es razonable respecto al área, tenemos un contorno cerrado
            if perimeter > 0 and area > 0:
                # Circularidad: 4*pi*area / perimeter^2 - más cercano a 1 = más cerrado
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.01:  # Umbral bajo para aceptar formas variadas
                    closed = dilated
                    break
        
        closed = dilated
    
    return closed


def extraer_contorno_preciso(edges, max_iterations=10, kernel_size=3):
    """
    Extrae el contorno siguiendo los bordes originales con mayor precisión.
    Usa dilatación para cerrar huecos, luego erosiona para compensar.
    
    Args:
        edges: Imagen binaria de bordes
        max_iterations: Máximo de iteraciones de dilatación
        kernel_size: Tamaño del kernel
        
    Returns:
        Contorno que sigue los bordes originales
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Paso 1: Dilatar para conectar bordes
    dilated = cv2.dilate(edges, kernel, iterations=max_iterations)
    
    # Paso 2: Rellenar huecos internos usando flood fill
    h, w = dilated.shape
    filled = dilated.copy()
    
    # Crear máscara para flood fill (debe ser 2 pixels más grande)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Flood fill desde las esquinas (asumiendo fondo negro)
    cv2.floodFill(filled, mask, (0, 0), 255)
    
    # Invertir para obtener el área interior
    filled_inv = cv2.bitwise_not(filled)
    
    # Combinar con la imagen dilatada original
    filled_shape = dilated | filled_inv
    
    # Paso 3: Erosionar para compensar la dilatación
    # Erosionamos un poco menos de lo que dilatamos para mantener contorno cerrado
    erosion_iterations = max(1, max_iterations - 2)
    eroded = cv2.erode(filled_shape, kernel, iterations=erosion_iterations)
    
    # Paso 4: Aplicar cierre morfológico final para suavizar
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
    
    # Paso 5: Extraer solo el borde exterior (esqueletizar)
    # Erosionar una vez más y restar para obtener solo el contorno
    inner = cv2.erode(closed, kernel, iterations=1)
    contour_edge = closed - inner
    
    return closed, contour_edge


def ajustar_contorno_a_bordes(contour_points, edges, search_radius=5):
    """
    Ajusta cada punto del contorno al borde más cercano en la imagen de bordes.
    
    Args:
        contour_points: Array de puntos del contorno (N, 2)
        edges: Imagen binaria de bordes originales
        search_radius: Radio de búsqueda para el borde más cercano
        
    Returns:
        Contorno ajustado
    """
    if len(contour_points) == 0:
        return contour_points
    
    adjusted = []
    h, w = edges.shape
    
    for point in contour_points:
        x, y = int(point[0]), int(point[1])
        
        # Definir región de búsqueda
        x_min = max(0, x - search_radius)
        x_max = min(w, x + search_radius + 1)
        y_min = max(0, y - search_radius)
        y_max = min(h, y + search_radius + 1)
        
        # Buscar píxeles de borde en la región
        region = edges[y_min:y_max, x_min:x_max]
        edge_pixels = np.where(region > 0)
        
        if len(edge_pixels[0]) > 0:
            # Encontrar el píxel de borde más cercano
            edge_coords = np.column_stack((edge_pixels[1] + x_min, edge_pixels[0] + y_min))
            distances = np.sqrt(np.sum((edge_coords - np.array([x, y])) ** 2, axis=1))
            nearest_idx = np.argmin(distances)
            adjusted.append(edge_coords[nearest_idx])
        else:
            # Si no hay borde cercano, mantener el punto original
            adjusted.append([x, y])
    
    return np.array(adjusted, dtype=np.float32)


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
            - use_tree: bool, usar RETR_TREE para jerarquía completa
            - trace_edges: bool, trazar bordes directamente en vez de buscar contornos cerrados
            - adjust_to_edges: bool, ajustar puntos al borde más cercano
            - search_radius: int, radio de búsqueda para ajuste
            
    Returns:
        Array de puntos (n, 2) en el orden del contorno
    """
    if config is None:
        config = {}
    
    # Guardar bordes originales para ajuste posterior
    original_edges = edges.copy()
    
    # Opción para trazar bordes directamente (mejor para dibujos lineales)
    trace_edges = config.get('trace_edges', False)
    
    if trace_edges:
        # Usar extracción precisa que compensa la dilatación
        closed_edges, _ = extraer_contorno_preciso(
            edges, 
            max_iterations=config.get('close_iterations', 5),
            kernel_size=config.get('close_kernel', 3)
        )
    else:
        # 1. Operaciones morfológicas para mejorar contornos
        closed_edges = aplicar_morfologia(edges, kernel, config.get('morph_op', 'close'))
    
    # 2. Encontrar contornos - usar RETR_TREE para obtener jerarquía completa si se requiere
    use_tree = config.get('use_tree', False)
    retrieval_mode = cv2.RETR_TREE if use_tree else cv2.RETR_EXTERNAL
    
    contours, hierarchy = cv2.findContours(
        closed_edges,
        retrieval_mode,
        cv2.CHAIN_APPROX_NONE   # Mantener TODOS los puntos
    )
    
    if not contours:
        return np.array([])
    
    # 3. Filtrar contornos por área mínima
    min_area = config.get('min_area', 100)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not contours:
        return np.array([])
    
    # 4. Seleccionar contorno(s) según configuración
    multi_contour_mode = config.get('multi_contour_mode', 'single')
    
    if multi_contour_mode == 'union':
        # Unir todos los contornos de forma conectada (mantiene formas individuales)
        main_contour = combinar_contornos_ordenados(contours, method='connected')
    elif multi_contour_mode == 'hull':
        # Crear el convex hull que envuelve todos los contornos
        all_points = np.vstack(contours)
        main_contour = cv2.convexHull(all_points)
    else:
        # 'single' - Solo el contorno más grande
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
    
    # 7. Ajustar puntos al borde original más cercano (NUEVO)
    adjust_to_edges = config.get('adjust_to_edges', True)  # Activado por defecto
    if adjust_to_edges and trace_edges:
        search_radius = config.get('search_radius', 10)
        points = ajustar_contorno_a_bordes(points, original_edges, search_radius)
    
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