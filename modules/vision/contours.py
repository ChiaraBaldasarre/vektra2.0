"""
Módulo avanzado de extracción de contornos.
Implementa múltiples algoritmos para obtener contornos precisos.

Contrato de datos:
    - edges entrada : np.ndarray uint8 (H, W), valores {0, 255} (salida de Canny)
    - kernel entrada: np.ndarray uint8 cuadrado — construido por el llamador con
                      KERNEL_MORPH de image_processing para garantizar un único
                      default consistente entre módulos.
    - Salida puntos : np.ndarray float32 (N, 2), coordenadas (x, y) en píxeles.

Morfología única:
    get_contours aplica exactamente UNA operación morfológica (cierre) sobre
    edges antes de buscar contornos. No existe ningún cierre previo ni posterior
    fuera de esta función, eliminando el doble cierre que desplazaba los bordes.
"""

import cv2
import numpy as np


def combinar_contornos_ordenados(contours: list, method: str = 'connected') -> np.ndarray:
    """
    Combina múltiples contornos en uno solo de forma ordenada.

    Args:
        contours: lista de contornos OpenCV
        method  : 'connected' — une por puntos más cercanos
                  'concat'    — concatenación directa

    Returns:
        np.ndarray int32 (N, 1, 2) contorno combinado
    """
    if len(contours) == 1:
        return contours[0]

    if method == 'concat':
        return np.vstack(contours)

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    combined = sorted_contours[0].reshape(-1, 2).tolist()
    remaining = [c.reshape(-1, 2) for c in sorted_contours[1:]]

    while remaining:
        last_point = np.array(combined[-1])
        min_dist = float('inf')
        best_idx = 0
        best_start_idx = 0

        for i, contour in enumerate(remaining):
            distances = np.sqrt(np.sum((contour - last_point) ** 2, axis=1))
            min_contour_dist = float(np.min(distances))
            if min_contour_dist < min_dist:
                min_dist = min_contour_dist
                best_idx = i
                best_start_idx = int(np.argmin(distances))

        next_contour = remaining.pop(best_idx)
        reordered = np.roll(next_contour, -best_start_idx, axis=0)
        combined.extend(reordered.tolist())

    return np.array(combined).reshape(-1, 1, 2).astype(np.int32)


def cerrar_bordes_adaptativo(edges: np.ndarray, max_iterations: int = 10, kernel_size: int = 3) -> np.ndarray:
    """
    Cierra bordes de forma adaptativa mediante dilatación iterativa.
    Conecta bordes ligeramente separados sin introducir operaciones morfológicas
    adicionales sobre la imagen ya procesada por get_contours.

    Args:
        edges         : np.ndarray uint8 (H, W) {0, 255}
        max_iterations: máximo de iteraciones de dilatación
        kernel_size   : tamaño del kernel de dilatación

    Returns:
        np.ndarray uint8 (H, W) con bordes cerrados
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = edges.copy()

    for _ in range(max_iterations):
        dilated = cv2.dilate(closed, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            perimeter = cv2.arcLength(max_contour, True)
            if perimeter > 0 and area > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.01:
                    closed = dilated
                    break

        closed = dilated

    return closed


def extraer_contorno_preciso(edges: np.ndarray, max_iterations: int = 10, kernel_size: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae el contorno siguiendo los bordes originales con mayor precisión.
    Dilata para cerrar huecos, erosiona para compensar y aplica UN solo cierre
    morfológico final antes de esqueletizar.

    Args:
        edges         : np.ndarray uint8 (H, W) {0, 255}
        max_iterations: máximo de iteraciones de dilatación
        kernel_size   : tamaño del kernel

    Returns:
        Tuple[np.ndarray, np.ndarray] — (forma_cerrada, borde_exterior) ambos uint8 (H, W)
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated = cv2.dilate(edges, kernel, iterations=max_iterations)

    h, w = dilated.shape
    filled = dilated.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    filled_shape = dilated | filled_inv

    erosion_iterations = max(1, max_iterations - 2)
    eroded = cv2.erode(filled_shape, kernel, iterations=erosion_iterations)

    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)

    inner = cv2.erode(closed, kernel, iterations=1)
    contour_edge = closed - inner

    return closed, contour_edge


def ajustar_contorno_a_bordes(contour_points: np.ndarray, edges: np.ndarray, search_radius: int = 5) -> np.ndarray:
    """
    Ajusta cada punto del contorno al borde más cercano en la imagen de bordes.

    Args:
        contour_points: np.ndarray float32 (N, 2) — coordenadas (x, y)
        edges         : np.ndarray uint8 (H, W) {0, 255} — bordes originales
        search_radius : radio de búsqueda en píxeles

    Returns:
        np.ndarray float32 (N, 2) contorno ajustado
    """
    if len(contour_points) == 0:
        return contour_points

    adjusted = []
    h, w = edges.shape

    for point in contour_points:
        x, y = int(point[0]), int(point[1])

        x_min = max(0, x - search_radius)
        x_max = min(w, x + search_radius + 1)
        y_min = max(0, y - search_radius)
        y_max = min(h, y + search_radius + 1)

        region = edges[y_min:y_max, x_min:x_max]
        edge_pixels = np.where(region > 0)

        if len(edge_pixels[0]) > 0:
            edge_coords = np.column_stack((edge_pixels[1] + x_min, edge_pixels[0] + y_min))
            distances = np.sqrt(np.sum((edge_coords - np.array([x, y])) ** 2, axis=1))
            adjusted.append(edge_coords[int(np.argmin(distances))])
        else:
            adjusted.append([x, y])

    return np.array(adjusted, dtype=np.float32)


def aplicar_morfologia(edges: np.ndarray, kernel: np.ndarray, operation: str = 'close') -> np.ndarray:
    """
    Aplica una única operación morfológica sobre la imagen de bordes.
    Punto centralizado de morfología: solo se invoca desde get_contours,
    garantizando que no exista más de un cierre por imagen en el pipeline.

    Args:
        edges    : np.ndarray uint8 (H, W) {0, 255}
        kernel   : np.ndarray uint8 cuadrado
        operation: 'close' | 'open' | 'dilate' | 'erode' | 'gradient'

    Returns:
        np.ndarray uint8 (H, W)
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


def simplificar_contorno_adaptativo(contour: np.ndarray, base_epsilon: float = 0.001) -> np.ndarray:
    """
    Simplifica el contorno de forma adaptativa según su complejidad relativa.

    Args:
        contour      : contorno OpenCV (N, 1, 2)
        base_epsilon : factor base de simplificación

    Returns:
        Contorno simplificado (M, 1, 2)
    """
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    complexity = perimeter / np.sqrt(area) if area > 0 else 1.0

    if complexity > 20:
        epsilon = base_epsilon * 0.1 * perimeter
    elif complexity > 10:
        epsilon = base_epsilon * 0.5 * perimeter
    else:
        epsilon = base_epsilon * perimeter

    return cv2.approxPolyDP(contour, epsilon, True)


def get_contours(edges: np.ndarray, kernel: np.ndarray, config: dict | None = None) -> np.ndarray:
    """
    Extrae el contorno principal de una imagen de bordes.

    Aplica exactamente UNA operación morfológica (cierre) antes de buscar
    contornos. El llamador NO debe aplicar ningún cierre previo sobre edges
    para evitar el doble desplazamiento de bordes.

    Args:
        edges : np.ndarray uint8 (H, W) {0, 255} — salida directa de Canny
        kernel: np.ndarray uint8 — construido con KERNEL_MORPH de image_processing
        config: dict con claves opcionales:
            simplify_method    : 'adaptive' | 'fixed' | 'none'
            epsilon_factor     : float
            min_area           : int — área mínima en px²
            morph_op           : str — operación para aplicar_morfologia
            multi_contour_mode : 'single' | 'union' | 'hull'
            trace_edges        : bool — usa extraer_contorno_preciso
            close_iterations   : int — iteraciones para extraer_contorno_preciso
            close_kernel       : int — kernel para extraer_contorno_preciso
            adjust_to_edges    : bool — ajusta puntos al borde más cercano
            search_radius      : int — radio de ajuste

    Returns:
        np.ndarray float32 (N, 2) — coordenadas (x, y) del contorno
        np.ndarray vacío si no se detectan contornos válidos
    """
    if config is None:
        config = {}

    original_edges = edges.copy()
    trace_edges = config.get('trace_edges', False)

    if trace_edges:
        closed_edges, _ = extraer_contorno_preciso(
            edges,
            max_iterations=config.get('close_iterations', 5),
            kernel_size=config.get('close_kernel', 3)
        )
    else:
        closed_edges = aplicar_morfologia(edges, kernel, config.get('morph_op', 'close'))

    retrieval_mode = cv2.RETR_TREE if config.get('use_tree', False) else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(closed_edges, retrieval_mode, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.array([])

    min_area = config.get('min_area', 100)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not contours:
        return np.array([])

    multi_contour_mode = config.get('multi_contour_mode', 'single')

    if multi_contour_mode == 'union':
        main_contour = combinar_contornos_ordenados(contours, method='connected')
    elif multi_contour_mode == 'hull':
        all_points = np.vstack(contours)
        main_contour = cv2.convexHull(all_points)
    else:
        main_contour = max(contours, key=cv2.contourArea)

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

    points = approx.reshape(-1, 2).astype(np.float32)

    if config.get('adjust_to_edges', True) and trace_edges:
        search_radius = config.get('search_radius', 10)
        points = ajustar_contorno_a_bordes(points, original_edges, search_radius)

    return points


def get_contours_multiples(edges: np.ndarray, kernel: np.ndarray, max_contours: int = 5, min_area: int = 100) -> list[np.ndarray]:
    """
    Extrae múltiples contornos de la imagen, ordenados por área descendente.

    Aplica un único cierre morfológico con el kernel provisto.

    Args:
        edges       : np.ndarray uint8 (H, W) {0, 255}
        kernel      : np.ndarray uint8 cuadrado
        max_contours: número máximo de contornos a devolver
        min_area    : área mínima en px²

    Returns:
        Lista de np.ndarray float32 (N, 2)
    """
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return []

    valid_contours = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= min_area]
    valid_contours.sort(key=lambda x: x[1], reverse=True)

    results = []
    for contour, _ in valid_contours[:max_contours]:
        approx = simplificar_contorno_adaptativo(contour)
        results.append(approx.reshape(-1, 2).astype(np.float32))

    return results


def get_contours_jerarquicos(edges: np.ndarray, kernel: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Extrae contornos con su jerarquía (para formas con huecos interiores).

    Aplica un único cierre morfológico con el kernel provisto.

    Args:
        edges : np.ndarray uint8 (H, W) {0, 255}
        kernel: np.ndarray uint8 cuadrado

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]] — (externos, internos)
        Cada elemento es np.ndarray float32 (N, 2)
    """
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if not contours or hierarchy is None:
        return [], []

    hierarchy = hierarchy[0]
    externos: list[np.ndarray] = []
    internos: list[np.ndarray] = []

    for contour, h in zip(contours, hierarchy):
        approx = simplificar_contorno_adaptativo(contour)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if h[3] == -1:
            externos.append(pts)
        else:
            internos.append(pts)

    return externos, internos


def refinar_contorno_subpixel(gray_image: np.ndarray, contour: np.ndarray, win_size: int = 5) -> np.ndarray:
    """
    Refina las posiciones del contorno a nivel subpíxel usando cornerSubPix.

    Args:
        gray_image: np.ndarray uint8 (H, W)
        contour   : np.ndarray (N, 1, 2) o (N, 2)
        win_size  : tamaño de la ventana de refinamiento

    Returns:
        np.ndarray float32 (N, 2) contorno refinado
    """
    points = contour.reshape(-1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    try:
        return cv2.cornerSubPix(gray_image, points, (win_size, win_size), (-1, -1), criteria)
    except Exception:
        return points


def calcular_curvatura(points: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Calcula la curvatura en cada punto del contorno.
    Útil para identificar vértices y esquinas significativas.

    Args:
        points: np.ndarray float32 (N, 2)
        window: tamaño de la ventana de vecindad

    Returns:
        np.ndarray float64 (N,) con valores de curvatura en radianes
    """
    n = len(points)
    curvatures = np.zeros(n)

    for i in range(n):
        p_prev = points[(i - window) % n]
        p_curr = points[i]
        p_next = points[(i + window) % n]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        curvatures[i] = abs(np.arctan2(float(np.cross(v1, v2)), float(np.dot(v1, v2))))

    return curvatures


def detectar_puntos_criticos(points: np.ndarray, curvature_threshold: float = 0.3) -> list[int]:
    """
    Detecta índices de puntos críticos (esquinas, curvas pronunciadas).

    Args:
        points              : np.ndarray float32 (N, 2)
        curvature_threshold : umbral mínimo de curvatura en radianes

    Returns:
        Lista de índices enteros de puntos críticos
    """
    curvatures = calcular_curvatura(points)
    critical_indices: list[int] = []

    for i in range(len(curvatures)):
        if curvatures[i] > curvature_threshold:
            prev_idx = (i - 1) % len(curvatures)
            next_idx = (i + 1) % len(curvatures)
            if curvatures[i] >= curvatures[prev_idx] and curvatures[i] >= curvatures[next_idx]:
                critical_indices.append(i)

    return critical_indices


def remuestrear_contorno(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Remuestrea el contorno para tener num_points uniformemente espaciados
    por longitud de arco, preservando mejor la forma que submuestreo directo.

    Args:
        points    : np.ndarray (N, 2)
        num_points: número deseado de puntos de salida

    Returns:
        np.ndarray float64 (num_points, 2)
    """
    if len(points) < 3:
        return points

    diff = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    cumulative = np.zeros(len(points))
    cumulative[1:] = np.cumsum(distances)
    total_length = cumulative[-1]

    if total_length == 0:
        return points[:num_points] if len(points) >= num_points else points

    target_distances = np.linspace(0, total_length, num_points, endpoint=False)
    new_points = np.zeros((num_points, 2))

    for i, target in enumerate(target_distances):
        idx = int(np.searchsorted(cumulative, target, side='right')) - 1
        idx = max(0, min(idx, len(points) - 2))

        segment_start = cumulative[idx]
        segment_end = cumulative[idx + 1]

        t = (target - segment_start) / (segment_end - segment_start) if segment_end > segment_start else 0.0
        new_points[i] = points[idx] + t * (points[idx + 1] - points[idx])

    return new_points


def suavizar_contorno_media_movil(points: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Suaviza el contorno usando media móvil circular.

    Args:
        points: np.ndarray (N, 2)
        window: tamaño de la ventana

    Returns:
        np.ndarray float64 (N, 2) suavizado
    """
    n = len(points)
    smoothed = np.zeros_like(points, dtype=np.float64)
    half = window // 2

    for i in range(n):
        indices = [(i + j) % n for j in range(-half, half + 1)]
        smoothed[i] = np.mean(points[indices], axis=0)

    return smoothed


def debug_contour_order(points: np.ndarray, image_edges: np.ndarray, kernel: np.ndarray) -> np.ndarray | None:
    """
    Visualiza el orden de puntos del contorno sobre la imagen de bordes.
    Solo para uso en desarrollo/debug.

    Args:
        points      : np.ndarray (N, 2)
        image_edges : np.ndarray uint8 (H, W) o (H, W, 3)
        kernel      : no utilizado, mantenido por compatibilidad de firma

    Returns:
        np.ndarray uint8 (H, W, 3) imagen anotada, o None si points está vacío
    """
    if len(points) == 0:
        return None

    display_img = cv2.cvtColor(image_edges, cv2.COLOR_GRAY2BGR) if image_edges.ndim == 2 else image_edges.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(display_img, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.putText(display_img, str(i), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    for i in range(len(points) - 1):
        cv2.line(display_img, tuple(points[i].astype(int)), tuple(points[i + 1].astype(int)), (0, 255, 0), 1)

    if len(points) > 2:
        cv2.line(display_img, tuple(points[-1].astype(int)), tuple(points[0].astype(int)), (0, 255, 0), 1)

    return display_img