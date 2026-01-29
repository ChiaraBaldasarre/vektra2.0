import numpy as np
from scipy.spatial import Delaunay

def normalize_points(points_img, target_size=1.0):
    """... tu código existente ..."""
    # Asegúrate de que points_img no esté vacío
    if len(points_img) == 0:
        return np.array([])
    
    points = points_img.astype(np.float32)
    
    # Manejar caso de un solo punto
    if len(points) < 2:
        return points
    
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Evitar división por cero
    if width == 0:
        width = 1
    if height == 0:
        height = 1
    
    scale = target_size / max(width, height)
    
    points[:, 0] = (points[:, 0] - (min_x + width/2)) * scale
    points[:, 1] = -(points[:, 1] - (min_y + height/2)) * scale
    
    return points

def extrude_polygon(points_2d, height=1.0):
    """... tu código existente ..."""
    if len(points_2d) < 3:
        return np.array([]), []
    
    n = len(points_2d)
    vertices = []
    
    for z in [0, height]:
        for point in points_2d:
            vertices.append([point[0], point[1], z])
    
    vertices = np.array(vertices)
    
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, i + n])
        faces.append([j, j + n, i + n])
    
    # Solo agregar base/tapa si hay suficientes puntos
    if n > 2:
        for i in range(1, n - 1):
            faces.append([0, i, i + 1])
            faces.append([n, n + i + 1, n + i])
    
    return vertices, faces

def create_plotly_mesh(vertices, faces, **kwargs):
    """... tu código existente ..."""
    if len(vertices) == 0 or len(faces) == 0:
        return {}
    
    faces_array = np.array(faces)
    
    plotly_data = {
        'x': vertices[:, 0].tolist(),
        'y': vertices[:, 1].tolist(),
        'z': vertices[:, 2].tolist(),
        'i': faces_array[:, 0].tolist(),
        'j': faces_array[:, 1].tolist(),
        'k': faces_array[:, 2].tolist(),
        'color': kwargs.get('color', 'lightblue'),
        'opacity': kwargs.get('opacity', 0.8),
        'flatshading': True,
    }
    
    return plotly_data

def sort_contour_points(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:,1] - center[1],
                        points[:,0] - center[0])
    return points[np.argsort(angles)]
import numpy as np
from scipy.spatial import Delaunay

def normalize_points(points_img, target_size=1.0):
    """Normaliza puntos 2D a un rango determinado."""
    if len(points_img) == 0:
        return np.array([])
    
    points = points_img.astype(np.float32)
    
    if len(points) < 2:
        return points
    
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0:
        width = 1
    if height == 0:
        height = 1
    
    scale = target_size / max(width, height)
    
    points[:, 0] = (points[:, 0] - (min_x + width/2)) * scale
    points[:, 1] = -(points[:, 1] - (min_y + height/2)) * scale
    
    return points

def triangulate_polygon(points_2d):
    """Triangula un polígono 2D (puede ser cóncavo)."""
    if len(points_2d) < 3:
        return []
    
    # Para polígonos simples, usar triangulación de Delaunay
    # Primero necesitamos crear una nube de puntos
    tri = Delaunay(points_2d)
    
    # Filtrar triángulos que están fuera del polígono
    triangles = []
    for simplex in tri.simplices:
        # Verificar si el centroide está dentro del polígono
        triangle_points = points_2d[simplex]
        centroid = triangle_points.mean(axis=0)
        
        # Verificar si está dentro del polígono (algoritmo winding number)
        if is_point_inside_polygon(centroid, points_2d):
            triangles.append(simplex.tolist())
    
    return triangles

def is_point_inside_polygon(point, polygon):
    """Verifica si un punto está dentro de un polígono usando winding number."""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def extrude_polygon(points_2d, height=1.0, triangulate=True):
    """Extruye un polígono 2D a 3D con triangulación correcta para formas cóncavas."""
    if len(points_2d) < 3:
        return np.array([]), []
    
    n = len(points_2d)
    vertices = []
    
    # Crear vértices en Z=0 y Z=height
    for z in [0, height]:
        for point in points_2d:
            vertices.append([point[0], point[1], z])
    
    vertices = np.array(vertices)
    faces = []
    
    # Caras laterales (siempre igual)
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, i + n])      # Triángulo 1
        faces.append([j, j + n, i + n])  # Triángulo 2
    
    # Base y tapa - IMPORTANTE: usar triangulación para formas cóncavas
    if triangulate and n > 3:
        # Triangulación de la base (puntos en orden inverso para normal hacia abajo)
        base_triangles = triangulate_polygon(points_2d)
        for tri in base_triangles:
            # Mantener orientación consistente
            faces.append([tri[0], tri[2], tri[1]])
        
        # Triangulación de la tapa (puntos desplazados +n)
        for tri in base_triangles:
            # Invertir orden para normal hacia arriba
            faces.append([tri[0] + n, tri[1] + n, tri[2] + n])
    else:
        # Método simple para formas convexas o triángulos
        if n == 3:
            # Triángulo
            faces.append([0, 2, 1])          # Base
            faces.append([n, n+1, n+2])      # Tapa
        elif n == 4:
            # Cuadrilátero - dividir en 2 triángulos
            faces.append([0, 1, 2])          # Base tri 1
            faces.append([0, 2, 3])          # Base tri 2
            faces.append([n, n+2, n+1])      # Tapa tri 1 (orden invertido)
            faces.append([n, n+3, n+2])      # Tapa tri 2
        else:
            # Polígono con más de 4 lados - abanico desde el primer punto
            for i in range(1, n - 1):
                faces.append([0, i, i + 1])          # Base
                faces.append([n, n + i + 1, n + i])  # Tapa
    
    return vertices, faces

def create_plotly_mesh(vertices, faces, **kwargs):
    """Crea diccionario de datos para Plotly Mesh3d."""
    if len(vertices) == 0 or len(faces) == 0:
        return {}
    
    faces_array = np.array(faces)
    
    plotly_data = {
        'x': vertices[:, 0].tolist(),
        'y': vertices[:, 1].tolist(),
        'z': vertices[:, 2].tolist(),
        'i': faces_array[:, 0].tolist(),
        'j': faces_array[:, 1].tolist(),
        'k': faces_array[:, 2].tolist(),
        'color': kwargs.get('color', 'lightblue'),
        'opacity': kwargs.get('opacity', 0.8),
        'flatshading': True,
    }
    
    return plotly_data

def sort_contour_points(points):
    """Ordena puntos angularmente (solo para formas convexas)."""
    if len(points) < 3:
        return points
    
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:,1] - center[1],
                        points[:,0] - center[0])
    return points[np.argsort(angles)]

# Versión alternativa específica para formas cóncavas
def ensure_closed_polygon(points, threshold=0.1):
    """Asegura que el polígono esté cerrado (primer y último punto cercanos)."""
    if len(points) < 3:
        return points
    
    # Calcular distancia entre primer y último punto
    first = points[0]
    last = points[-1]
    distance = np.linalg.norm(first - last)
    
    # Si no está cerrado, añadir el primer punto al final
    if distance > threshold:
        return np.vstack([points, [points[0]]])
    
    return points