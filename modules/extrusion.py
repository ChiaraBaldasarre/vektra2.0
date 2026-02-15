"""
Módulo de extrusión 3D.
Contiene funciones para normalizar puntos, extruir polígonos y crear mallas para Plotly.
"""

import numpy as np
from scipy.spatial import Delaunay


def normalize_points(points_img, target_size=1.0):
    """
    Normaliza puntos 2D a un rango determinado.
    
    Args:
        points_img: Array de puntos 2D
        target_size: Tamaño objetivo para normalización
        
    Returns:
        Array de puntos normalizados
    """
    if len(points_img) == 0:
        return np.array([])
    
    points = points_img.astype(np.float32)
    
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


def is_point_inside_polygon(point, polygon):
    """Verifica si un punto está dentro de un polígono usando ray casting."""
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


def triangulate_polygon(points_2d):
    """Triangula un polígono 2D (puede ser cóncavo)."""
    if len(points_2d) < 3:
        return []
    
    try:
        tri = Delaunay(points_2d)
        
        # Filtrar triángulos que están fuera del polígono
        triangles = []
        for simplex in tri.simplices:
            triangle_points = points_2d[simplex]
            centroid = triangle_points.mean(axis=0)
            
            if is_point_inside_polygon(centroid, points_2d):
                triangles.append(simplex.tolist())
        
        return triangles
    except Exception:
        return []


def extrude_polygon(points_2d, height=1.0, triangulate=True):
    """
    Extruye un polígono 2D a 3D con triangulación correcta para formas cóncavas.
    
    Args:
        points_2d: Array de puntos 2D del polígono
        height: Altura de extrusión
        triangulate: Si usar triangulación para formas cóncavas
        
    Returns:
        Tuple (vértices, caras)
    """
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
    
    # Caras laterales
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, i + n])
        faces.append([j, j + n, i + n])
    
    # Base y tapa
    if triangulate and n > 3:
        base_triangles = triangulate_polygon(points_2d)
        for tri in base_triangles:
            faces.append([tri[0], tri[2], tri[1]])
        
        for tri in base_triangles:
            faces.append([tri[0] + n, tri[1] + n, tri[2] + n])
    else:
        if n == 3:
            faces.append([0, 2, 1])
            faces.append([n, n+1, n+2])
        elif n == 4:
            faces.append([0, 1, 2])
            faces.append([0, 2, 3])
            faces.append([n, n+2, n+1])
            faces.append([n, n+3, n+2])
        else:
            for i in range(1, n - 1):
                faces.append([0, i, i + 1])
                faces.append([n, n + i + 1, n + i])
    
    return vertices, faces


def create_plotly_mesh(vertices, faces, **kwargs):
    """
    Crea diccionario de datos para Plotly Mesh3d.
    
    Args:
        vertices: Array de vértices 3D
        faces: Lista de caras (triángulos)
        **kwargs: color, opacity, etc.
        
    Returns:
        Dict con datos para Plotly Mesh3d
    """
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
    """Ordena puntos angularmente respecto al centroide."""
    if len(points) < 3:
        return points
    
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:,1] - center[1],
                        points[:,0] - center[0])
    return points[np.argsort(angles)]


def ensure_closed_polygon(points, threshold=0.1):
    """Asegura que el polígono esté cerrado."""
    if len(points) < 3:
        return points
    
    first = points[0]
    last = points[-1]
    distance = np.linalg.norm(first - last)
    
    if distance > threshold:
        return np.vstack([points, [points[0]]])
    
    return points
