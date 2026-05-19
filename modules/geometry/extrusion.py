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
    Crea diccionario de datos para Plotly Mesh3d con opciones mejoradas.
    
    Args:
        vertices: Array de vértices 3D
        faces: Lista de caras (triángulos)
        **kwargs: color, opacity, lighting, etc.
        
    Returns:
        Dict con datos para Plotly Mesh3d
    """
    if len(vertices) == 0 or len(faces) == 0:
        return {}
    
    faces_array = np.array(faces)
    
    # Calcular normales para mejor iluminación
    intensity = None
    if kwargs.get('compute_intensity', True):
        # Usar coordenada Z para intensidad (simula iluminación desde arriba)
        intensity = vertices[:, 2].tolist()
    
    plotly_data = {
        'x': vertices[:, 0].tolist(),
        'y': vertices[:, 1].tolist(),
        'z': vertices[:, 2].tolist(),
        'i': faces_array[:, 0].tolist(),
        'j': faces_array[:, 1].tolist(),
        'k': faces_array[:, 2].tolist(),
        'color': kwargs.get('color', 'lightblue'),
        'opacity': kwargs.get('opacity', 0.8),
        'flatshading': kwargs.get('flatshading', False),
        'lighting': dict(
            ambient=0.4,
            diffuse=0.7,
            specular=0.3,
            roughness=0.5,
            fresnel=0.2
        ),
        'lightposition': dict(x=100, y=200, z=300),
    }
    
    # Agregar intensidad si está disponible
    if intensity and kwargs.get('use_intensity', False):
        plotly_data['intensity'] = intensity
        plotly_data['colorscale'] = kwargs.get('colorscale', 'Blues')
        del plotly_data['color']
    
    return plotly_data


def sort_contour_points(points, method='original'):
    """
    Ordena los puntos del contorno.
    
    Args:
        points: Array de puntos 2D
        method: 
            - 'original': Mantiene el orden de OpenCV (recomendado para figuras complejas)
            - 'angular': Ordena angularmente respecto al centroide (solo para convexos)
            - 'optimized': Verifica y corrige la orientación sin reordenar
    
    Returns:
        Array de puntos ordenados
    """
    if len(points) < 3:
        return points
    
    points = np.array(points, dtype=np.float64)
    
    if method == 'angular':
        # Ordenamiento angular (solo para formas convexas simples)
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:,1] - center[1],
                            points[:,0] - center[0])
        return points[np.argsort(angles)]
    
    elif method == 'optimized':
        # Verificar orientación (debe ser antihorario para Delaunay)
        # Calcular el área con signo usando la fórmula del polígono
        n = len(points)
        signed_area = 0.0
        for i in range(n):
            j = (i + 1) % n
            signed_area += points[i, 0] * points[j, 1]
            signed_area -= points[j, 0] * points[i, 1]
        signed_area /= 2.0
        
        # Si el área es negativa, invertir el orden (hacerlo antihorario)
        if signed_area < 0:
            return points[::-1]
        return points
    
    else:  # 'original'
        # Mantener el orden original de OpenCV
        # Solo verificar orientación antihoraria
        n = len(points)
        signed_area = 0.0
        for i in range(n):
            j = (i + 1) % n
            signed_area += points[i, 0] * points[j, 1]
            signed_area -= points[j, 0] * points[i, 1]
        
        if signed_area < 0:
            return points[::-1]
        return points


def ensure_closed_polygon(points, threshold=0.1):
    """Asegura que el polígono esté cerrado."""
    if len(points) < 3:
        return points
    
    points = np.array(points, dtype=np.float64)
    first = points[0]
    last = points[-1]
    distance = np.linalg.norm(first - last)
    
    if distance > threshold:
        return np.vstack([points, [points[0]]])
    
    return points


def calcular_normales_vertices(vertices, faces):
    """
    Calcula las normales por vértice para mejor iluminación.
    
    Args:
        vertices: Array de vértices 3D
        faces: Lista de caras (triángulos)
        
    Returns:
        Array de normales por vértice
    """
    normals = np.zeros_like(vertices)
    
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Calcular normal de la cara
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        
        # Normalizar
        norm = np.linalg.norm(face_normal)
        if norm > 0:
            face_normal /= norm
        
        # Acumular en cada vértice
        for idx in face:
            normals[idx] += face_normal
    
    # Normalizar todas las normales de vértices
    for i in range(len(normals)):
        norm = np.linalg.norm(normals[i])
        if norm > 0:
            normals[i] /= norm
    
    return normals


def optimizar_triangulacion(points_2d, faces):
    """
    Optimiza la triangulación evitando triángulos degenerados.
    
    Args:
        points_2d: Puntos 2D del polígono
        faces: Lista de caras originales
        
    Returns:
        Lista de caras optimizadas
    """
    optimized_faces = []
    
    for face in faces:
        if len(face) != 3:
            continue
            
        # Verificar que el triángulo no sea degenerado
        p0, p1, p2 = points_2d[face[0]], points_2d[face[1]], points_2d[face[2]]
        
        # Calcular área del triángulo
        area = abs(np.cross(p1 - p0, p2 - p0)) / 2
        
        # Solo incluir triángulos con área significativa
        if area > 1e-8:
            optimized_faces.append(face)
    
    return optimized_faces


def crear_malla_suave(points_2d, height=1.0, subdivisions=1):
    """
    Crea una malla 3D con subdivisión para mayor suavidad.
    
    Args:
        points_2d: Puntos 2D del polígono
        height: Altura de extrusión
        subdivisions: Número de subdivisiones en altura
        
    Returns:
        Tuple (vértices, caras)
    """
    if len(points_2d) < 3:
        return np.array([]), []
    
    n = len(points_2d)
    vertices = []
    faces = []
    
    # Crear vértices en múltiples niveles de Z
    num_levels = subdivisions + 2  # Base + subdivisiones + tapa
    z_levels = np.linspace(0, height, num_levels)
    
    for z in z_levels:
        for point in points_2d:
            vertices.append([point[0], point[1], z])
    
    vertices = np.array(vertices)
    
    # Caras laterales con subdivisiones
    for level in range(num_levels - 1):
        offset_bottom = level * n
        offset_top = (level + 1) * n
        
        for i in range(n):
            j = (i + 1) % n
            
            # Dos triángulos por cara lateral
            faces.append([offset_bottom + i, offset_bottom + j, offset_top + i])
            faces.append([offset_bottom + j, offset_top + j, offset_top + i])
    
    # Base y tapa usando triangulación
    base_triangles = triangulate_polygon(points_2d)
    
    for tri in base_triangles:
        # Base (invertir orientación)
        faces.append([tri[0], tri[2], tri[1]])
        # Tapa
        top_offset = (num_levels - 1) * n
        faces.append([top_offset + tri[0], top_offset + tri[1], top_offset + tri[2]])
    
    return vertices, faces
