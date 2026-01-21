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
