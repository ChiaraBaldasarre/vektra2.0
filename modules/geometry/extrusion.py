"""
Módulo de extrusión 3D (Versión Final con Métricas Físicas).
Centraliza la configuración, validación, extrusión y cálculos analíticos en POO.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.spatial import Delaunay
import streamlit as st

# Límite máximo de puntos para evitar ralentizaciones en la triangulación
_MAX_TRIANGULATION_POINTS = 800


@dataclass(frozen=True)
class ExtrusionConfig:
    """Clase inmutable para gestionar de forma centralizada los parámetros de extrusión."""
    height: float = 1.0
    target_size: float = 1.0
    triangulate: bool = True
    closed: bool = True
    subdivisions: int = 1
    sort_method: str = 'original'  # Opciones: 'original', 'angular', 'optimized'
    suavizado_de_cara: bool = False  # Parámetro solicitado en el ticket de Chiara
    color: str = 'lightblue'
    opacity: float = 0.8
    flatshading: bool = False
    use_intensity: bool = False
    colorscale: str = 'Blues'


class Extruder:
    """Encapsula el motor geométrico para la generación adaptativa de mallas 3D."""

    def __init__(self, config: ExtrusionConfig = None):
        self.config = config if config is not None else ExtrusionConfig()

    def _validate_input(self, points: Any) -> np.ndarray:
        """Valida defensivamente que la entrada sea una matriz de puntos 2D válida."""
        if points is None:
            return np.empty((0, 2), dtype=np.float32)

        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Estructura de puntos inválida: Se esperaba forma (N, 2), se recibió {points.shape}")

        return points

    def normalize_points(self, points_img: np.ndarray) -> np.ndarray:
        """Normaliza puntos 2D de la imagen al tamaño objetivo de la configuración."""
        try:
            points_cleaned = self._validate_input(points_img)
            if len(points_cleaned) < 2:
                return points_cleaned

            points = points_cleaned.astype(np.float32)
            min_x, min_y = points.min(axis=0)
            max_x, max_y = points.max(axis=0)
            width = max_x - min_x
            height = max_y - min_y

            if width == 0: width = 1.0
            if height == 0: height = 1.0

            scale = self.config.target_size / max(width, height)

            points[:, 0] = (points[:, 0] - (min_x + width / 2)) * scale
            points[:, 1] = -(points[:, 1] - (min_y + height / 2)) * scale
            return points
        except Exception as e:
            st.error(f"Error crítico en la normalización geométrica: {e}")
            return np.empty((0, 2), dtype=np.float32)

    @staticmethod
    def _is_point_inside_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
        """Algoritmo de Ray Casting para verificar si un punto está contenido en un polígono."""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @st.cache_data
    def triangulate_polygon(_self, points_2d: np.ndarray) -> List[List[int]]:
        """Triangula formas complejas por Delaunay cacheando el costo computacional alto."""
        if len(points_2d) < 3:
            return []
        try:
            tri = Delaunay(points_2d)
            triangles = []
            for simplex in tri.simplices:
                triangle_points = points_2d[simplex]
                centroid = triangle_points.mean(axis=0)

                if _self._is_point_inside_polygon(centroid, points_2d):
                    p0, p1, p2 = triangle_points[0], triangle_points[1], triangle_points[2]
                    area = abs(np.cross(p1 - p0, p2 - p0)) / 2.0
                    if area > 1e-7:
                        triangles.append(simplex.tolist())
            return triangles
        except Exception:
            return []

    def sort_contour_points(self, points: np.ndarray) -> np.ndarray:
        """Garantiza la correcta ordenación y orientación antihoraria de los contornos."""
        points = self._validate_input(points)
        if len(points) < 3:
            return points

        points = points.astype(np.float64)
        method = self.config.sort_method

        if method == 'angular':
            center = np.mean(points, axis=0)
            angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
            return points[np.argsort(angles)]

        n = len(points)
        signed_area = 0.0
        for i in range(n):
            j = (i + 1) % n
            signed_area += points[i, 0] * points[j, 1] - points[j, 0] * points[i, 1]
        signed_area /= 2.0

        if signed_area < 0:
            return points[::-1]
        return points

    def process(self, points_2d: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
        """Ejecuta el pipeline completo de extrusión volumétrica con topología adaptativa."""
        try:
            points_cleaned = self._validate_input(points_2d)
            min_points = 3 if self.config.closed else 2
            if len(points_cleaned) < min_points:
                return np.array([]), []

            points_sorted = self.sort_contour_points(points_cleaned)
            num_levels = self.config.subdivisions + 2 if self.config.suavizado_de_cara else 2
            z_levels = np.linspace(0, self.config.height, num_levels)

            n = len(points_sorted)
            vertices = []
            faces = []

            # 1. Generar Vértices 3D
            for z in z_levels:
                for point in points_sorted:
                    vertices.append([point[0], point[1], z])
            vertices = np.array(vertices)

            # 2. Generar Caras Laterales
            for level in range(num_levels - 1):
                offset_bottom = level * n
                offset_top = (level + 1) * n
                limit = n if self.config.closed else n - 1

                for i in range(limit):
                    j = (i + 1) % n
                    faces.append([offset_bottom + i, offset_bottom + j, offset_top + i])
                    faces.append([offset_bottom + j, offset_top + j, offset_top + i])

            # 3. Generar Tapas (Si es un objeto cerrado)
            if self.config.closed:
                can_triangulate = self.config.triangulate and 3 < n <= _MAX_TRIANGULATION_POINTS
                if can_triangulate:
                    base_triangles = self.triangulate_polygon(points_sorted)
                    for tri in base_triangles:
                        faces.append([tri[0], tri[2], tri[1]])
                        top_offset = (num_levels - 1) * n
                        faces.append([top_offset + tri[0], top_offset + tri[1], top_offset + tri[2]])
                else:
                    top_offset = (num_levels - 1) * n
                    if n == 3:
                        faces.append([0, 2, 1])
                        faces.append([top_offset, top_offset + 1, top_offset + 2])
                    elif n == 4:
                        faces.append([0, 1, 2])
                        faces.append([0, 2, 3])
                        faces.append([top_offset, top_offset + 2, top_offset + 1])
                        faces.append([top_offset, top_offset + 3, top_offset + 2])
                    else:
                        for i in range(1, n - 1):
                            faces.append([0, i, i + 1])
                            faces.append([top_offset, top_offset + i + 1, top_offset + i])

            return vertices, faces
        except Exception as e:
            st.error(f"Fallo en la generación de la malla 3D: {e}")
            return np.array([]), []

    def create_plotly_mesh(self, vertices: np.ndarray, faces: List[List[int]]) -> Dict[str, Any]:
        """Empaqueta los arrays en el formato nativo para renderizado en Plotly Mesh3D."""
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
            'opacity': self.config.opacity,
            'flatshading': self.config.flatshading,
            'lighting': dict(ambient=0.4, diffuse=0.7, specular=0.3, roughness=0.5, fresnel=0.2),
            'lightposition': dict(x=100, y=200, z=300),
        }

        if self.config.use_intensity:
            plotly_data['intensity'] = vertices[:, 2].tolist()
            plotly_data['colorscale'] = self.config.colorscale
        else:
            plotly_data['color'] = self.config.color

        return plotly_data

    def calculate_volume(self, vertices: np.ndarray, faces: List[List[int]]) -> float:
        """Calcula el volumen total usando el Teorema de la Divergencia (Solo sólidos cerrados)."""
        if len(vertices) == 0 or len(faces) == 0 or not self.config.closed:
            return 0.0

        volumen = 0.0
        faces_array = np.array(faces)
        for face in faces_array:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            v_tetraedro = np.dot(v0, np.cross(v1, v2)) / 6.0
            volumen += v_tetraedro

        return abs(volumen)

    def calculate_surface_area(self, vertices: np.ndarray, faces: List[List[int]]) -> float:
        """Calcula el área superficial externa total de la malla."""
        if len(vertices) == 0 or len(faces) == 0:
            return 0.0

        area_total = 0.0
        faces_array = np.array(faces)
        for face in faces_array:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            area_triangulo = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0
            area_total += area_triangulo

        return area_total

    @staticmethod
    def compute_vertex_normals(vertices: np.ndarray, faces: List[List[int]]) -> np.ndarray:
        """Calcula las normales analíticas de los vértices para corregir la iluminación."""
        normals = np.zeros_like(vertices)
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            face_normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(face_normal)
            if norm > 0:
                face_normal /= norm
            for idx in face:
                normals[idx] += face_normal

        for i in range(len(normals)):
            norm = np.linalg.norm(normals[i])
            if norm > 0:
                normals[i] /= norm
        return normals