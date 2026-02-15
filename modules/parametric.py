"""
Módulo de Superficies Paramétricas.
Genera formas 3D a partir de ecuaciones matemáticas de vectores.
Permite crear superficies usando funciones f(u,v) → (x, y, z).
"""

import numpy as np


def evaluar_superficie_parametrica(func_x, func_y, func_z, u_range, v_range, resolution=50):
    """
    Evalúa una superficie paramétrica definida por tres funciones.
    
    Args:
        func_x: Función λ(u,v) → x
        func_y: Función λ(u,v) → y
        func_z: Función λ(u,v) → z
        u_range: Tuple (u_min, u_max)
        v_range: Tuple (v_min, v_max)
        resolution: Número de puntos por eje
        
    Returns:
        Tuple (vertices, faces) para Plotly Mesh3d
    """
    u = np.linspace(u_range[0], u_range[1], resolution)
    v = np.linspace(v_range[0], v_range[1], resolution)
    U, V = np.meshgrid(u, v)
    
    # Evaluar funciones
    X = func_x(U, V)
    Y = func_y(U, V)
    Z = func_z(U, V)
    
    # Convertir a vertices
    vertices = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    # Generar caras (triángulos)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Índices del cuadrilátero
            p1 = i * resolution + j
            p2 = i * resolution + (j + 1)
            p3 = (i + 1) * resolution + (j + 1)
            p4 = (i + 1) * resolution + j
            
            # Dividir en 2 triángulos
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])
    
    return vertices, faces


# ============ SUPERFICIES CLÁSICAS ============

def generar_paraboloide(a=1.0, b=1.0, height=2.0, resolution=40):
    """
    Genera un paraboloide: z = (x/a)² + (y/b)²
    
    Args:
        a, b: Parámetros de escala
        height: Altura máxima
        resolution: Resolución de la malla
    """
    r_max = np.sqrt(height) * max(a, b)
    
    def fx(u, v): return u
    def fy(u, v): return v
    def fz(u, v): return (u/a)**2 + (v/b)**2
    
    vertices, faces = evaluar_superficie_parametrica(
        fx, fy, fz,
        (-r_max, r_max), (-r_max, r_max),
        resolution
    )
    
    # Filtrar puntos que excedan la altura
    mask = vertices[:, 2] <= height
    return _filtrar_superficie(vertices, faces, mask)


def generar_silla_montar(a=1.0, b=1.0, size=2.0, resolution=40):
    """
    Genera una silla de montar (paraboloide hiperbólico): z = (x/a)² - (y/b)²
    """
    def fx(u, v): return u
    def fy(u, v): return v
    def fz(u, v): return (u/a)**2 - (v/b)**2
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (-size, size), (-size, size),
        resolution
    )


def generar_onda_seno(amplitud=1.0, frecuencia=1.0, size=3.0, resolution=50):
    """
    Genera una superficie de onda: z = A * sin(f*x) * cos(f*y)
    """
    def fx(u, v): return u
    def fy(u, v): return v
    def fz(u, v): return amplitud * np.sin(frecuencia * u) * np.cos(frecuencia * v)
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (-size, size), (-size, size),
        resolution
    )


def generar_helice(radio=1.0, paso=0.5, vueltas=3, resolution=100):
    """
    Genera una hélice 3D: r(t) = (R*cos(t), R*sin(t), paso*t)
    
    Args:
        radio: Radio de la hélice
        paso: Distancia vertical por vuelta
        vueltas: Número de vueltas
        resolution: Puntos por vuelta
    """
    t = np.linspace(0, 2 * np.pi * vueltas, resolution * vueltas)
    
    x = radio * np.cos(t)
    y = radio * np.sin(t)
    z = paso * t / (2 * np.pi)
    
    # Para una curva, crear un tubo alrededor
    return _crear_tubo_desde_curva(x, y, z, radio_tubo=0.1, resolution_tubo=12)


def generar_espiral_conica(radio_base=2.0, altura=3.0, vueltas=4, resolution=150):
    """
    Genera una espiral cónica (como un resorte que se estrecha).
    """
    t = np.linspace(0, 2 * np.pi * vueltas, resolution * vueltas)
    
    # Radio decrece con la altura
    r = radio_base * (1 - t / (2 * np.pi * vueltas))
    
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = altura * t / (2 * np.pi * vueltas)
    
    return _crear_tubo_desde_curva(x, y, z, radio_tubo=0.08, resolution_tubo=10)


def generar_mobius(radio=1.0, ancho=0.4, resolution=60):
    """
    Genera una banda de Möbius.
    Superficie no orientable con un solo lado.
    """
    def fx(u, v):
        return (radio + v * np.cos(u/2)) * np.cos(u)
    
    def fy(u, v):
        return (radio + v * np.cos(u/2)) * np.sin(u)
    
    def fz(u, v):
        return v * np.sin(u/2)
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (0, 2 * np.pi), (-ancho/2, ancho/2),
        resolution
    )


def generar_klein_bottle(resolution=50):
    """
    Genera una botella de Klein (versión inmersión en 3D).
    Superficie no orientable sin borde.
    """
    def fx(u, v):
        return (2 + np.cos(v/2) * np.sin(u) - np.sin(v/2) * np.sin(2*u)) * np.cos(v)
    
    def fy(u, v):
        return (2 + np.cos(v/2) * np.sin(u) - np.sin(v/2) * np.sin(2*u)) * np.sin(v)
    
    def fz(u, v):
        return np.sin(v/2) * np.sin(u) + np.cos(v/2) * np.sin(2*u)
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (0, 2 * np.pi), (0, 2 * np.pi),
        resolution
    )


def generar_toro_anudado(p=2, q=3, radio_mayor=1.0, radio_menor=0.3, resolution=100):
    """
    Genera un nudo toroidal (p,q).
    
    Args:
        p: Número de veces que rodea el eje de revolución
        q: Número de veces que rodea el eje del toro
        radio_mayor: Radio del toro
        radio_menor: Radio del tubo
    """
    t = np.linspace(0, 2 * np.pi, resolution * max(p, q))
    
    r = np.cos(q * t) + 2
    x = r * np.cos(p * t)
    y = r * np.sin(p * t)
    z = -np.sin(q * t)
    
    # Escalar
    x *= radio_mayor / 3
    y *= radio_mayor / 3
    z *= radio_mayor / 3
    
    return _crear_tubo_desde_curva(x, y, z, radio_tubo=radio_menor, resolution_tubo=16)


def generar_superficie_custom(ecuacion_x, ecuacion_y, ecuacion_z, 
                               u_min=-2, u_max=2, v_min=-2, v_max=2,
                               resolution=40):
    """
    Genera una superficie desde ecuaciones en string.
    Usa eval() con un namespace seguro.
    
    Args:
        ecuacion_x: String con ecuación para X (ej: "u * np.cos(v)")
        ecuacion_y: String con ecuación para Y
        ecuacion_z: String con ecuación para Z
        u_min, u_max: Rango del parámetro u
        v_min, v_max: Rango del parámetro v
        resolution: Resolución de la malla
        
    Returns:
        Tuple (vertices, faces)
    """
    # Namespace seguro para evaluación
    safe_namespace = {
        'np': np,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi,
        'e': np.e,
    }
    
    try:
        # Crear funciones desde strings
        def fx(u, v):
            safe_namespace['u'] = u
            safe_namespace['v'] = v
            return eval(ecuacion_x, {"__builtins__": {}}, safe_namespace)
        
        def fy(u, v):
            safe_namespace['u'] = u
            safe_namespace['v'] = v
            return eval(ecuacion_y, {"__builtins__": {}}, safe_namespace)
        
        def fz(u, v):
            safe_namespace['u'] = u
            safe_namespace['v'] = v
            return eval(ecuacion_z, {"__builtins__": {}}, safe_namespace)
        
        return evaluar_superficie_parametrica(
            fx, fy, fz,
            (u_min, u_max), (v_min, v_max),
            resolution
        )
    except Exception as e:
        print(f"Error evaluando ecuación: {e}")
        return np.array([]), []


def generar_funcion_z(ecuacion_z, x_min=-2, x_max=2, y_min=-2, y_max=2, resolution=40):
    """
    Genera una superficie z = f(x, y) desde una ecuación en string.
    
    Args:
        ecuacion_z: String con ecuación (ej: "x**2 + y**2", "sin(x)*cos(y)")
        x_min, x_max: Rango de X
        y_min, y_max: Rango de Y
        resolution: Resolución
        
    Returns:
        Tuple (vertices, faces)
    """
    safe_namespace = {
        'np': np,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi,
        'e': np.e,
    }
    
    try:
        def fz(x, y):
            safe_namespace['x'] = x
            safe_namespace['y'] = y
            return eval(ecuacion_z, {"__builtins__": {}}, safe_namespace)
        
        return evaluar_superficie_parametrica(
            lambda u, v: u,
            lambda u, v: v,
            fz,
            (x_min, x_max), (y_min, y_max),
            resolution
        )
    except Exception as e:
        print(f"Error evaluando ecuación: {e}")
        return np.array([]), []


# ============ FUNCIONES AUXILIARES ============

def _filtrar_superficie(vertices, faces, mask):
    """Filtra vértices y reconstruye caras."""
    # Mapeo de índices viejos a nuevos
    new_indices = np.full(len(vertices), -1)
    new_vertices = vertices[mask]
    new_indices[mask] = np.arange(len(new_vertices))
    
    # Filtrar caras
    new_faces = []
    for face in faces:
        if all(mask[i] for i in face):
            new_face = [new_indices[i] for i in face]
            new_faces.append(new_face)
    
    return new_vertices, new_faces


def _crear_tubo_desde_curva(x, y, z, radio_tubo=0.1, resolution_tubo=12):
    """
    Crea un tubo 3D alrededor de una curva.
    Útil para visualizar hélices y curvas.
    """
    n_points = len(x)
    vertices = []
    faces = []
    
    # Calcular vectores tangentes
    for i in range(n_points):
        # Punto actual
        p = np.array([x[i], y[i], z[i]])
        
        # Vector tangente (aproximado)
        if i == 0:
            t = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
        elif i == n_points - 1:
            t = np.array([x[-1] - x[-2], y[-1] - y[-2], z[-1] - z[-2]])
        else:
            t = np.array([x[i+1] - x[i-1], y[i+1] - y[i-1], z[i+1] - z[i-1]])
        
        t = t / (np.linalg.norm(t) + 1e-10)
        
        # Vectores perpendiculares (normal y binormal)
        if abs(t[2]) < 0.9:
            n = np.cross(t, np.array([0, 0, 1]))
        else:
            n = np.cross(t, np.array([1, 0, 0]))
        n = n / (np.linalg.norm(n) + 1e-10)
        
        b = np.cross(t, n)
        
        # Crear círculo alrededor del punto
        for j in range(resolution_tubo):
            theta = 2 * np.pi * j / resolution_tubo
            offset = radio_tubo * (np.cos(theta) * n + np.sin(theta) * b)
            vertices.append(p + offset)
    
    vertices = np.array(vertices)
    
    # Crear caras
    for i in range(n_points - 1):
        for j in range(resolution_tubo):
            j_next = (j + 1) % resolution_tubo
            
            p1 = i * resolution_tubo + j
            p2 = i * resolution_tubo + j_next
            p3 = (i + 1) * resolution_tubo + j_next
            p4 = (i + 1) * resolution_tubo + j
            
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])
    
    return vertices, faces


def crear_mesh_plotly(vertices, faces, color='#9b59b6', opacity=0.85, name='Paramétrica'):
    """Crea diccionario compatible con Plotly Mesh3d."""
    if len(vertices) == 0 or len(faces) == 0:
        return {}
    
    faces_array = np.array(faces)
    
    return {
        'x': vertices[:, 0].tolist(),
        'y': vertices[:, 1].tolist(),
        'z': vertices[:, 2].tolist(),
        'i': faces_array[:, 0].tolist(),
        'j': faces_array[:, 1].tolist(),
        'k': faces_array[:, 2].tolist(),
        'color': color,
        'opacity': opacity,
        'name': name,
        'flatshading': False,  # Suave para superficies matemáticas
    }


# Catálogo de superficies disponibles
SUPERFICIES = {
    'paraboloide': generar_paraboloide,
    'silla_montar': generar_silla_montar,
    'onda_seno': generar_onda_seno,
    'helice': generar_helice,
    'espiral_conica': generar_espiral_conica,
    'mobius': generar_mobius,
    'klein': generar_klein_bottle,
    'toro_anudado': generar_toro_anudado,
}


# Ejemplos de fórmulas para el usuario
EJEMPLOS_FORMULAS = """
# ═══════════════════════════════════════════════
# 📐 SUPERFICIES PARAMÉTRICAS - Fórmulas
# ═══════════════════════════════════════════════

# Superficie simple z = f(x,y):
# Paraboloide:     z = x² + y²
# Silla montar:    z = x² - y²
# Onda:            z = sin(x) * cos(y)
# Gaussiana:       z = exp(-(x² + y²))

# Superficies paramétricas r(u,v) = (x, y, z):
# Esfera:          x = cos(u)*sin(v), y = sin(u)*sin(v), z = cos(v)
# Toro:            x = (R + r*cos(v))*cos(u), y = (R + r*cos(v))*sin(u), z = r*sin(v)
# Möbius:          x = (1 + v*cos(u/2))*cos(u), y = (1 + v*cos(u/2))*sin(u), z = v*sin(u/2)

# Curvas 3D r(t) = (x, y, z):
# Hélice:          x = cos(t), y = sin(t), z = t
# Espiral:         x = t*cos(t), y = t*sin(t), z = t
# Nudo trebol:     x = sin(t) + 2*sin(2t), y = cos(t) - 2*cos(2t), z = -sin(3t)
"""
