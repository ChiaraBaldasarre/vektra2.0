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


def generar_cilindro(radio=1.0, altura=2.0, resolution=40):
    """
    Genera un cilindro: x = R*cos(u), y = R*sin(u), z = v
    """
    def fx(u, v): return radio * np.cos(u)
    def fy(u, v): return radio * np.sin(u)
    def fz(u, v): return v
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (0, 2 * np.pi), (-altura/2, altura/2),
        resolution
    )


def generar_cono_parametrico(radio_base=1.0, altura=2.0, resolution=40):
    """
    Genera un cono paramétrico: x = v*cos(u), y = v*sin(u), z = v
    """
    def fx(u, v): return v * np.cos(u)
    def fy(u, v): return v * np.sin(u)
    def fz(u, v): return v
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (0, 2 * np.pi), (0, altura),
        resolution
    )


def generar_toro(radio_mayor=1.0, radio_menor=0.4, resolution=40):
    """
    Genera un toro (dónut).
    r(u,v) = ((R + r*cos(v))*cos(u), (R + r*cos(v))*sin(u), r*sin(v))
    """
    def fx(u, v): return (radio_mayor + radio_menor * np.cos(v)) * np.cos(u)
    def fy(u, v): return (radio_mayor + radio_menor * np.cos(v)) * np.sin(u)
    def fz(u, v): return radio_menor * np.sin(v)
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (0, 2 * np.pi), (0, 2 * np.pi),
        resolution
    )


def generar_pseudoesfera(altura=2.0, resolution=40):
    """
    Genera una pseudoesfera (superficie de revolución hiperbólica).
    x = cos(u)*sinh(v), y = sin(u)*sinh(v), z = v - cosh(v)
    """
    def fx(u, v): return np.cos(u) * np.sinh(v)
    def fy(u, v): return np.sin(u) * np.sinh(v)
    def fz(u, v): return v - np.cosh(v)
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (0, 2 * np.pi), (0, altura),
        resolution
    )


def generar_enneper(size=1.5, resolution=50):
    """
    Genera la superficie de Enneper (minimal surface).
    x = u - u³/3 + u*v²
    y = v - v³/3 + v*u²
    z = u² - v²
    """
    def fx(u, v): return u - (u**3)/3 + u*(v**2)
    def fy(u, v): return v - (v**3)/3 + v*(u**2)
    def fz(u, v): return u**2 - v**2
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (-size, size), (-size, size),
        resolution
    )


def generar_catalan(resolution=50):
    """
    Genera la superficie de Catalan (minimal surface).
    x = u - sin(u)*cosh(v)
    y = 1 - cos(u)*cosh(v)
    z = 4*sin(u/2)*sinh(v/2)
    """
    def fx(u, v): return u - np.sin(u)*np.cosh(v)
    def fy(u, v): return 1 - np.cos(u)*np.cosh(v)
    def fz(u, v): return 4*np.sin(u/2)*np.sinh(v/2)
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (0, 2*np.pi), (-1, 1),
        resolution
    )


def generar_hiperboloide(a=1.0, b=1.0, c=1.0, altura=2.0, resolution=40):
    """
    Genera un hiperboloide de una hoja.
    x = a*cos(u)*cosh(v), y = b*sin(u)*cosh(v), z = c*sinh(v)
    """
    def fx(u, v): return a * np.cos(u) * np.cosh(v)
    def fy(u, v): return b * np.sin(u) * np.cosh(v)
    def fz(u, v): return c * np.sinh(v)
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (0, 2 * np.pi), (-altura/2, altura/2),
        resolution
    )


def generar_helicoide(radio=1.0, paso=0.3, vueltas=3, resolution=50):
    """
    Genera una superficie helicoidal (tornillo 3D).
    x = u*cos(v), y = u*sin(v), z = paso*v
    """
    def fx(u, v): return u * np.cos(v)
    def fy(u, v): return u * np.sin(v)
    def fz(u, v): return paso * v
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (0, radio), (0, 2*np.pi*vueltas),
        resolution
    )


def generar_vela(resolution=50):
    """
    Genera la superficie Vela (Sail Surface).
    x = u*cos(v), y = u*sin(v), z = ln(u)
    """
    def fx(u, v): return u * np.cos(v)
    def fy(u, v): return u * np.sin(v)
    def fz(u, v): return np.log(u + 0.1)  # +0.1 para evitar log(0)
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (0.1, 2.0), (0, 2*np.pi),
        resolution
    )


def generar_romboidal(resolution=50):
    """
    Genera una superficie romboidal.
    x = sin(u)*cos(v), y = cos(u)*cos(v), z = sin(v) + u/π
    """
    def fx(u, v): return np.sin(u) * np.cos(v)
    def fy(u, v): return np.cos(u) * np.cos(v)
    def fz(u, v): return np.sin(v) + u / np.pi
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (0, 2*np.pi), (0, 2*np.pi),
        resolution
    )


def generar_catenoide(radio=1.0, altura=3.0, resolution=50):
    """
    Genera un catenoide (minimal surface of revolution).
    x = a*cosh(v/a)*cos(u)
    y = a*cosh(v/a)*sin(u)
    z = v
    """
    def fx(u, v): return radio * np.cosh(v/radio) * np.cos(u)
    def fy(u, v): return radio * np.cosh(v/radio) * np.sin(u)
    def fz(u, v): return v
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (0, 2*np.pi), (-altura/2, altura/2),
        resolution
    )


def generar_ondulatoria_parametrica(amplitud=1.0, freq_n=2, freq_m=3, resolution=50):
    """
    Genera una superficie ondulatoria paramétrica.
    x = u*cos(v), y = u*sin(v), z = A*sin(n*v)*sin(m*u)
    """
    def fx(u, v): return u * np.cos(v)
    def fy(u, v): return u * np.sin(v)
    def fz(u, v): return amplitud * np.sin(freq_n * v) * np.sin(freq_m * u)
    
    return evaluar_superficie_parametrica(
        fx, fy, fz,
        (-2, 2), (0, 2*np.pi),
        resolution
    )


def generar_nudo_trebol(resolution=200):
    """
    Genera un nudo de trebol (como tubo 3D).
    x = sin(t) + 2*sin(2t)
    y = cos(t) - 2*cos(2t)
    z = -sin(3t)
    """
    t = np.linspace(0, 2*np.pi, resolution)
    
    x = np.sin(t) + 2*np.sin(2*t)
    y = np.cos(t) - 2*np.cos(2*t)
    z = -np.sin(3*t)
    
    return _crear_tubo_desde_curva(x, y, z, radio_tubo=0.15, resolution_tubo=12)


def generar_nudo_figura_ocho(resolution=200):
    """
    Genera un nudo figura-ocho (como tubo 3D).
    x = (2 + cos(2t))*cos(3t)
    y = (2 + cos(2t))*sin(3t)
    z = sin(4t)
    """
    t = np.linspace(0, 2*np.pi, resolution)
    
    x = (2 + np.cos(2*t)) * np.cos(3*t)
    y = (2 + np.cos(2*t)) * np.sin(3*t)
    z = np.sin(4*t)
    
    return _crear_tubo_desde_curva(x, y, z, radio_tubo=0.12, resolution_tubo=12)


def generar_espiral_toroidal(resolution=200):
    """
    Genera una espiral toroidal (como tubo 3D).
    x = (2 + cos(5t))*cos(t)
    y = (2 + cos(5t))*sin(t)
    z = sin(5t)
    """
    t = np.linspace(0, 2*np.pi, resolution)
    
    x = (2 + np.cos(5*t)) * np.cos(t)
    y = (2 + np.cos(5*t)) * np.sin(t)
    z = np.sin(5*t)
    
    return _crear_tubo_desde_curva(x, y, z, radio_tubo=0.1, resolution_tubo=10)


def generar_hipocicloide(R=5.0, r=3.0, resolution=200):
    """
    Genera una hipocicloide 3D (como tubo 3D).
    x = (R-r)*cos(t) + r*cos((R-r)*t/r)
    y = (R-r)*sin(t) - r*sin((R-r)*t/r)
    z = sin(5t)
    """
    t = np.linspace(0, 2*np.pi*max(R,r), int(resolution*max(R,r)//4))
    
    x = (R-r)*np.cos(t) + r*np.cos((R-r)*t/r)
    y = (R-r)*np.sin(t) - r*np.sin((R-r)*t/r)
    z = np.sin(5*t)
    
    return _crear_tubo_desde_curva(x, y, z, radio_tubo=0.1, resolution_tubo=10)


def generar_epicicloide(R=5.0, r=2.0, k=3, resolution=200):
    """
    Genera una epicicloide 3D (como tubo 3D).
    x = (R+r)*cos(t) - r*cos((R+r)*t/r)
    y = (R+r)*sin(t) - r*sin((R+r)*t/r)
    z = sin(k*t)
    """
    t = np.linspace(0, 2*np.pi*max(R,r), int(resolution*max(R,r)//4))
    
    x = (R+r)*np.cos(t) - r*np.cos((R+r)*t/r)
    y = (R+r)*np.sin(t) - r*np.sin((R+r)*t/r)
    z = np.sin(k*t)
    
    return _crear_tubo_desde_curva(x, y, z, radio_tubo=0.1, resolution_tubo=10)


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
        # Funciones hiperbólicas
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        # Funciones trigonométricas inversas
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'atan2': np.arctan2,
        # Otras funciones útiles
        'power': np.power,
        'floor': np.floor,
        'ceil': np.ceil,
        'sign': np.sign,
        'mod': np.mod,
        'maximum': np.maximum,
        'minimum': np.minimum,
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
        # Funciones hiperbólicas
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        # Funciones trigonométricas inversas
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'atan2': np.arctan2,
        # Otras funciones útiles
        'power': np.power,
        'floor': np.floor,
        'ceil': np.ceil,
        'sign': np.sign,
        'mod': np.mod,
        'maximum': np.maximum,
        'minimum': np.minimum,
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
    # Superficies clásicas z = f(x,y)
    'paraboloide': generar_paraboloide,
    'silla_montar': generar_silla_montar,
    'onda_seno': generar_onda_seno,
    
    # Superficies paramétricas básicas
    'cilindro': generar_cilindro,
    'cono': generar_cono_parametrico,
    'toro': generar_toro,
    'pseudoesfera': generar_pseudoesfera,
    'enneper': generar_enneper,
    'catalan': generar_catalan,
    'hiperboloide': generar_hiperboloide,
    'helicoide': generar_helicoide,
    'vela': generar_vela,
    'romboidal': generar_romboidal,
    'catenoide': generar_catenoide,
    'ondulatoria': generar_ondulatoria_parametrica,
    
    # Curvas 3D (como tubos)
    'helice': generar_helice,
    'espiral_conica': generar_espiral_conica,
    'nudo_trebol': generar_nudo_trebol,
    'nudo_figura_ocho': generar_nudo_figura_ocho,
    'espiral_toroidal': generar_espiral_toroidal,
    'hipocicloide': generar_hipocicloide,
    'epicicloide': generar_epicicloide,
    
    # Superficies especiales
    'mobius': generar_mobius,
    'klein': generar_klein_bottle,
    'toro_anudado': generar_toro_anudado,
}


# Ejemplos de fórmulas para el usuario
EJEMPLOS_FORMULAS = """
╔════════════════════════════════════════════════════════════════╗
║          📐 SUPERFICIES MATEMÁTICAS - FÓRMULAS VECTORIALES     ║
╚════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SUPERFICIES CLÁSICAS z = f(x,y)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Paraboloide:          z = x² + y²
• Silla de Montar:      z = x² - y²
• Onda Bidimensional:   z = sin(x) * cos(y)
• Ondas Amortiguadas:   z = exp(-√(x²+y²)) * sin(5*√(x²+y²))
• Campana Gaussiana:    z = exp(-(x² + y²))
• Cono:                 z = √(x² + y²)
• Hiperbólica:          z = tanh(x) * tanh(y)
• Ondulatoria Compleja: z = sin(x) * sin(y) + cos(2x) * cos(2y)
• Roseta Polar:         z = sin(3*atan2(y,x)) * (x² + y²)
• Senoidal 3D Radial:   z = sin(√(x²+y²)) * cos(√(x²+y²))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. SUPERFICIES PARAMÉTRICAS r(u,v) = (x, y, z)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• CILINDRO:
  x = R*cos(u),  y = R*sin(u),  z = v
  u ∈ [0, 2π],  v ∈ [-h, h]

• CONO PARAMÉTRICO:
  x = v*cos(u),  y = v*sin(u),  z = v
  u ∈ [0, 2π],  v ∈ [0, H]

• TORO (Toroide):
  x = (R + r*cos(v))*cos(u)
  y = (R + r*cos(v))*sin(u)
  z = r*sin(v)
  u,v ∈ [0, 2π]  (R=radio mayor, r=radio menor)

• PSEUDOESFERA:
  x = cos(u)*sinh(v)
  y = sin(u)*sinh(v)
  z = v - cosh(v)
  u ∈ [0, 2π],  v ∈ [0, h]

• SUPERFICIE DE ENNEPER:
  x = u - u³/3 + u*v²
  y = v - v³/3 + v*u²
  z = u² - v²
  u,v ∈ [-1.5, 1.5]

• SUPERFICIE DE CATALAN:
  x = u - sin(u)*cosh(v)
  y = 1 - cos(u)*cosh(v)
  z = 4*sin(u/2)*sinh(v/2)

• HIPERBOLOIDE DE UNA HOJA:
  x = a*cos(u)*cosh(v)
  y = b*sin(u)*cosh(v)
  z = c*sinh(v)
  u ∈ [0, 2π],  v ∈ [-h, h]

• HELICOIDE (Tornillo 3D):
  x = u*cos(v),  y = u*sin(v),  z = a*v
  u ∈ [0, 2π],  v ∈ [0, 2π]

• VELA (Sail Surface):
  x = u*cos(v),  y = u*sin(v),  z = ln(u)
  u ∈ [0.1, 2π],  v ∈ [0, 2π]

• SUPERFICIE ROMBOIDAL:
  x = sin(u)*cos(v)
  y = cos(u)*cos(v)
  z = sin(v) + u/π

• BANDA DE MÖBIUS:
  x = (R + v*cos(u/2))*cos(u)
  y = (R + v*cos(u/2))*sin(u)
  z = v*sin(u/2)
  (Superficie no orientable con un solo lado)

• BOTELLA DE KLEIN:
  x = (2 + cos(v/2)*sin(u) - sin(v/2)*sin(2u))*cos(v)
  y = (2 + cos(v/2)*sin(u) - sin(v/2)*sin(2u))*sin(v)
  z = sin(v/2)*sin(u) + cos(v/2)*sin(2u)
  (Superficie no orientable sin borde)

• CATENOIDE (Minimal):
  x = a*cosh(v/a)*cos(u)
  y = a*cosh(v/a)*sin(u)
  z = v

• SUPERFICIE ONDULATORIA PARAMÉTRICA:
  x = u*cos(v),  y = u*sin(v),  z = A*sin(n*v)*sin(m*u)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. CURVAS 3D r(t) = (x, y, z) - Para Tubos y Filamentos
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• HÉLICE CLÁSICA:
  x = cos(t),  y = sin(t),  z = t

• ESPIRAL LOGARÍTMICA:
  x = t*cos(t),  y = t*sin(t),  z = t

• NUDO DE TREBOL:
  x = sin(t) + 2*sin(2t)
  y = cos(t) - 2*cos(2t)
  z = -sin(3t)

• NUDO FIGURA-OCHO:
  x = (2 + cos(2t))*cos(3t)
  y = (2 + cos(2t))*sin(3t)
  z = sin(4t)

• ESPIRAL TOROIDAL:
  x = (2 + cos(5t))*cos(t)
  y = (2 + cos(5t))*sin(t)
  z = sin(5t)

• HIPOCICLOIDE 3D:
  x = (R-r)*cos(t) + r*cos((R-r)*t/r)
  y = (R-r)*sin(t) - r*sin((R-r)*t/r)
  z = sin(5t)

• EPICICLOIDE 3D:
  x = (R+r)*cos(t) - r*cos((R+r)*t/r)
  y = (R+r)*sin(t) - r*sin((R+r)*t/r)
  z = sin(k*t)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. FÓRMULAS PERSONALIZADAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Puedes usar generar_funcion_z() o generar_superficie_custom() con:
• Variables: x, y, u, v
• Funciones: sin, cos, tan, exp, log, sqrt, abs
• Constantes: pi, e
• Operadores: +, -, *, /, **

EJEMPLOS:
  x**2 + y**2               (Paraboloide)
  sin(x) * cos(y)           (Onda)
  exp(-sqrt(x**2 + y**2))   (Gaussiana)
  sqrt(x**2 + y**2)         (Cono)
"""

# Diccionario de fórmulas concisas por superficie
FORMULAS_SUPERFICIES = {
    "paraboloide": "z = x² + y²",
    "silla_montar": "z = x² - y²",
    "onda_seno": "z = sin(x)cos(y)",
    "cilindro": "x=R·cos(u), y=R·sin(u), z=v",
    "cono": "x=v·cos(u), y=v·sin(u), z=v",
    "toro": "x=(R+r·cos(v))·cos(u), y=(R+r·cos(v))·sin(u), z=r·sin(v)",
    "pseudoesfera": "x=cos(u)·sech(v), y=sin(u)·sech(v), z=v-tanh(v)",
    "enneper": "x=u(1-u²/3+v²)/2, y=v(1-v²/3+u²)/2, z=(u²-v²)/2",
    "catalan": "x=u-sin(u)·cosh(v), y=1-cos(u)·cosh(v), z=4·sin(u/2)·sinh(v/2)",
    "hiperboloide": "x²/a² + y²/b² - z²/c² = 1",
    "helicoide": "x=v·cos(u), y=v·sin(u), z=a·u",
    "vela": "x=u·cos(v), y=u·sin(v), z=ln(u)",
    "romboidal": "x=sin(u)·cos(v), y=cos(u)·cos(v), z=sin(v)+u/π",
    "catenoide": "z = a·ln(tan(θ/2)), catenaria rotativa",
    "ondulatoria": "z = sin(nπx)·sin(mπy)",
    "helice": "x=cos(t), y=sin(t), z=c·t",
    "espiral_conica": "r=θ, z=θ en coord. cilíndricas",
    "nudo_trebol": "x=sin(t)+2·sin(2t), y=cos(t)-2·cos(2t), z=sin(3t)",
    "nudo_figura_ocho": "x=(2+cos(2t))·cos(3t), y=(2+cos(2t))·sin(3t), z=sin(4t)",
    "espiral_toroidal": "x=(2+cos(5t))·cos(t), y=(2+cos(5t))·sin(t), z=sin(5t)",
    "hipocicloide": "x=(R-r)·cos(t)+r·cos((R-r)t/r), y=(R-r)·sin(t)-r·sin((R-r)t/r), z=sin(5t)",
    "epicicloide": "x=(R+r)·cos(t)-r·cos((R+r)t/r), y=(R+r)·sin(t)-r·sin((R+r)t/r), z=sin(kt)",
    "mobius": "x=(R+v·cos(u/2))·cos(u), y=(R+v·cos(u/2))·sin(u), z=v·sin(u/2)",
    "klein": "x=(2+cos(v/2)·sin(u)-sin(v/2)·sin(2u))·cos(v), y=(2+cos(v/2)·sin(u)-sin(v/2)·sin(2u))·sin(v), z=sin(v/2)·sin(u)+cos(v/2)·sin(2u)",
    "toro_anudado": "r=cos(qt)+2, x=r·cos(pt), y=r·sin(pt), z=-sin(qt)"
}

# Ejemplos de fórmulas z = f(x,y)
EJEMPLOS_Z_FXY = {
    "Paraboloide": "x**2 + y**2",
    "Silla de Montar": "x**2 - y**2",
    "Onda 2D": "sin(x) * cos(y)",
    "Gaussiana": "exp(-(x**2 + y**2))",
    "Ondas Radiales": "sin(sqrt(x**2 + y**2))",
    "Crestas": "sin(x) + sin(y)",
    "Senoidal Amortiguada": "exp(-sqrt(x**2 + y**2)) * sin(5*sqrt(x**2 + y**2))",
    "Hiperbólica": "tanh(x) * tanh(y)",
    "Roseta Polar": "sin(3*atan2(y,x)) * (x**2 + y**2)",
    "Montaña": "cos(x) * cos(y)",
    "Campana": "exp(-(x**2 + y**2) / 0.5)",
    "Ondas Cruzadas": "sin(x) + cos(y)",
    "Logarítmica": "log(1 + x**2 + y**2)",
    "Senoidal Compleja": "sin(x) * sin(y) + cos(2*x) * cos(2*y)",
    "Cono Suave": "sqrt(abs(x**2 + y**2))",
    "Seno 3D": "sin(x)*cos(y)*sin(sqrt(x**2+y**2))",
    "Valles": "cos(x*y)",
    "Ondas Amortiguadas": "cos(sqrt(x**2 + y**2)) * exp(-0.1*sqrt(x**2 + y**2))",
    "Función Mezcla": "(x**2 - y**2) / (1 + x**2 + y**2)",
    "Ripples": "sin(2*pi*sqrt(x**2 + y**2))"
}

# Ejemplos de superficies paramétricas r(u,v) = (x,y,z)
EJEMPLOS_PARAMETRICAS = {
    "Esfera": {
        "descripcion": "Esfera de radio 1",
        "x": "cos(u) * sin(v)",
        "y": "sin(u) * sin(v)",
        "z": "cos(v)",
        "u_rango": (0, 2*np.pi),
        "v_rango": (0, np.pi)
    },
    "Toro": {
        "descripcion": "Toro (dónut) clásico",
        "x": "(2 + cos(v)) * cos(u)",
        "y": "(2 + cos(v)) * sin(u)",
        "z": "sin(v)",
        "u_rango": (0, 2*np.pi),
        "v_rango": (0, 2*np.pi)
    },
    "Cono": {
        "descripcion": "Cono paramétrico",
        "x": "u * cos(v)",
        "y": "u * sin(v)",
        "z": "u",
        "u_rango": (0, 2),
        "v_rango": (0, 2*np.pi)
    },
    "Cilindro": {
        "descripcion": "Cilindro de altura 4",
        "x": "cos(u)",
        "y": "sin(u)",
        "z": "v",
        "u_rango": (0, 2*np.pi),
        "v_rango": (-2, 2)
    },
    "Paraboloide Paramétrico": {
        "descripcion": "Paraboloide 3D",
        "x": "u * cos(v)",
        "y": "u * sin(v)",
        "z": "u**2",
        "u_rango": (0, 2),
        "v_rango": (0, 2*np.pi)
    },
    "Hiperboloide": {
        "descripcion": "Hiperboloide de 1 hoja",
        "x": "cosh(v) * cos(u)",
        "y": "cosh(v) * sin(u)",
        "z": "sinh(v)",
        "u_rango": (0, 2*np.pi),
        "v_rango": (-1, 1)
    },
    "Silla de Montar": {
        "descripcion": "Silla hiperbólica en 3D",
        "x": "u",
        "y": "v",
        "z": "u**2 - v**2",
        "u_rango": (-2, 2),
        "v_rango": (-2, 2)
    },
    "Onda 3D": {
        "descripcion": "Superficie ondulada",
        "x": "u",
        "y": "v",
        "z": "sin(u) * cos(v)",
        "u_rango": (-np.pi, np.pi),
        "v_rango": (-np.pi, np.pi)
    },
    "Botella Klein": {
        "descripcion": "Superficie no orientable",
        "x": "(2 + cos(v/2)*sin(u) - sin(v/2)*sin(2*u)) * cos(v)",
        "y": "(2 + cos(v/2)*sin(u) - sin(v/2)*sin(2*u)) * sin(v)",
        "z": "sin(v/2)*sin(u) + cos(v/2)*sin(2*u)",
        "u_rango": (0, 2*np.pi),
        "v_rango": (0, 2*np.pi)
    },
    "Helicoide": {
        "descripcion": "Superficie de tornillo",
        "x": "u * cos(v)",
        "y": "u * sin(v)",
        "z": "0.3 * v",
        "u_rango": (0, 2),
        "v_rango": (0, 4*np.pi)
    },
    "Seno Toroidal": {
        "descripcion": "Toro con ondulación",
        "x": "(2 + 0.5*cos(2*v)) * cos(u)",
        "y": "(2 + 0.5*cos(2*v)) * sin(u)",
        "z": "sin(v) + 0.3*sin(3*u)",
        "u_rango": (0, 2*np.pi),
        "v_rango": (0, 2*np.pi)
    },
    "Onda Senoidal": {
        "descripcion": "Superficie con patrón de onda",
        "x": "u * cos(v)",
        "y": "u * sin(v)",
        "z": "sin(3*v) + 0.2*u",
        "u_rango": (0, 3),
        "v_rango": (0, 2*np.pi)
    }
}
