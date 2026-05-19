"""
Módulo de Funciones Sonoras - Series de Fourier
Calcula las señales periódicas mediante sumatorias trigonométricas.
"""
import numpy as np

def calcular_onda_sonora(tipo_funcion, armonicos, rango_x=10.0, resolucion=600):
    """
    Calcula los puntos X, Y de la serie de Fourier
    """
    x = np.linspace(0, rango_x, resolucion)
    y = np.zeros_like(x)

    # 1. Onda Cuadrada Estándar (k impar)
    if tipo_funcion == "Onda Cuadrada":
        for k in range(1, armonicos + 1):
            coeficiente = 4 / (np.pi * (2 * k - 1))
            frecuencia = 2 * k - 1
            y += coeficiente * np.sin(frecuencia * x)

    # 2. Onda Diente de Sierra (Alternada o Progresiva)
    elif tipo_funcion == "Onda Diente de Sierra":
        for k in range(1, armonicos + 1):
            coeficiente = 2 / (np.pi * k)
            frecuencia = k
            y += coeficiente * np.sin(frecuencia * x)

    # 3. Onda Triangular (Solo armónicos impares con caída cuadrática y signo alternado)
    elif tipo_funcion == "Onda Triangular":
        for k in range(1, armonicos + 1):
            n = 2 * k - 1
            coeficiente = (8 / (np.pi**2)) * ((-1)**(k - 1) / (n**2))
            y += coeficiente * np.sin(n * x)

    # 4. Onda de Tren de Pulsos Modulados
    elif tipo_funcion == "Tren de Pulsos":
        for k in range(1, armonicos + 1):
            coeficiente = 5.0 / (k + 1)
            frecuencia = 4 * k + 1
            y += coeficiente * np.sin(frecuencia * x)

    # 5. Onda de Sierra Armónica Asimétrica
    elif tipo_funcion == "Sierra Asimétrica":
        for k in range(1, armonicos + 1):
            coeficiente = 6.0 / (2 * k + 1)
            frecuencia = 2 * k + 1
            y += coeficiente * np.sin(frecuencia * x)

    # 6. Onda de Pulso Cuadrático Complejo
    elif tipo_funcion == "Pulso Cuadrático":
        for k in range(1, armonicos + 1):
            coeficiente = 4.0 / (k**2 + 1)
            frecuencia = 3 * k - 1
            y += coeficiente * np.sin(frecuencia * x)

    return x, y

def extruir_onda_a_malla_3d(x, y, profundidad=5.0, subdivisiones_z=30):
    """
    Proyecta la curva 2D en una superficie tridimensional continua (Muro de Espectro).
    """
    z_levels = np.linspace(0, profundidad, subdivisiones_z)
    vertices = []
    faces = []
    num_puntos_onda = len(x)

    # Generación de la grilla de vértices
    for z in z_levels:
        for idx in range(num_puntos_onda):
            vertices.append([x[idx], y[idx], z])

    vertices = np.array(vertices)

    # Generación de índices para las caras triangulares del Mesh
    for level in range(subdivisiones_z - 1):
        offset_actual = level * num_puntos_onda
        offset_siguiente = (level + 1) * num_puntos_onda

        for i in range(num_puntos_onda - 1):
            p1 = offset_actual + i
            p2 = offset_actual + i + 1
            p3 = offset_siguiente + i + 1
            p4 = offset_siguiente + i

            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

    return vertices, np.array(faces)