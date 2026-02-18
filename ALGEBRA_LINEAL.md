# 📐 VEKTRA - Documentación sobre la aplicacion de algebra lineal

> Motor de Vectorización de Imágenes y Renderizado Procedural 3D

Este documento describe las funciones del proyecto que implementan conceptos de álgebra lineal, explicando la matemática detrás de cada operación.

---

## 📑 Índice

1. [Conceptos Fundamentales](#1-conceptos-fundamentales)
2. [Módulo de Extrusión](#2-módulo-de-extrusión-extrusionpy)
3. [Módulo de Contornos](#3-módulo-de-contornos-contourspy)
4. [Módulo de Primitivas](#4-módulo-de-primitivas-primitivespy)
5. [Módulo Paramétrico](#5-módulo-paramétrico-parametricpy)
6. [Tabla Resumen](#6-tabla-resumen-de-operaciones)

---

## 1. Conceptos Fundamentales

### 1.1 Vectores en ℝ² y ℝ³

Un **vector** es una entidad matemática con magnitud y dirección. En el proyecto utilizamos:

```
Vector 2D: v = (x, y) ∈ ℝ²
Vector 3D: v = (x, y, z) ∈ ℝ³
```

### 1.2 Producto Punto (Dot Product)

El **producto punto** de dos vectores devuelve un escalar:

```
a · b = a₁b₁ + a₂b₂ + a₃b₃ = |a||b|cos(θ)
```

**Uso en el proyecto:** Calcular ángulos entre vectores tangentes para determinar la curvatura de contornos.

```python
# En calcular_curvatura()
dot = np.dot(v1, v2)
angle = np.arctan2(cross, dot)
```

### 1.3 Producto Cruz (Cross Product)

El **producto cruz** produce un vector perpendicular a ambos operandos:

```
a × b = (a₂b₃ - a₃b₂, a₃b₁ - a₁b₃, a₁b₂ - a₂b₁)
```

**Usos en el proyecto:**
- Calcular normales de caras en mallas 3D
- Determinar el área de triángulos
- Calcular curvatura usando el signo del producto cruz 2D

```python
# Cálculo de normal de una cara triangular
edge1 = v1 - v0
edge2 = v2 - v0
face_normal = np.cross(edge1, edge2)
```

### 1.4 Norma de un Vector

La **norma** (magnitud) de un vector es su longitud:

```
||v|| = √(x² + y² + z²)
```

**Uso:** Normalizar vectores, calcular distancias, verificar si un polígono está cerrado.

```python
distance = np.linalg.norm(first - last)
normalized_vector = vector / np.linalg.norm(vector)
```

### 1.5 Transformaciones Lineales

Las **transformaciones afines** incluyen traslación y escalado:

```
P' = s(P - centro) + traslación
```

---

## 2. Módulo de Extrusión (`extrusion.py`)

Este módulo convierte contornos 2D en mallas 3D.

### 2.1 `normalize_points()` - Normalización de Puntos

**Propósito:** Transforma puntos 2D para centrarlos en el origen y escalarlos a un tamaño objetivo.

**Álgebra Lineal:**

1. **Cálculo de extremos:**
   ```python
   min_x, min_y = points.min(axis=0)
   max_x, max_y = points.max(axis=0)
   ```

2. **Centro del bounding box:**
   ```
   centro_x = min_x + ancho/2
   centro_y = min_y + alto/2
   ```

3. **Factor de escala:**
   ```
   scale = target_size / max(ancho, alto)
   ```

4. **Transformación afín:**
   ```
   P'_x = (P_x - centro_x) × scale
   P'_y = -(P_y - centro_y) × scale
   ```

```python
def normalize_points(points_img, target_size=1.0):
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    width, height = max_x - min_x, max_y - min_y
    
    scale = target_size / max(width, height)
    
    points[:, 0] = (points[:, 0] - (min_x + width/2)) * scale
    points[:, 1] = -(points[:, 1] - (min_y + height/2)) * scale
    
    return points
```

---

### 2.2 `sort_contour_points()` - Ordenamiento Angular

**Propósito:** Ordena los puntos de un contorno angularmente respecto a su centroide.

**Álgebra Lineal:**

1. **Cálculo del centroide:**
   ```
   C = (1/n) × Σ Pᵢ
   ```

2. **Ángulo polar de cada punto:**
   ```
   θᵢ = atan2(Pᵢ_y - C_y, Pᵢ_x - C_x)
   ```

3. **Ordenamiento por ángulo**

```python
def sort_contour_points(points):
    center = np.mean(points, axis=0)  # Centroide
    angles = np.arctan2(
        points[:,1] - center[1],
        points[:,0] - center[0]
    )
    return points[np.argsort(angles)]
```

> **Nota:** `atan2(y, x)` devuelve el ángulo en radianes considerando el cuadrante correcto, a diferencia de `atan(y/x)`.

---

### 2.3 `calcular_normales_vertices()` - Normales de Vértices

**Propósito:** Calcula vectores normales para cada vértice, esenciales para iluminación correcta.

**Álgebra Lineal:**

1. **Vectores de aristas** de un triángulo (V₀, V₁, V₂):
   ```
   edge1 = V₁ - V₀
   edge2 = V₂ - V₀
   ```

2. **Normal de la cara** usando producto cruz:
   ```
   N_cara = edge1 × edge2
   ```

3. **Normalización:**
   ```
   N̂_cara = N_cara / ||N_cara||
   ```

4. **Acumulación:** Cada vértice suma las normales de sus caras adyacentes

5. **Normalización final** del vector suma

```python
def calcular_normales_vertices(vertices, faces):
    normals = np.zeros_like(vertices)
    
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)  # Producto cruz
        
        norm = np.linalg.norm(face_normal)
        if norm > 0:
            face_normal /= norm  # Normalizar
        
        for idx in face:
            normals[idx] += face_normal  # Acumular
    
    # Normalizar todas las normales de vértices
    for i in range(len(normals)):
        norm = np.linalg.norm(normals[i])
        if norm > 0:
            normals[i] /= norm
    
    return normals
```

---

### 2.4 `triangulate_polygon()` - Triangulación de Delaunay

**Propósito:** Divide un polígono 2D (posiblemente cóncavo) en triángulos.

**Álgebra Lineal:**

- **Triangulación de Delaunay:** Maximiza el ángulo mínimo de todos los triángulos, evitando triángulos degenerados.

- **Verificación de punto en polígono (Ray Casting):** Cuenta intersecciones de un rayo con los bordes del polígono.

```python
def triangulate_polygon(points_2d):
    tri = Delaunay(points_2d)
    
    triangles = []
    for simplex in tri.simplices:
        triangle_points = points_2d[simplex]
        centroid = triangle_points.mean(axis=0)  # Promedio vectorial
        
        if is_point_inside_polygon(centroid, points_2d):
            triangles.append(simplex.tolist())
    
    return triangles
```

---

### 2.5 `optimizar_triangulacion()` - Cálculo de Áreas

**Propósito:** Elimina triángulos degenerados (área ≈ 0).

**Álgebra Lineal:**

El área de un triángulo se calcula con el **producto cruz**:

```
Área = ||(P₁ - P₀) × (P₂ - P₀)|| / 2
```

En 2D, el producto cruz da un escalar:

```
Área = |(P₁_x - P₀_x)(P₂_y - P₀_y) - (P₁_y - P₀_y)(P₂_x - P₀_x)| / 2
```

```python
def optimizar_triangulacion(points_2d, faces):
    optimized_faces = []
    
    for face in faces:
        p0, p1, p2 = points_2d[face[0]], points_2d[face[1]], points_2d[face[2]]
        
        # Área usando producto cruz 2D
        area = abs(np.cross(p1 - p0, p2 - p0)) / 2
        
        if area > 1e-8:  # Solo triángulos con área significativa
            optimized_faces.append(face)
    
    return optimized_faces
```

---

### 2.6 `ensure_closed_polygon()` - Verificación de Cierre

**Propósito:** Asegura que el polígono esté cerrado.

**Álgebra Lineal:** Calcula la distancia euclidiana entre el primer y último punto:

```
distancia = ||P_primero - P_último||
```

```python
def ensure_closed_polygon(points, threshold=0.1):
    distance = np.linalg.norm(points[0] - points[-1])
    
    if distance > threshold:
        return np.vstack([points, [points[0]]])
    
    return points
```

---

## 3. Módulo de Contornos (`contours.py`)

Procesa contornos extraídos de imágenes.

### 3.1 `calcular_curvatura()` - Curvatura Local

**Propósito:** Calcula la curvatura en cada punto del contorno para identificar esquinas.

**Álgebra Lineal:**

1. **Vectores tangentes:**
   ```
   v₁ = Pᵢ - Pᵢ₋ₖ  (hacia atrás)
   v₂ = Pᵢ₊ₖ - Pᵢ  (hacia adelante)
   ```

2. **Producto cruz 2D** (da el sentido del giro):
   ```
   cross = v₁_x × v₂_y - v₁_y × v₂_x
   ```

3. **Producto punto** (magnitud del ángulo):
   ```
   dot = v₁ · v₂
   ```

4. **Ángulo de curvatura:**
   ```
   θ = atan2(cross, dot)
   curvatura = |θ|
   ```

```python
def calcular_curvatura(points, window=5):
    n = len(points)
    curvatures = np.zeros(n)
    
    for i in range(n):
        prev_idx = (i - window) % n
        next_idx = (i + window) % n
        
        v1 = points[i] - points[prev_idx]
        v2 = points[next_idx] - points[i]
        
        cross = np.cross(v1, v2)  # Producto cruz 2D
        dot = np.dot(v1, v2)       # Producto punto
        
        angle = np.arctan2(cross, dot)
        curvatures[i] = abs(angle)
    
    return curvatures
```

---

### 3.2 `remuestrear_contorno()` - Interpolación Lineal

**Propósito:** Redistribuye los puntos uniformemente a lo largo del contorno.

**Álgebra Lineal:**

1. **Distancia entre puntos consecutivos** (norma euclidiana):
   ```
   dᵢ = ||Pᵢ₊₁ - Pᵢ||
   ```

2. **Longitud acumulada:**
   ```
   Lᵢ = Σⱼ₌₀ⁱ dⱼ
   ```

3. **Interpolación lineal** entre Pᵢ y Pᵢ₊₁:
   ```
   P_nuevo = Pᵢ + t × (Pᵢ₊₁ - Pᵢ)    donde t ∈ [0, 1]
   ```

```python
def remuestrear_contorno(points, num_points):
    # Calcular distancias entre puntos consecutivos
    diff = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))  # ||Pᵢ₊₁ - Pᵢ||
    
    # Longitud acumulada
    cumulative = np.zeros(len(points))
    cumulative[1:] = np.cumsum(distances)
    
    total_length = cumulative[-1]
    
    # Puntos uniformemente espaciados
    target_distances = np.linspace(0, total_length, num_points, endpoint=False)
    
    new_points = np.zeros((num_points, 2))
    
    for i, target in enumerate(target_distances):
        idx = np.searchsorted(cumulative, target, side='right') - 1
        
        # Interpolación lineal
        t = (target - cumulative[idx]) / (cumulative[idx + 1] - cumulative[idx])
        new_points[i] = points[idx] + t * (points[idx + 1] - points[idx])
    
    return new_points
```

---

### 3.3 `suavizar_contorno_media_movil()` - Filtro de Media

**Propósito:** Reduce el ruido calculando el promedio de puntos vecinos.

**Álgebra Lineal:**

Para cada punto Pᵢ, calcula el **promedio vectorial** de una ventana:

```
P'ᵢ = (1/w) × Σⱼ₌₋ᵥ/₂ᵛ/² Pᵢ₊ⱼ
```

Esta es una **convolución discreta** con un kernel uniforme.

```python
def suavizar_contorno_media_movil(points, window=3):
    n = len(points)
    smoothed = np.zeros_like(points, dtype=np.float64)
    
    for i in range(n):
        indices = [(i + j) % n for j in range(-window//2, window//2 + 1)]
        smoothed[i] = np.mean(points[indices], axis=0)  # Promedio vectorial
    
    return smoothed
```

---

## 4. Módulo de Primitivas (`primitives.py`)

Genera figuras geométricas 3D básicas.

### 4.1 `get_sphere()` - Coordenadas Esféricas

**Propósito:** Genera una esfera mediante parametrización esférica.

**Transformación de Coordenadas:**

Las **coordenadas esféricas** (r, φ, θ) se transforman a cartesianas (x, y, z):

```
x = r × sin(φ) × cos(θ)
y = r × sin(φ) × sin(θ)
z = r × cos(φ)
```

Donde:
- **φ (phi):** ángulo polar, de 0 a π
- **θ (theta):** ángulo azimutal, de 0 a 2π
- **r:** radio (1 para esfera unitaria)

```python
def get_sphere(n=30):
    phi = np.linspace(0, np.pi, n)
    theta = np.linspace(0, 2*np.pi, n)
    phi, theta = np.meshgrid(phi, theta)  # Malla 2D de parámetros

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    return vertices, faces
```

---

### 4.2 `get_cylinder()` - Coordenadas Cilíndricas

**Transformación de Coordenadas:**

Las **coordenadas cilíndricas** (r, θ, z) se transforman a cartesianas:

```
x = r × cos(θ)
y = r × sin(θ)
z = z
```

```python
def get_cylinder(n=32):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    x = np.cos(theta)  # r = 1
    y = np.sin(theta)

    top = np.column_stack([x, y, np.ones(n)])   # z = 1
    bot = np.column_stack([x, y, -np.ones(n)])  # z = -1

    vertices = np.vstack([top, bot])
    return vertices, faces
```

---

## 5. Módulo Paramétrico (`parametric.py`)

Genera superficies 3D a partir de ecuaciones matemáticas.

### 5.1 `evaluar_superficie_parametrica()` - Parametrización

**Propósito:** Evalúa una superficie paramétrica definida por tres funciones.

**Álgebra Lineal:**

Una **superficie paramétrica** es un mapeo:

```
r: ℝ² → ℝ³
r(u, v) = (f_x(u,v), f_y(u,v), f_z(u,v))
```

**Algoritmo:**

1. Crear **malla de parámetros** (u, v) con `meshgrid`
2. Evaluar las funciones componentes
3. Construir la **matriz de vértices** (n² × 3)
4. Generar las **caras triangulares**

```python
def evaluar_superficie_parametrica(func_x, func_y, func_z, u_range, v_range, resolution=50):
    u = np.linspace(u_range[0], u_range[1], resolution)
    v = np.linspace(v_range[0], v_range[1], resolution)
    U, V = np.meshgrid(u, v)  # Malla 2D
    
    X = func_x(U, V)
    Y = func_y(U, V)
    Z = func_z(U, V)
    
    vertices = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    # Generar caras triangulares
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            p1 = i * resolution + j
            p2 = p1 + 1
            p3 = p1 + resolution + 1
            p4 = p1 + resolution
            
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])
    
    return vertices, faces
```

---

### 5.2 Ejemplos de Superficies Paramétricas

| Superficie | Ecuaciones |
|------------|------------|
| **Paraboloide** | x=u, y=v, z=(u/a)² + (v/b)² |
| **Silla de Montar** | x=u, y=v, z=(u/a)² - (v/b)² |
| **Toro** | x=(R+r·cos(v))·cos(u), y=(R+r·cos(v))·sin(u), z=r·sin(v) |
| **Banda de Möbius** | x=(1+v·cos(u/2))·cos(u), y=(1+v·cos(u/2))·sin(u), z=v·sin(u/2) |
| **Hélice** | x=R·cos(t), y=R·sin(t), z=p·t |
| **Enneper** | x=u-u³/3+uv², y=v-v³/3+vu², z=u²-v² |
| **Catenoide** | x=a·cosh(v/a)·cos(u), y=a·cosh(v/a)·sin(u), z=v |

---

### 5.3 `_crear_tubo_desde_curva()` - Marco de Frenet-Serret

**Propósito:** Crea un tubo 3D alrededor de una curva espacial (para hélices y nudos).

**Álgebra Lineal:**

Para cada punto de la curva se construye un **marco ortonormal** (T, N, B):

1. **Vector Tangente (T):** Derivada normalizada
   ```
   T = r'(t) / ||r'(t)||
   ```

2. **Vector Normal (N):** Perpendicular a T
   ```
   N = T × [0,0,1]  (o [1,0,0] si T es casi vertical)
   N = N / ||N||
   ```

3. **Vector Binormal (B):** Completa el sistema
   ```
   B = T × N
   ```

4. **Punto en el tubo:**
   ```
   P_tubo = P_curva + r × (cos(θ) × N + sin(θ) × B)
   ```

```python
def _crear_tubo_desde_curva(x, y, z, radio_tubo=0.1, resolution_tubo=12):
    for i in range(n_points):
        p = np.array([x[i], y[i], z[i]])
        
        # Vector tangente (aproximado por diferencias finitas)
        t = np.array([x[i+1] - x[i-1], y[i+1] - y[i-1], z[i+1] - z[i-1]])
        t = t / np.linalg.norm(t)  # Normalizar
        
        # Vector normal (producto cruz con eje Z)
        n = np.cross(t, np.array([0, 0, 1]))
        n = n / np.linalg.norm(n)
        
        # Vector binormal
        b = np.cross(t, n)
        
        # Crear círculo alrededor del punto
        for j in range(resolution_tubo):
            theta = 2 * np.pi * j / resolution_tubo
            offset = radio_tubo * (np.cos(theta) * n + np.sin(theta) * b)
            vertices.append(p + offset)
```

---

## 6. Tabla Resumen de Operaciones

| Operación | Función NumPy | Uso en el Proyecto |
|-----------|---------------|-------------------|
| **Producto Punto** | `np.dot(a, b)` | Ángulos entre vectores |
| **Producto Cruz** | `np.cross(a, b)` | Normales, áreas de triángulos |
| **Norma** | `np.linalg.norm(v)` | Distancias, normalización |
| **Media** | `np.mean(arr, axis=0)` | Centroides, suavizado |
| **Mínimo/Máximo** | `np.min/max(arr, axis=0)` | Bounding box |
| **Arcotangente 2** | `np.arctan2(y, x)` | Ángulos polares, curvatura |
| **Meshgrid** | `np.meshgrid(u, v)` | Parametrización de superficies |
| **Column Stack** | `np.column_stack([x,y,z])` | Matrices de vértices |
| **Diferencias** | `np.diff(arr, axis=0)` | Vectores entre puntos |
| **Suma Acumulada** | `np.cumsum(arr)` | Longitud de curvas |
| **Ordenamiento** | `np.argsort(arr)` | Ordenamiento angular |

---

## Conclusión

El proyecto **Vektra** demuestra la aplicación práctica del álgebra lineal en gráficos 3D:

- **Procesamiento de contornos:** Suavizado, remuestreo e interpolación de curvas 2D
- **Generación de mallas 3D:** Triangulación, cálculo de normales y extrusión
- **Superficies paramétricas:** Transformaciones de coordenadas y marcos ortonormales
- **Iluminación:** Cálculo de normales por vértice para sombreado realista

El uso eficiente de **NumPy** permite realizar estas operaciones de manera **vectorizada**, optimizando el rendimiento.

---
