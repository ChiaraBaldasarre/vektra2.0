# 📚 Documentación de Opciones - Vektra

## Guía completa de los paneles de configuración

Este documento explica en detalle todas las opciones disponibles en los paneles laterales de la aplicación Vektra para vectorización de imágenes y generación de superficies 3D.

---

## 🖼️ Tab: Vectorizador de Imagen

### ⚙️ Parámetros de Detección Canny

El algoritmo Canny es el método principal para detectar bordes en la imagen.

| Opción | Rango | Valor por defecto | Descripción |
|--------|-------|-------------------|-------------|
| **Tamaño del kernel** | 1-31 (impares) | 5 | Controla el desenfoque gaussiano aplicado antes de detectar bordes. **Valores altos** reducen ruido pero pierden detalles finos. **Valores bajos** mantienen detalles pero son más sensibles al ruido. |
| **Umbral Bajo (Canny)** | 0-200 | 50 | Umbral inferior para la detección de bordes. Los píxeles con gradiente menor a este valor se descartan. **Valores bajos** detectan más bordes (incluyendo ruido). |
| **Umbral Alto (Canny)** | Umbral Bajo+10 a 500 | 150 | Umbral superior. Los píxeles con gradiente mayor a este valor se consideran bordes seguros. **La diferencia entre ambos umbrales** determina la histéresis del algoritmo. |

#### 💡 Consejos para ajustar Canny:
- **Imágenes con mucho ruido**: Aumenta el kernel (9-15) y los umbrales
- **Imágenes limpias/dibujos**: Kernel bajo (3-5) y umbrales bajos (30-100)
- **Bordes débiles**: Reduce ambos umbrales
- **Demasiados bordes falsos**: Aumenta el umbral bajo

---

### 🎛️ Parámetros de Extrusión 3D

Controlan cómo se genera el modelo 3D a partir del contorno detectado.

| Opción | Rango | Valor por defecto | Descripción |
|--------|-------|-------------------|-------------|
| **Altura de extrusión** | 0.1-3.0 | 1.0 | Profundidad del modelo 3D. Define qué tan "grueso" será el objeto extruido. |
| **Color de la malla** | Selector de color | #1E90FF (azul) | Color del modelo 3D renderizado. |
| **Opacidad** | 0.1-1.0 | 0.8 | Transparencia del modelo. 1.0 = completamente opaco, 0.1 = casi transparente. |

---

### 💡 Mejora de Detección

Diferentes algoritmos para detectar los bordes del objeto.

| Modo | Mejor para | Descripción |
|------|------------|-------------|
| **Canny Estándar** | Imágenes con bordes claros | Algoritmo clásico de detección de bordes. Funciona bien en la mayoría de casos. |
| **Umbral Adaptativo** | Iluminación desigual | Calcula umbrales locales para cada región. Ideal cuando hay sombras o variaciones de luz. |
| **Multi-Escala** | Detalles variados | Detecta bordes en múltiples resoluciones y los combina. Captura tanto detalles finos como contornos grandes. |
| **Segmentación Automática** | Separar objeto del fondo | Usa GrabCut para segmentar automáticamente. Mejor cuando el objeto está claramente separado del fondo. |

#### Opciones específicas por modo:

**Umbral Adaptativo:**
| Opción | Rango | Descripción |
|--------|-------|-------------|
| Tamaño de bloque | 3-51 (impares) | Tamaño del vecindario para calcular el umbral local. Mayor = más suave. |
| Constante C | -10 a 20 | Valor restado al umbral calculado. Ajusta la sensibilidad. |

**Segmentación Automática:**
| Opción | Rango | Descripción |
|--------|-------|-------------|
| Iteraciones GrabCut | 1-10 | Más iteraciones = mejor segmentación pero más lento. |

---

### 🔧 Calidad de Malla

Opciones de preprocesamiento y mejora del contorno.

| Opción | Tipo | Descripción |
|--------|------|-------------|
| **Preprocesado avanzado (CLAHE)** | Checkbox | **CLAHE** (Contrast Limited Adaptive Histogram Equalization) mejora el contraste local. Actívalo para imágenes con bajo contraste. |
| **Método de reducción de ruido** | Selector | Algoritmo para suavizar la imagen antes de detectar bordes. |

#### Métodos de reducción de ruido:

| Método | Velocidad | Preserva bordes | Mejor para |
|--------|-----------|-----------------|------------|
| **Bilateral** | Rápido | ✅ Excelente | Uso general (recomendado) |
| **Gaussiano** | Muy rápido | ❌ No | Imágenes muy ruidosas |
| **Mediana** | Rápido | ✅ Bueno | Ruido "sal y pimienta" |
| **NL-Means** | Lento | ✅ Excelente | Máxima calidad |

| Opción | Tipo | Descripción |
|--------|------|-------------|
| **Suavizar contorno** | Checkbox | Aplica suavizado de media móvil al contorno para eliminar irregularidades. |
| **Ventana de suavizado** | 3-15 | Tamaño de la ventana de suavizado. Mayor = contorno más suave pero menos detallado. |
| **Remuestrear contorno** | Checkbox | Redistribuye los puntos uniformemente a lo largo del contorno. Mejora la calidad de la malla 3D. |
| **Puntos de la malla** | 50-500 | Número de puntos del contorno remuestreado. Más puntos = más detalle pero más procesamiento. |

---

### 🎯 Ajuste de Contorno

Control fino sobre cómo se extrae y procesa el contorno.

| Opción | Valores | Descripción |
|--------|---------|-------------|
| **Modo de contorno** | Solo el más grande / Unidos / Hull | Define cómo manejar múltiples contornos en la imagen. |

#### Modos de contorno explicados:

| Modo | Comportamiento |
|------|----------------|
| **Solo el más grande** | Selecciona únicamente el contorno con mayor área. Ignora objetos pequeños. |
| **Todos los contornos (unidos)** | Une todos los contornos detectados en uno solo, conectándolos por puntos cercanos. |
| **Todos los contornos (hull)** | Crea el casco convexo que envuelve todos los contornos. |

| Opción | Rango | Descripción |
|--------|-------|-------------|
| **Tamaño kernel morfológico** | 3-21 | Tamaño del kernel para operaciones morfológicas (cierre). **Mayor** = cierra huecos más grandes en el contorno. |
| **Área mínima del contorno** | 10-5000 px² | Ignora contornos más pequeños que este valor. Filtra ruido y objetos pequeños. |
| **Simplificación del contorno** | Ninguna/Baja/Media/Alta | Reduce la cantidad de puntos usando el algoritmo Douglas-Peucker. |

#### Niveles de simplificación:

| Nivel | Efecto | Usar cuando |
|-------|--------|-------------|
| **Ninguna** | Mantiene todos los puntos | Máximo detalle requerido |
| **Baja** | Simplificación mínima | Balance detalle/rendimiento |
| **Media** | Simplificación moderada | Uso general |
| **Alta** | Simplificación agresiva | Formas simples, mejor rendimiento |

| Opción | Tipo | Descripción |
|--------|------|-------------|
| **Invertir bordes** | Checkbox | Invierte la imagen de bordes (blanco↔negro). Útil cuando el objeto es más oscuro que el fondo. |
| **Trazar bordes (mejor para dibujos)** | Checkbox | Usa cierre adaptativo para seguir los bordes con mayor precisión. **Recomendado para dibujos lineales.** |

#### Opciones de trazado de bordes:

| Opción | Rango | Descripción |
|--------|-------|-------------|
| **Iteraciones de cierre** | 1-20 | Número de iteraciones de dilatación para conectar bordes separados. **Mayor** = conecta bordes más lejanos. |
| **Radio de ajuste a bordes** | 1-30 px | Radio de búsqueda para ajustar cada punto del contorno al borde original más cercano. **Mayor** = más tolerancia. |

| Opción | Tipo | Descripción |
|--------|------|-------------|
| **Usar Convex Hull** | Checkbox | Envuelve el contorno en su casco convexo. Elimina todas las concavidades (entrantes). |
| **Orden de puntos** | Selector | Método para ordenar los puntos del contorno antes de crear la malla 3D. |

#### Métodos de ordenamiento:

| Método | Descripción |
|--------|-------------|
| **Original (recomendado)** | Mantiene el orden natural del contorno. Funciona para cualquier forma. |
| **Angular (solo convexos)** | Ordena por ángulo desde el centroide. Solo funciona bien con formas convexas. |
| **Optimizado** | Intenta optimizar el orden para minimizar auto-intersecciones. |

---

## 📐 Tab: Superficies Matemáticas

### Modos de generación

| Modo | Descripción |
|------|-------------|
| **Superficies predefinidas** | Catálogo de 25 superficies matemáticas clásicas listas para usar. |
| **Fórmula z = f(x,y)** | Define tu propia superficie como función de x e y. |
| **Superficie paramétrica** | Define una superficie usando ecuaciones paramétricas r(u,v) = (x, y, z). |

### Parámetros comunes

| Opción | Rango | Descripción |
|--------|-------|-------------|
| **Resolución** | 20-100 | Número de puntos por eje. **Mayor** = más detalle pero más lento. |
| **Color** | Selector | Color base de la superficie. |
| **Opacidad** | 0.1-1.0 | Transparencia de la superficie. |

### Superficies predefinidas disponibles

#### Superficies clásicas z = f(x,y):
| Nombre | Fórmula | Descripción |
|--------|---------|-------------|
| Paraboloide | z = x² + y² | Forma de "tazón" |
| Silla de montar | z = x² - y² | Paraboloide hiperbólico |
| Onda seno | z = sin(x)·cos(y) | Superficie ondulada |

#### Superficies paramétricas:
| Nombre | Descripción |
|--------|-------------|
| Cilindro | Superficie cilíndrica |
| Cono | Superficie cónica |
| Toro | Forma de "dona" |
| Esfera (pseudoesfera) | Superficie esférica |
| Enneper | Superficie minimal |
| Catalan | Superficie minimal |
| Hiperboloide | Hiperboloide de una hoja |
| Helicoide | Superficie de tornillo |
| Vela | Superficie de vela |
| Romboidal | Superficie romboidal |
| Catenoide | Superficie minimal de revolución |
| Ondulatoria | Superficie con ondas paramétricas |

#### Curvas 3D (tubos):
| Nombre | Descripción |
|--------|-------------|
| Hélice | Curva helicoidal |
| Espiral cónica | Espiral que se estrecha |
| Nudo trébol | Nudo de tres lóbulos |
| Nudo figura-ocho | Nudo en forma de 8 |
| Espiral toroidal | Espiral sobre un toro |
| Hipocicloide | Curva cicloidal interior |
| Epicicloide | Curva cicloidal exterior |

#### Superficies especiales:
| Nombre | Descripción |
|--------|-------------|
| Möbius | Banda de Möbius (no orientable, un solo lado) |
| Klein | Botella de Klein (no orientable, sin borde) |
| Toro anudado | Nudo toroidal (p,q) |

### Funciones matemáticas disponibles

Para las fórmulas personalizadas puedes usar:

| Categoría | Funciones |
|-----------|-----------|
| **Trigonométricas** | sin, cos, tan |
| **Inversas** | arcsin, arccos, arctan, atan2 |
| **Hiperbólicas** | sinh, cosh, tanh |
| **Exponenciales** | exp, log |
| **Otras** | sqrt, abs, power, floor, ceil, sign, mod |
| **Constantes** | pi, e |
| **Comparación** | maximum, minimum |

#### Ejemplos de fórmulas z = f(x,y):
```python
# Paraboloide
x**2 + y**2

# Silla de montar
x**2 - y**2

# Ondas radiales
sin(sqrt(x**2 + y**2))

# Gaussiana
exp(-(x**2 + y**2))

# Hiperbólica
tanh(x) * tanh(y)

# Roseta polar
sin(3*atan2(y,x)) * (x**2 + y**2)
```

---

## 🎯 Flujo de trabajo recomendado

### Para vectorizar una imagen:

1. **Carga la imagen** y observa la detección inicial
2. **Ajusta los umbrales de Canny** hasta ver los bordes del objeto claramente
3. **Activa "Trazar bordes"** si es un dibujo lineal
4. **Ajusta las iteraciones de cierre** para conectar bordes separados
5. **Aumenta el radio de ajuste** si el contorno no sigue los bordes
6. **Reduce la simplificación** si pierdes detalle
7. **Ajusta la extrusión** y color para el modelo 3D final

### Para superficies matemáticas:

1. **Selecciona el modo** de generación apropiado
2. **Elige una superficie predefinida** o escribe tu fórmula
3. **Ajusta la resolución** según el detalle requerido
4. **Modifica los parámetros específicos** de cada superficie
5. **Explora la visualización 3D** interactiva

---

## ⚠️ Solución de problemas comunes

| Problema | Solución |
|----------|----------|
| No se detectan bordes | Reduce los umbrales de Canny o prueba otro modo de detección |
| Demasiado ruido | Aumenta el kernel gaussiano y usa filtro bilateral |
| Contorno no sigue los bordes | Activa "Trazar bordes" y aumenta el radio de ajuste |
| Contorno tiene huecos | Aumenta el kernel morfológico y las iteraciones de cierre |
| Malla 3D irregular | Activa remuestreo y aumenta los puntos de malla |
| Superficie matemática vacía | Verifica la sintaxis de la fórmula y los rangos de parámetros |
| Superficie con huecos | Aumenta la resolución |

---

*Documentación generada para Vektra v1.0*
