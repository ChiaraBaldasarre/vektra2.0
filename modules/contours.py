import cv2
import numpy as np

def get_contours(edges, kernel):
    """
    Extrae el contorno principal de una imagen de bordes.
    
    IMPORTANTE: Devuelve los puntos en el ORDEN ORIGINAL de findContours
    que es el orden correcto para formas complejas (no usar ordenamiento angular).
    
    Args:
        edges: Imagen binaria de bordes (Canny)
        kernel: Kernel para operaciones morfológicas
        
    Returns:
        Array de puntos (n, 2) en el orden del contorno
    """
    # 1. Cierre morfológico para unir bordes rotos
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 2. Encontrar contornos
    contours, _ = cv2.findContours(
        closed_edges,
        cv2.RETR_EXTERNAL,      # Solo contorno externo
        cv2.CHAIN_APPROX_NONE   # Mantener TODOS los puntos
    )
    
    # Verificar que hay contornos
    if not contours:
        print("⚠️ No se encontraron contornos")
        return np.array([])
    
    # 3. Seleccionar el contorno principal (mayor área)
    main_contour = max(contours, key=cv2.contourArea)
    
    # 4. Simplificación ADAPTATIVA (más suave para formas complejas)
    perimeter = cv2.arcLength(main_contour, True)
    
    # Para formas grandes/complejas, simplificar menos
    if perimeter > 1000:  # Forma compleja o grande
        epsilon = 0.00003 * perimeter  # 0.3% (muy poco)
    else:  # Forma pequeña/simple
        epsilon = 0.01 * perimeter   # 1%
    
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    
    # 5. Convertir a array 2D
    points = approx.reshape(-1, 2)
    
    # ⚠️ IMPORTANTE: NO REORDENAR LOS PUNTOS
    # findContours YA los devuelve en el orden correcto del contorno
    # Reordenarlos rompe formas complejas como árboles, letras, etc.
    
    return points  # Orden natural del contorno


# ===== FUNCIÓN AUXILIAR para debug (opcional) =====

def debug_contour_order(points, image_edges, kernel):
    """
    Función para visualizar el orden de puntos (solo para debug).
    No usar en producción.
    """
    if len(points) == 0:
        return None
    
    # Crear imagen para mostrar
    if len(image_edges.shape) == 2:  # Si es escala de grises
        display_img = cv2.cvtColor(image_edges, cv2.COLOR_GRAY2BGR)
    else:
        display_img = image_edges.copy()
    
    # Dibujar puntos numerados
    for i, (x, y) in enumerate(points):
        # Punto rojo
        cv2.circle(display_img, (int(x), int(y)), 4, (0, 0, 255), -1)
        # Número blanco
        cv2.putText(display_img, str(i), (int(x)+5, int(y)-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Conectar puntos con líneas verdes
    for i in range(len(points) - 1):
        pt1 = tuple(points[i].astype(int))
        pt2 = tuple(points[i+1].astype(int))
        cv2.line(display_img, pt1, pt2, (0, 255, 0), 1)
    
    # Cerrar el polígono
    if len(points) > 2:
        pt1 = tuple(points[-1].astype(int))
        pt2 = tuple(points[0].astype(int))
        cv2.line(display_img, pt1, pt2, (0, 255, 0), 1)
    
    return display_img