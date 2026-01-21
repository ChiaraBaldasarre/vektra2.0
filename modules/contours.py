import cv2
import numpy as np

def get_contours(edges, kernel):
  ## Encontrar contornos y seleccionar la forma mas externa
  closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
  contours, _ = cv2.findContours(
    closed_edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
  )

  main_contour = max(contours, key=cv2.contourArea)

  ## Simplificacion
  epsilon = 0.01 * cv2.arcLength(main_contour, True)
  approx = cv2.approxPolyDP(main_contour, epsilon, True)

  ## Conversion a arreglo de NumPy
  points = approx.reshape(-1, 2)
  

  ## Ordenar puntos criticos
  center = points.mean(axis=0)

  angles = np.arctan(
    points[:, 1] - center[1],
    points[:, 0] - center[0]
  )

  ordered_points = points[np.argsort(angles)]
  return ordered_points