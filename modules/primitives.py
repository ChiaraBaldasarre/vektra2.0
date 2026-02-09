import numpy as np

def get_cube():
    # Centrado de -0.5 a 0.5 en todos los ejes
    vertices = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ])
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    return vertices, i, j, k

def get_pyramid():
    # Base en Z = -0.5, Punta en Z = 0.5
    vertices = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [0, 0, 0.5]
    ])
    i = [0, 0, 0, 1, 2, 3]
    j = [1, 2, 1, 2, 3, 0]
    k = [2, 3, 4, 4, 4, 4]
    return vertices, i, j, k

def get_sphere(n=20):
    # Ya centrada por naturaleza en (0,0,0)
    phi = np.linspace(0, np.pi, n)
    theta = np.linspace(0, 2 * np.pi, n)
    phi, theta = np.meshgrid(phi, theta)
    x = (0.5 * np.sin(phi) * np.cos(theta)).flatten()
    y = (0.5 * np.sin(phi) * np.sin(theta)).flatten()
    z = (0.5 * np.cos(phi)).flatten()
    return np.stack([x, y, z], axis=1), None, None, None

def get_cylinder(n=20):
    # Base en Z = -0.5, Techo en Z = 0.5
    theta = np.linspace(0, 2*np.pi, n)
    x = 0.5 * np.cos(theta)
    y = 0.5 * np.sin(theta)
    v_bottom = np.stack([x, y, np.full_like(x, -0.5)], axis=1)
    v_top = np.stack([x, y, np.full_like(x, 0.5)], axis=1)
    return np.vstack([v_bottom, v_top]), None, None, None

def get_cone(n=20):
    # Base en Z = -0.5, Punta en Z = 0.5
    theta = np.linspace(0, 2*np.pi, n)
    x = 0.5 * np.cos(theta)
    y = 0.5 * np.sin(theta)
    v_base = np.stack([x, y, np.full_like(x, -0.5)], axis=1)
    v_tip = np.array([[0, 0, 0.5]])
    return np.vstack([v_base, v_tip]), None, None, None

def get_prisma(n=6):
    # Base en Z = -0.5, Techo en Z = 0.5
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = 0.5 * np.cos(theta)
    y = 0.5 * np.sin(theta)
    v_bottom = np.stack([x, y, np.full_like(x, -0.5)], axis=1)
    v_top = np.stack([x, y, np.full_like(x, 0.5)], axis=1)
    return np.vstack([v_bottom, v_top]), None, None, None