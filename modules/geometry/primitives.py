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

def get_sphere(n=30):
    # Ya centrada por naturaleza en (0,0,0)
    phi = np.linspace(0, np.pi, n)
    theta = np.linspace(0, 2*np.pi, n)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    v = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])

    faces = np.array(faces)
    return v, faces[:,0], faces[:,1], faces[:,2]

def get_cylinder(n=32):
    # Base en Z = -0.5, Techo en Z = 0.5
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    top = np.column_stack([x, y, np.ones(n)])
    bot = np.column_stack([x, y, -np.ones(n)])

    v = np.vstack([top, bot])
    faces = []

    for i in range(n):
        a = i
        b = (i+1) % n
        c = i + n
        d = (b + n)
        faces.append([a, b, d])
        faces.append([a, d, c])

    faces = np.array(faces)
    return v, faces[:,0], faces[:,1], faces[:,2]

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
    """
    Prisma regular de n lados centrado en el origen
    Altura total = 2  (z = -1 a 1)
    Radio = 1
    """
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # vértices
    top = np.column_stack([x, y, np.ones(n)])
    bottom = np.column_stack([x, y, -np.ones(n)])
    v = np.vstack([top, bottom])

    faces = []

    # caras laterales
    for i in range(n):
        a = i
        b = (i + 1) % n
        c = i + n
        d = b + n
        faces.append([a, b, d])
        faces.append([a, d, c])

    # tapa superior
    for i in range(1, n - 1):
        faces.append([0, i, i + 1])

    # tapa inferior
    base = n
    for i in range(1, n - 1):
        faces.append([base, base + i + 1, base + i])

    faces = np.array(faces)
    return v, faces[:, 0], faces[:, 1], faces[:, 2]