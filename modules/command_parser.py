import re
from modules.primitives import (
    get_cube, get_sphere, get_cylinder,
    get_cone, get_pyramid, get_prisma
)

def parse_commands(code: str):
    """
    Recibe texto con múltiples líneas de comandos
    y devuelve una lista de figuras 3D listas para renderizar
    """
    figures = []
    lines = code.strip().splitlines()

    for line in lines:
        line = line.strip().lower()
        if not line:
            continue

        parts = re.split(r'\s+', line)
        cmd = parts[0]

        # -------- CUBO --------
        if cmd == "cube":
            v, i, j, k = get_cube()
            figures.append((v, i, j, k))

        # -------- ESFERA --------
        elif cmd == "sphere":
            v, i, j, k = get_sphere()
            figures.append((v, i, j, k))

        # -------- CILINDRO --------
        elif cmd == "cylinder":
            v, i, j, k = get_cylinder()
            figures.append((v, i, j, k))

        # -------- CONO --------
        elif cmd == "cone":
            v, i, j, k = get_cone()
            figures.append((v, i, j, k))

        # -------- PIRÁMIDE --------
        elif cmd == "pyramid":
            v, i, j, k = get_pyramid()
            figures.append((v, i, j, k))

        # -------- PRISMA --------
        elif cmd == "prisma":
            n = 6
            for p in parts[1:]:
                if p.startswith("n="):
                    n = int(p.split("=")[1])
            v, i, j, k = get_prisma(n)
            figures.append((v, i, j, k))

        else:
            raise ValueError(f"Comando no reconocido: {cmd}")

    return figures
