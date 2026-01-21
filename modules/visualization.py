"""
Modulo de visualizacion 3D.
Contiene una función de ejemplo para generar un cubo 3D.
"""

import plotly.graph_objects as go


def generar_cubo_3d():
    """
    Genera un cubo 3D de ejemplo.
    
    Returns:
        Figura de Plotly con un cubo 3D
    """
    fig = go.Figure(data=[
        go.Mesh3d(
            x=[0, 1, 1, 0, 0, 1, 1, 0],
            y=[0, 0, 1, 1, 0, 0, 1, 1],
            z=[0, 0, 0, 0, 1, 1, 1, 1],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color='cyan', 
            opacity=0.5,
            name='Cubo'
        )
    ])
    fig.update_layout(
        title="Visualización 3D - Cubo de Ejemplo",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    return fig
