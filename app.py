import streamlit as st
import plotly.graph_objects as go


# Ejemplo de como Jose y Chiara pasarán datos a la UI
def generar_cubo_test():
    fig = go.Figure(data=[
        go.Mesh3d(
            x=[0, 1, 1, 0, 0, 1, 1, 0],
            y=[0, 0, 1, 1, 0, 0, 1, 1],
            z=[0, 0, 0, 0, 1, 1, 1, 1],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color='cyan', opacity=0.5
        )
    ])
    return fig


st.plotly_chart(generar_cubo_test(), use_container_width=True)
