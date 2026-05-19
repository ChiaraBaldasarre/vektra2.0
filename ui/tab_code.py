import streamlit as st
import plotly.graph_objects as go
from modules.utils.command_parser import parse_commands


def render_code_editor():
    st.subheader("⌨️ Editor de Código para Figuras 3D")

    example_code = """# Ejemplo de comandos
cube
sphere
prisma n=8
cylinder
"""

    code = st.text_area(
        "Escribí comandos para generar figuras 3D:",
        value=example_code,
        height=300
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        figura_color = st.color_picker("Color de las figuras", value="#00CED1", key="code_editor_color")

    with col2:
        st.write("")

    if st.button("▶ Ejecutar código"):
        try:
            figures = parse_commands(code)
            st.session_state.code_figures = figures
            st.session_state.code_executed = True
        except Exception as e:
            st.error(f"❌ Error en el código: {e}")
            st.session_state.code_executed = False

    if hasattr(st.session_state, 'code_executed') and st.session_state.code_executed:
        for v, i, j, k in st.session_state.code_figures:
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=v[:, 0], y=v[:, 1], z=v[:, 2],
                    i=i, j=j, k=k,
                    opacity=0.9, color=figura_color, flatshading=False
                )
            ])
            fig.update_layout(scene=dict(aspectmode='cube'), margin=dict(l=0, r=0, b=0, t=0), height=350)
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("👆 Haz clic en 'Ejecutar código' para generar las figuras")