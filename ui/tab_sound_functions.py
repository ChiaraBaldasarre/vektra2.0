import streamlit as st
import plotly.graph_objects as go
from modules.geometry.fourier import calcular_onda_sonora, extruir_onda_a_malla_3d

def render_sound_functions():
    st.subheader("Funciones Sonoras Procedurales")
    st.markdown("Visualización geométrica de ondas armónicas complejas por Series de Fourier.")

    col_controles, col_graficos = st.columns([1, 2])

    with col_controles:
        st.markdown("**Selector de Ondas**")

        tipo_funcion = st.selectbox(
            "Selecciona la función armónica",
            [
                "Onda Cuadrada",
                "Onda Diente de Sierra",
                "Onda Triangular",
                "Tren de Pulsos",
                "Sierra Asimétrica",
                "Pulso Cuadrático"
            ]
        )

        st.markdown("**Ecuación de Análisis Analítico:**")

        # Diccionario dinámico de visualización de fórmulas matemáticas
        if tipo_funcion == "Onda Cuadrada":
            st.latex(r"y = \frac{4}{\pi} \sum_{k=1}^{N} \frac{1}{2k-1} \sin((2k-1)x)")

        elif tipo_funcion == "Onda Diente de Sierra":
            st.latex(r"y = \frac{2}{\pi} \sum_{k=1}^{N} \frac{1}{k} \sin(kx)")

        elif tipo_funcion == "Onda Triangular":
            st.latex(r"y = \frac{8}{\pi^2} \sum_{k=1}^{N} \frac{(-1)^{k-1}}{(2k-1)^2} \sin((2k-1)x)")

        elif tipo_funcion == "Tren de Pulsos":
            st.latex(r"y = 5 \sum_{k=1}^{N} \frac{1}{k+1} \sin((4k + 1)x)")

        elif tipo_funcion == "Sierra Asimétrica":
            st.latex(r"y = 6 \sum_{k=1}^{N} \frac{1}{2k + 1} \sin((2k + 1)x)")

        elif tipo_funcion == "Pulso Cuadrático":
            st.latex(r"y = \sum_{k=1}^{N} \frac{4}{k^2 + 1} \sin((3k - 1)x)")

        st.markdown("---")
        st.markdown("**Modificadores**")

        armonicos = st.slider(
            "Cantidad de Armónicos (N)",
            min_value=1,
            max_value=40,
            value=12,
            help="Controla el límite superior de corte de la sumatoria."
        )

        profundidad = st.slider("Extrusión Z (Eje Temporal)", 1.0, 12.0, 5.0)
        color_render = st.color_picker("Color del espectro", "#00CED1")

    with col_graficos:
        # Calcular coordenadas vectoriales usando numpy
        x_2d, y_2d = calcular_onda_sonora(tipo_funcion, armonicos)

        # 1. Gráfica 2D Analítica continua
        fig_2d = go.Figure()
        fig_2d.add_trace(go.Scatter(
            x=x_2d, y=y_2d,
            mode='lines',
            line=dict(color=color_render, width=2.5),
            name="Señal continua f(x)"
        ))
        fig_2d.update_layout(
            title=f"Función Analítica: {tipo_funcion} (N = {armonicos})",
            xaxis_title="Tiempo / Espacio (X)",
            yaxis_title="Amplitud (Y)",
            template="plotly_dark",
            height=260,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_2d, use_container_width=True)

        # 2. Renderizado de la Malla 3D extruida
        vertices, faces = extruir_onda_a_malla_3d(x_2d, y_2d, profundidad=profundidad)

        fig_3d = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color_render,
                opacity=0.85,
                flatshading=True,
                name="Malla Extruida Vektra"
            )
        ])
        fig_3d.update_layout(
            title="Espectro Tridimensional de la Señal (Mesh3D)",
            scene=dict(
                xaxis_title='X (Onda)',
                yaxis_title='Y (Amplitud)',
                zaxis_title='Z (Profundidad)'
            ),
            template="plotly_dark",
            height=340,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_3d, use_container_width=True)