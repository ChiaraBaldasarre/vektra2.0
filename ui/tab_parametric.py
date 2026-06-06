import streamlit as st
import numpy as np
import plotly.graph_objects as go
from modules.geometry.parametric import (
    generar_funcion_z, generar_superficie_custom,
    crear_mesh_plotly as crear_mesh_parametrico,
    SUPERFICIES, EJEMPLOS_FORMULAS, FORMULAS_SUPERFICIES,
    EJEMPLOS_Z_FXY, EJEMPLOS_PARAMETRICAS
)


def render_parametric():
    st.subheader("📐 Superficies Paramétricas")
    st.markdown("Genera superficies 3D usando **fórmulas matemáticas de vectores**")

    col_formula, col_vista = st.columns([1, 2])

    with col_formula:
        st.markdown("**🔢 Configuración**")

        modo_superficie = st.radio(
            "Modo de generación",
            options=["Superficies predefinidas", "Fórmula z = f(x,y)", "Superficie paramétrica"],
            horizontal=True
        )

        if modo_superficie == "Superficies predefinidas":
            emojis_superficies = {
                "paraboloide": "🔵", "silla_montar": "🐴", "onda_seno": "🌊",
                "cilindro": "📦", "cono": "🔺", "toro": "🍩",
                "pseudoesfera": "🌐", "enneper": "🌈", "catalan": "🎨",
                "hiperboloide": "⚡", "helicoide": "🌀", "vela": "⛵",
                "romboidal": "◇", "catenoide": "⛓️", "ondulatoria": "〰️",
                "helice": "🧬", "espiral_conica": "🌀", "nudo_trebol": "🍀",
                "nudo_figura_ocho": "8️⃣", "espiral_toroidal": "🌪️",
                "hipocicloide": "🔄", "epicicloide": "⭕", "mobius": "♾️",
                "klein": "🍶", "toro_anudado": "🔗"
            }

            labels_superficies = {
                sup: f"{emojis_superficies.get(sup, '')} {sup.replace('_', ' ').title()} ({FORMULAS_SUPERFICIES.get(sup, 'N/A')})"
                for sup in sorted(SUPERFICIES.keys())
            }

            superficie_sel = st.selectbox(
                "Selecciona una superficie",
                options=sorted(SUPERFICIES.keys()),
                format_func=lambda x: labels_superficies.get(x, x.replace("_", " ").title())
            )

            param_resolution = st.slider("Resolución", 20, 100, 50, key="param_res")


            with st.expander("📐 Ver fórmula"):
                st.code(EJEMPLOS_FORMULAS, language="python")

            try:
                if superficie_sel == "paraboloide":
                    param_a = st.slider("Escala A", 0.5, 2.0, 1.0)
                    param_h = st.slider("Altura máx", 1.0, 4.0, 2.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](a=param_a, height=param_h,
                                                                              resolution=param_resolution)
                elif superficie_sel == "silla_montar":
                    param_size = st.slider("Tamaño", 1.0, 4.0, 2.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](size=param_size,
                                                                              resolution=param_resolution)
                elif superficie_sel == "onda_seno":
                    param_amp = st.slider("Amplitud", 0.5, 2.0, 1.0)
                    param_freq = st.slider("Frecuencia", 0.5, 3.0, 1.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](amplitud=param_amp, frecuencia=param_freq,
                                                                              resolution=param_resolution)
                elif superficie_sel == "helice":
                    param_vueltas = st.slider("Vueltas", 1, 6, 3)
                    param_paso = st.slider("Paso", 0.2, 1.0, 0.5)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](vueltas=param_vueltas, paso=param_paso,
                                                                              resolution=param_resolution)
                elif superficie_sel == "espiral_conica":
                    param_vueltas = st.slider("Vueltas", 2, 8, 4)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](vueltas=param_vueltas,
                                                                              resolution=param_resolution)
                elif superficie_sel == "mobius":
                    param_ancho = st.slider("Ancho banda", 0.2, 0.8, 0.4)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](ancho=param_ancho,
                                                                              resolution=param_resolution)
                elif superficie_sel == "toro":
                    param_r_mayor = st.slider("Radio mayor", 0.5, 2.0, 1.0)
                    param_r_menor = st.slider("Radio menor", 0.2, 0.8, 0.4)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](radio_mayor=param_r_mayor,
                                                                              radio_menor=param_r_menor,
                                                                              resolution=param_resolution)
                elif superficie_sel == "toro_anudado":
                    param_p = st.slider("Parámetro p", 2, 5, 2)
                    param_q = st.slider("Parámetro q", 2, 7, 3)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](p=param_p, q=param_q,
                                                                              resolution=param_resolution)
                elif superficie_sel == "cono":
                    param_radio = st.slider("Radio base", 0.5, 2.0, 1.0)
                    param_altura = st.slider("Altura", 1.0, 4.0, 2.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](radio_base=param_radio,
                                                                              altura=param_altura,
                                                                              resolution=param_resolution)
                elif superficie_sel == "cilindro":
                    param_radio = st.slider("Radio", 0.5, 2.0, 1.0)
                    param_altura = st.slider("Altura", 1.0, 4.0, 2.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](radio=param_radio, altura=param_altura,
                                                                              resolution=param_resolution)
                elif superficie_sel == "pseudoesfera":
                    param_altura = st.slider("Altura", 0.5, 3.0, 2.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](altura=param_altura,
                                                                              resolution=param_resolution)
                elif superficie_sel == "hiperboloide":
                    param_a = st.slider("Escala a", 0.5, 2.0, 1.0)
                    param_b = st.slider("Escala b", 0.5, 2.0, 1.0)
                    param_c = st.slider("Escala c", 0.5, 2.0, 1.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](a=param_a, b=param_b, c=param_c,
                                                                              resolution=param_resolution)
                elif superficie_sel == "helicoide":
                    param_paso = st.slider("Paso", 0.2, 1.0, 0.3)
                    param_vueltas = st.slider("Vueltas", 1, 5, 3)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](paso=param_paso, vueltas=param_vueltas,
                                                                              resolution=param_resolution)
                elif superficie_sel == "catenoide":
                    param_radio = st.slider("Radio", 0.5, 2.0, 1.0)
                    param_altura = st.slider("Altura", 1.0, 4.0, 3.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](radio=param_radio, altura=param_altura,
                                                                              resolution=param_resolution)
                elif superficie_sel == "ondulatoria":
                    param_amp = st.slider("Amplitud", 0.5, 2.0, 1.0)
                    param_n = st.slider("Frecuencia n", 1, 5, 2)
                    param_m = st.slider("Frecuencia m", 1, 5, 3)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](amplitud=param_amp, freq_n=param_n,
                                                                              freq_m=param_m,
                                                                              resolution=param_resolution)
                elif superficie_sel == "hipocicloide":
                    param_R = st.slider("Radio mayor (R)", 3.0, 8.0, 5.0)
                    param_r = st.slider("Radio menor (r)", 1.0, 4.0, 3.0)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](R=param_R, r=param_r,
                                                                              resolution=param_resolution)
                elif superficie_sel == "epicicloide":
                    param_R = st.slider("Radio mayor (R)", 3.0, 8.0, 5.0)
                    param_r = st.slider("Radio menor (r)", 1.0, 4.0, 2.0)
                    param_k = st.slider("Frecuencia (k)", 1, 5, 3)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](R=param_R, r=param_r, k=param_k,
                                                                              resolution=param_resolution)
                elif superficie_sel == "enneper":
                    param_size = st.slider("Tamaño", 0.5, 3.0, 1.5)
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](size=param_size,
                                                                              resolution=param_resolution)
                elif superficie_sel in ["klein", "nudo_trebol", "nudo_figura_ocho", "espiral_toroidal"]:
                    st.info(f"💡 Generando: {superficie_sel.replace('_', ' ').title()}")
                    res_multiplier = 5 if superficie_sel != "klein" else 1
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](
                        resolution=param_resolution * res_multiplier)
                else:
                    vertices_param, faces_param = SUPERFICIES[superficie_sel](resolution=param_resolution)

            except Exception as e:
                st.error(f"Error generando superficie: {e}")
                vertices_param, faces_param = np.array([]), []

        elif modo_superficie == "Fórmula z = f(x,y)":
            st.markdown("**Ingresa tu fórmula:**")

            with st.expander("📚 Ver ejemplos de fórmulas"):
                col_selec, col_code = st.columns([1, 2])
                with col_selec:
                    st.markdown("**Ejemplos disponibles:**")
                    ejemplo_seleccionado = st.selectbox("Elige un ejemplo", options=list(EJEMPLOS_Z_FXY.keys()),
                                                        key="ejemplo_z_select")
                with col_code:
                    st.markdown(f"**Fórmula:** {ejemplo_seleccionado}")
                    st.code(EJEMPLOS_Z_FXY[ejemplo_seleccionado], language="python")
                if st.button("📌 Usar este ejemplo", key="usar_ejemplo_z"):
                    st.session_state.usar_formula_ejemplo = ejemplo_seleccionado

            formula_default = EJEMPLOS_Z_FXY.get(st.session_state.get("usar_formula_ejemplo", ""),
                                                 "sin(sqrt(x**2 + y**2))")
            formula_z = st.text_input("z = f(x, y)", value=formula_default,
                                      help="Usa: sin, cos, tan, exp, log, sqrt, abs, pi, e")

            col_range1, col_range2 = st.columns(2)
            with col_range1:
                x_min = st.number_input("X mín", value=-3.0)
                y_min = st.number_input("Y mín", value=-3.0)
            with col_range2:
                x_max = st.number_input("X máx", value=3.0)
                y_max = st.number_input("Y máx", value=3.0)

            param_resolution = st.slider("Resolución", 20, 80, 50, key="formula_res")
            param_color = st.color_picker("Color", value="#3498db", key="formula_color")
            param_opacity = st.slider("Opacidad", 0.1, 1.0, 0.85, key="formula_opacity")

            try:
                vertices_param, faces_param = generar_funcion_z(
                    formula_z, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, resolution=param_resolution
                )
            except Exception as e:
                st.error(f"Error en fórmula: {e}")
                vertices_param, faces_param = np.array([]), []

        else:
            st.markdown("**Vector r(u,v) = (x, y, z):**")

            with st.expander("📚 Ver ejemplos paramétricos"):
                col_selec, col_code = st.columns([1, 2])
                with col_selec:
                    st.markdown("**Ejemplos disponibles:**")
                    ejemplo_param = st.selectbox("Elige un ejemplo", options=list(EJEMPLOS_PARAMETRICAS.keys()),
                                                 key="ejemplo_param_select")
                    data = EJEMPLOS_PARAMETRICAS[ejemplo_param]
                    st.markdown(f"**Descripción:** {data['descripcion']}")
                with col_code:
                    st.markdown(f"**Fórmulas paramétricas:**")
                    codigo_param = f"# {ejemplo_param}\nx: {data['x']}\ny: {data['y']}\nz: {data['z']}"
                    st.code(codigo_param, language="python")
                if st.button("📌 Usar este ejemplo", key="usar_ejemplo_param"):
                    st.session_state.usar_param_ejemplo = ejemplo_param

            ejemplo_activo = st.session_state.get("usar_param_ejemplo", "Toro")
            ejemplo_data = EJEMPLOS_PARAMETRICAS.get(ejemplo_activo, EJEMPLOS_PARAMETRICAS["Toro"])

            eq_x = st.text_input("x(u,v) =", value=ejemplo_data["x"])
            eq_y = st.text_input("y(u,v) =", value=ejemplo_data["y"])
            eq_z = st.text_input("z(u,v) =", value=ejemplo_data["z"])

            st.markdown("**Rangos de parámetros:**")
            col_u, col_v = st.columns(2)
            with col_u:
                u_min = st.number_input("u mín", value=float(ejemplo_data["u_rango"][0]))
                u_max = st.number_input("u máx", value=float(ejemplo_data["u_rango"][1]))
            with col_v:
                v_min = st.number_input("v mín", value=float(ejemplo_data["v_rango"][0]))
                v_max = st.number_input("v máx", value=float(ejemplo_data["v_rango"][1]))

            param_resolution = st.slider("Resolución", 20, 80, 40, key="custom_res")
            param_color = st.color_picker("Color", value="#e74c3c", key="custom_color")
            param_opacity = st.slider("Opacidad", 0.1, 1.0, 0.85, key="custom_opacity")

            try:
                vertices_param, faces_param = generar_superficie_custom(
                    eq_x, eq_y, eq_z, u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max, resolution=param_resolution
                )
            except Exception as e:
                st.error(f"Error en ecuaciones: {e}")
                vertices_param, faces_param = np.array([]), []

    with col_vista:
        st.markdown("**🔷 Visualización 3D**")
        param_cmap = st.selectbox(
            "Mapa de Colores",
            ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"],
            key="cmap_selector"
        )

        param_opacity = st.slider("Opacidad", 0.0, 1.0, 0.85)
        if len(vertices_param) > 0 and len(faces_param) > 0:
            mesh_param = crear_mesh_parametrico(
                vertices_param, faces_param,
                color_map=param_cmap,
                opacity=param_opacity,
                name="Superficie"
            )

            fig_param = go.Figure(data=[go.Mesh3d(**mesh_param)])
            fig_param.data[0].update(
                intensity=vertices_param[:, 2],
                colorscale= param_cmap, showscale=True, colorbar=dict(title="Z")
            )

            fig_param.update_layout(
                scene=dict(
                    xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                    aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
                ),
                margin=dict(l=0, r=0, b=0, t=30), height=500, title="Superficie Matemática"
            )

            st.plotly_chart(fig_param, width="stretch")

            stats1, stats2, stats3 = st.columns(3)
            with stats1:
                st.metric("Vértices", len(vertices_param))
            with stats2:
                st.metric("Caras", len(faces_param))
            with stats3:
                z_range = vertices_param[:, 2].max() - vertices_param[:, 2].min()
                st.metric("Rango Z", f"{z_range:.2f}")

            with st.expander("📊 Información de la superficie"):
                st.markdown(f"""
                **Estadísticas:**
                - X: [{vertices_param[:, 0].min():.2f}, {vertices_param[:, 0].max():.2f}]
                - Y: [{vertices_param[:, 1].min():.2f}, {vertices_param[:, 1].max():.2f}]
                - Z: [{vertices_param[:, 2].min():.2f}, {vertices_param[:, 2].max():.2f}]
                - Resolución: {param_resolution}×{param_resolution}
                """)
        else:
            st.warning("No se pudo generar la superficie. Verifica los parámetros.")