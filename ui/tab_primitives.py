import streamlit as st
import plotly.graph_objects as go
from modules.geometry.primitives import get_cube, get_pyramid, get_sphere, get_cylinder, get_cone, get_prisma

def render_primitives():
    st.subheader("Generador de Primitivas Geométricas 3D")
    col_ui, col_viz = st.columns([1, 2])

    with col_ui:
        st.markdown("**🎨 Configuración**")
        tipo_figura = st.selectbox(
            "Selecciona la figura",
            ["Cubo", "Esfera", "Cilindro", "Pirámide", "Cono", "Prisma"],
            key="sel_prim"
        )
        color_p = st.color_picker("Color de la figura", "#00CED1", key="col_prim")
        alpha_p = st.slider("Opacidad", 0.1, 1.0, 0.8, key="alpha_prim")
        st.info("Figuras generadas proceduralmente en el origen (0,0,0).")

    with col_viz:
        st.markdown("**🔷 Previsualización 3D**")
        v, i, j, k = None, None, None, None
        m_args = {'color': color_p, 'opacity': alpha_p, 'flatshading': True}

        if tipo_figura == "Cubo": v, i, j, k = get_cube()
        elif tipo_figura == "Pirámide": v, i, j, k = get_pyramid()
        elif tipo_figura == "Esfera": v, i, j, k = get_sphere(); m_args['alphahull'] = 0
        elif tipo_figura == "Cilindro": v, i, j, k = get_cylinder(); m_args['alphahull'] = 0
        elif tipo_figura == "Cono": v, i, j, k = get_cone(); m_args['alphahull'] = 0
        elif tipo_figura == "Prisma":
            v, i, j, k = get_prisma(n=6)
            m_args['alphahull'] = 0

        if v is not None:
            fig_p = go.Figure(data=[go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=i, j=j, k=k, **m_args)])
            fig_p.update_layout(scene=dict(aspectmode='cube'), margin=dict(l=0, r=0, b=0, t=0), height=500)
            st.plotly_chart(fig_p, width="stretch")