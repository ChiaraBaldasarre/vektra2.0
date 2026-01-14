import streamlit as st

st.set_page_config(page_title='Vektra - 3D Engine', layout='wide')
st.title('📐 Vektra: Motor de Renderizado')

modo = st.sidebar.radio('Selecciona un modo:', ['Imagen a 3D', 'Modelado por Código'])

tab1, tab2 = st.tabs(['Visualizador 3D', 'Logs'])

with tab1:
    st.info('Bienvenido a Vektra. Selecciona una opción en el menú lateral.')