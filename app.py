import streamlit as st

from ui.tab_vectorizer import render_vectorizer
from ui.tab_primitives import render_primitives
from ui.tab_code import render_code_editor
from ui.tab_parametric import render_parametric

st.set_page_config(page_title="Vektra", layout="wide")

def main():
    st.title("Motor de Vectorización de Imágenes y Renderizado Procedural")

    tabs = st.tabs([
        "Vectorizador de Imagen",
        "Generador de Figuras Primitivas",
        "Editor por Código",
        "Superficies Matemáticas"
    ])

    with tabs[0]:
        render_vectorizer()

    with tabs[1]:
        render_primitives()

    with tabs[2]:
        render_code_editor()

    with tabs[3]:
        render_parametric()

if __name__ == "__main__":
    main()