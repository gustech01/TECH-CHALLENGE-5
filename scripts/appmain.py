import streamlit as st

# Inicialize o estado da página apenas uma vez
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Menu lateral
st.sidebar.title("Menu")
menu = st.sidebar.radio("Selecione uma página:", ["Home", "Sobre o Negócio", "Sobre o Projeto", "MVP"])

# Atualize o estado da página
if st.session_state.current_page != menu:
    st.session_state.current_page = menu

# Funções para exibir cada página
def show_home():
    import home
    home.show()

def show_sobre_o_negocio():
    import sobreonegocio
    sobreonegocio.show()

def show_sobre_o_projeto():
    import sobreoprojeto
    sobreoprojeto.show()

def show_mvp():
    import mvpfinal
    mvpfinal.show()

# Navegação entre as páginas
if st.session_state.current_page == "Home":
    show_home()
elif st.session_state.current_page == "Sobre o Negócio":
    show_sobre_o_negocio()
elif st.session_state.current_page == "Sobre o Projeto":
    show_sobre_o_projeto()
elif st.session_state.current_page == "MVP":
    show_mvp()
