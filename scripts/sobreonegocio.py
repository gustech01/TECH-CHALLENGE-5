import streamlit as st

@st.cache_resource
def carregar_imagem():
    return "imagens/fiap.png"

def show():
    left, cent, right = st.columns(3)
    with right:
        st.image(carregar_imagem())

    st.title('Sobre o Negócio')
    st.markdown(
        '''
        <div style="text-align: justify;">
            <p>
                A ONG "Passos Mágicos" é dedicada a melhorar a vida de crianças e jovens em vulnerabilidade social através da educação. Com base em dados extensivos de desenvolvimento educacional de 2020, 2021 e 2022, a ONG está focada em criar oportunidades de mudança positiva e progresso para essas comunidades.
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
