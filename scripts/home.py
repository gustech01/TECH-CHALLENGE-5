import streamlit as st

@st.cache_resource
def carregar_imagem():
    return "imagens/fiap.png"

def show():
    # Layout inicial com imagem
    left, cent, right = st.columns(3)
    with right:
        st.image(carregar_imagem())
    
    st.title('Objetivo do Projeto')
    st.markdown(
        '''
        <div style="text-align: justify;">
            <p>
                Este projeto tem por objetivo o desenvolvimento de um dashboard interativo capaz de gerar insights relevantes para tomada de decisão no que diz respeito ao negócio do petróleo Brent, o que inclui a implementação de um modelo de Machine Learning que traga o forecasting dos preços.
            </p>
            <p>
                Esta aplicação é um MVP. O projeto completo está disponível em <b><a style='text-decoration:none', href='https://github.com/gustech01/TECH-CHALLENGE-4-FIAP'>repositório</a></b> GitHub.
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
