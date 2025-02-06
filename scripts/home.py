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
                O grande objetivo do Datathon é permitir que cientistas de dados criem uma proposta preditiva para demonstrar o impacto que a ONG "Passos Mágicos" tem realizado na comunidade que atende. A associação busca instrumentalizar o uso da educação como ferramenta para a mudança das condições de vida das crianças e jovens em vulnerabilidade social.
            </p>
            <p>
                Esta aplicação é um MVP. O projeto completo está disponível em MUDAR <b><a style='text-decoration:none', href='https://github.com/gustech01/TECH-CHALLENGE-4-FIAP'>repositório</a></b> GitHub.
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
