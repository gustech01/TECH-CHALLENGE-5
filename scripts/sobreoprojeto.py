import streamlit as st
from pathlib import Path



    
@st.cache_resource
def carregar_imagem():
    return "imagens/fiap.png"

def show():
    left, cent, right = st.columns(3)
    with right:
        st.image(carregar_imagem())

def show():
    # Layout inicial com imagem no canto direito
    left, cent, right = st.columns(3)
    with right:
        imagem = carregar_imagem('imagens/fiap.png')
        if imagem:
            st.image(imagem)

    # Título do projeto
    st.title('Sobre o Projeto')
  
    st.markdown(
        '''
        <div style="text-align: justify;">
            <p>
                O Datathon propõe a criação de uma proposta preditiva. Os participantes criarão um modelo preditivo para prever o comportamento dos estudantes com base em variáveis cruciais. A ideia é utilizar conhecimentos aprendidos no curso, como técnicas de machine learning, deep learning ou processamento de linguagem natural, para propor soluções de algoritmos supervisionados ou não supervisionados.
                
        
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Informações sobre os dados e implementação
    st.markdown(
        '''
        <div style="text-align: justify;">
   <p>
    Os dados foram obtidos das informações educacionais e socioeconômicas dos estudantes da Passos Mágicos, incluindo:
</p>
<p>
    Duas bases de dados com as características de desenvolvimento educacional e questões socioeconômicas dos estudantes e um dicionário de dados com o mapeamento de todas as variáveis.
</p>

<ul>
    <li>
        <b><a style='text-decoration:none;color:blue' href='https://drive.google.com/drive/folders/1Z1j6uzzCOgjB2a6i3Ym1pmJRsasfm7cD'>Bases de Dados</a></b>.
    </li>
</ul>


        ''',
        unsafe_allow_html=True
    )
