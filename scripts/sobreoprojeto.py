import streamlit as st
from pathlib import Path

@st.cache_resource
def carregar_imagem(caminho):
    """Carrega o caminho da imagem."""
    imagem_path = Path(caminho)
    if imagem_path.is_file():
        return str(imagem_path)
    else:
        st.error(f"Imagem não encontrada: {caminho}")
        return None

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
                
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Informações sobre os dados e implementação
    st.markdown(
        '''
        <div style="text-align: justify;">
            <p>
                Os dados foram obtidos do site do IPEA (Instituto de Pesquisa Econômica Aplicada), incluindo:
                <ul>
                    <li>
                        <b><a style='text-decoration:none', href='http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'>Tabela de preços do petróleo Brent</a></b> 
                        (preços por barril em dias úteis, sem incluir frete e seguro).
                    </li>
                    <li>
                        <b><a style='text-decoration:none', href='http://www.ipeadata.gov.br/ExibeSerie.aspx?serid=38590&module=M'>Tabela de preços do dólar</a></b> 
                        para o mesmo período.
                    </li>
                </ul>
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
