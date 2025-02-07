import streamlit as st
from PIL import Image
import os

# Função para carregar imagens com verificação
@st.cache_resource
def carregar_imagem(nome_arquivo):
    caminho = os.path.join("imagens", nome_arquivo)  # Caminho completo da imagem
    if os.path.exists(caminho):  # Verifica se a imagem existe
        return Image.open(caminho)
    else:
        st.error(f"Imagem '{nome_arquivo}' não encontrada. Verifique o caminho e tente novamente.")
        return None

def show():
    # Layout inicial com a imagem da FIAP no canto direito
    left, cent, right = st.columns(3)
    with right:
        imagem_fiap = carregar_imagem("fiap.png")
        if imagem_fiap:
            st.image(imagem_fiap, use_container_width=True)

    # Título e descrição do projeto
    st.title('Objetivo do Projeto')
    st.markdown(
        '''
        <div style="text-align: justify;">
            <p>
                O grande objetivo do Datathon é permitir que cientistas de dados criem uma proposta preditiva para demonstrar o impacto que a ONG "Passos Mágicos" tem realizado na comunidade que atende. A associação busca instrumentalizar o uso da educação como ferramenta para a mudança das condições de vida das crianças e jovens em vulnerabilidade social.
            </p>
            <p>
                Esta aplicação é um MVP. O projeto completo está disponível em <b><a style='text-decoration:none', href='https://github.com/gustech01/TECH-CHALLENGE-5.git'>repositório</a></b> GitHub.
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Espaço centralizado para a imagem do Passos Mágicos no rodapé
    st.divider()  # Linha divisória para separar o conteúdo
    _, col_central, _ = st.columns([1, 2, 1])  # Criar uma coluna centralizada
    with col_central:
        imagem_passos = carregar_imagem("Passos-magicos-icon-cor.png")
        if imagem_passos:
            st.image(imagem_passos, use_container_width=True)

# Executar o aplicativo
if __name__ == "__main__":
    show()
