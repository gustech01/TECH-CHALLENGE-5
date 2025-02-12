import streamlit as st
from PIL import Image
import os

# Função para carregar imagens com verificação de existência
def carregar_imagem(nome_arquivo):
    caminho = os.path.join("imagens", nome_arquivo)  # Define o caminho completo
    if os.path.exists(caminho):  # Verifica se a imagem existe
        return Image.open(caminho)
    else:
        st.error(f"Imagem '{nome_arquivo}' não encontrada. Verifique o caminho e tente novamente.")
        return None

def show():
    # Layout inicial com imagem no canto direito
    left, cent, right = st.columns(3)
    with right:
        imagem_fiap = carregar_imagem("fiap.png")
        if imagem_fiap:
            st.image(imagem_fiap, use_container_width=True)

    # Título do projeto
    st.title('Sobre o Projeto')

    st.markdown(
        '''
        <div style="text-align: justify;">
            <p>
                O Datathon é uma competição dedicada a cientistas e analistas de dados, cujo objetivo é desenvolver soluções inovadoras utilizando dados educacionais e socioeconômicos dos estudantes atendidos pela ONG "Passos Mágicos". Esta ONG tem como missão transformar a vida de crianças e jovens em situação de vulnerabilidade social através da educação.
            </p>
            <p>
                <b>Foco no Modelo Preditivo:</b> Nesta edição do Datathon, o destaque vai para a criação de um modelo preditivo. Este modelo tem a finalidade de prever o comportamento dos estudantes com base em variáveis cruciais, ajudando a ONG a identificar padrões e tomar decisões que impactem positivamente o desenvolvimento dos alunos.
                <ul>
                    <li><b>Proposta Preditiva:</b> Os participantes devem utilizar técnicas avançadas de aprendizado de máquina (machine learning), aprendizado profundo (deep learning) ou processamento de linguagem natural para desenvolver um algoritmo preditivo. O objetivo é encontrar as melhores soluções criativas para realizar previsões precisas sobre o desempenho dos estudantes.</li>
                </ul>
            </p>
            <p>
                <b>Base de Dados:</b> Os dados fornecidos incluem informações detalhadas sobre o desempenho educacional e as condições socioeconômicas dos estudantes de 2020 a 2022. Além disso, relatórios de pesquisa da Passos Mágicos estão disponíveis para ajudar no entendimento do contexto e na identificação de variáveis importantes para o modelo preditivo.
            </p>
            <p>
                <b>Base de Dados Disponível:</b> <a href="https://drive.google.com/drive/folders/1Z1j6uzzCOgjB2a6i3Ym1pmJRsasfm7cD" target="_blank">Link de Acesso</a>
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Espaço centralizado para a imagem no rodapé
    st.divider()  # Linha divisória para separar o conteúdo
    _, col_central, _ = st.columns([1, 2, 1])  # Coluna centralizada
    with col_central:
        imagem_passos = carregar_imagem("Passos-magicos-icon-cor.png")
        if imagem_passos:
            st.image(imagem_passos, use_container_width=True)

# Executar o aplicativo
if __name__ == "__main__":
    show()
