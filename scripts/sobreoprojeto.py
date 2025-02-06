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
                O Datathon é uma competição voltada para cientistas e analistas de dados, cujo objetivo é criar soluções inovadoras utilizando dados educacionais e socioeconômicos dos estudantes atendidos pela ONG "Passos Mágicos". Esta ONG tem como missão transformar a vida de crianças e jovens em situação de vulnerabilidade social através da educação.
            </p>
            <p>
                <b>Foco no Modelo Preditivo:</b> Para esta edição do Datathon, o destaque vai para a criação de um modelo preditivo capaz de prever o comportamento dos estudantes com base em variáveis cruciais. Esse modelo será fundamental para ajudar a ONG a identificar padrões e tomar decisões informadas que impactem positivamente o desenvolvimento dos alunos.
                <ul>
                    <li><b>Proposta Preditiva:</b> Utilizando técnicas avançadas de aprendizado de máquina (machine learning), aprendizado profundo (deep learning) ou processamento de linguagem natural, os participantes devem desenvolver um algoritmo que faça previsões precisas. A criatividade é encorajada para encontrar as melhores soluções preditivas.</li>
                </ul>
            </p>
            <p>
                <b>Base de Dados:</b> Os dados fornecidos incluem informações detalhadas sobre o desempenho educacional e as condições socioeconômicas dos estudantes de 2020 a 2022. Além disso, relatórios de pesquisa da Passos Mágicos estão disponíveis para ajudar no entendimento do contexto e na identificação de variáveis importantes para o modelo preditivo.
            </p>
            <p>
                <b>Entrega do Projeto:</b> Os participantes devem entregar o modelo preditivo com deploy realizado no Streamlit. A entrega deve incluir todos os arquivos utilizados e o código-fonte, que podem ser compartilhados por meio de um repositório no GitHub, juntamente com o link para o modelo preditivo.
            </p>
            <p>
                Ao focar na proposta preditiva, você estará contribuindo diretamente para o desenvolvimento de soluções que podem transformar a vida dos estudantes atendidos pela ONG "Passos Mágicos".
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
