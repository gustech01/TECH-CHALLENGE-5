import streamlit as st
from pathlib import Path

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
            st.image(imagem, use_column_width=True)
        else:
            st.error("Erro ao carregar a imagem. Verifique o caminho e tente novamente.")

    # Título do projeto
    st.title('Sobre o Projeto Datathon')
  
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
                <b>Base de Dados Disponível:</b> A base de dados completa pode ser acessada <a href="https://drive.google.com/drive/folders/1Z1j6uzzCOgjB2a6i3Ym1pmJRsasfm7cD" target="_blank">neste link</a>.
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

if __name__ == "__main__":

