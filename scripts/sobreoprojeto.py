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
    st.title('Sobre o Projeto Datathon')
  
    st.markdown(
        '''
        <div style="text-align: justify;">
            <p>
                O Datathon é um evento voltado para cientistas e analistas de dados que têm como missão criar soluções que demonstrem o impacto da ONG "Passos Mágicos" na comunidade que atende. Esta ONG utiliza a educação como ferramenta para melhorar a vida de crianças e jovens em situação de vulnerabilidade social.
            </p>
            <p>
                <b>Objetivo do Datathon:</b> O objetivo principal é utilizar dados educacionais e socioeconômicos dos anos de 2020, 2021 e 2022 para criar análises e modelos preditivos. Existem duas principais propostas que os participantes podem escolher:
                <ul>
                    <li><b>Proposta Analítica:</b> Criar um dashboard e contar uma história com os dados, mostrando como a ONG impactou o desempenho dos estudantes. Isso ajuda a ONG a tomar decisões baseadas em indicadores de performance e entender melhor o perfil dos estudantes.</li>
                    <li><b>Proposta Preditiva:</b> Desenvolver um modelo preditivo que prevê o comportamento dos estudantes com base em variáveis importantes. Aqui, a criatividade é bem-vinda, e os participantes podem usar técnicas de aprendizado de máquina, aprendizado profundo ou processamento de linguagem natural para criar suas soluções.</li>
                </ul>
            </p>
            <p>
                <b>Base de Dados:</b> Os dados utilizados no Datathon incluem informações educacionais e socioeconômicas dos estudantes atendidos pela ONG "Passos Mágicos". Além das bases de dados, também estão disponíveis relatórios de pesquisa para ajudar os participantes a entenderem melhor o contexto e o impacto da ONG.
            </p>
            <p>
                <b>Entrega do Projeto:</b> Os participantes podem optar por entregar uma ou ambas as propostas. Para a proposta analítica, a entrega deve incluir um dashboard e um relatório de análise. Para a proposta preditiva, a entrega deve incluir um modelo preditivo implementado no Streamlit.
            </p>
            <p>
                Os projetos podem ser compartilhados por meio de um repositório no GitHub, incluindo todos os arquivos utilizados e links para o dashboard ou modelo preditivo.
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
