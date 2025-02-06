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
                Este projeto utiliza a metodologia CRISP-DM (CRoss Industry Standard Process for Data Mining), amplamente aplicada em projetos de dados. A metodologia segue 6 etapas principais:
                <ul>
                    <li><b>Análise do negócio (business understanding):</b> Entendimento do produto/serviço, público-alvo e estratégias do setor;</li>
                    <li><b>Análise dos dados (data understanding):</b> Seleção e avaliação dos dados úteis para a solução do problema;</li>
                    <li><b>Preparação dos dados (data preparation):</b> Pré-processamento dos dados para atender aos requisitos das soluções propostas;</li>
                    <li><b>Modelagem (modeling):</b> Extração de insights úteis ao negócio por meio de modelagem dos dados;</li>
                    <li><b>Avaliação (evaluation):</b> Verificação do desempenho do modelo aplicado;</li>
                    <li><b>Implementação (deployment):</b> Disponibilização dos resultados às partes interessadas, como dashboards ou relatórios.</li>
                </ul>
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
