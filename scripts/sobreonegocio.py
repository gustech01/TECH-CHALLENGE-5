import streamlit as st

@st.cache_resource
def carregar_imagem(nome_arquivo):
    return f"imagens/{nome_arquivo}"

def show():
    # Layout inicial: logo no canto direito superior
    left, cent, right = st.columns(3)
    with right:
        st.image(carregar_imagem("fiap.png"))

    # Título e texto
    st.title('Sobre o Negócio')
    st.markdown(
        '''
        <div style="text-align: justify;">
            <p>
                A ONG "Passos Mágicos" é dedicada a melhorar a vida de crianças e jovens em vulnerabilidade social através da educação. Com base em dados extensivos de desenvolvimento educacional de 2020, 2021 e 2022, a ONG está focada em criar oportunidades de mudança positiva e progresso para essas comunidades.
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Espaço centralizado para a imagem no rodapé
    st.divider()  # Linha divisória para separar o conteúdo
    _, col_central, _ = st.columns([1, 2, 1])  # Coluna centralizada
    with col_central:
        st.image(carregar_imagem("Passos-magicos-icon-cor.png"), use_container_width=True)

# Executar o aplicativo
if __name__ == "__main__":
    show()
