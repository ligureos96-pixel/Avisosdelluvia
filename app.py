import streamlit as st
import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

# CONFIGURACIÓN SEGURA
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error: No se encontró la API KEY en los Secrets.")
    st.stop()

st.set_page_config(
    page_title="Bot de Seguridad",
    page_icon="🌊"
)

# BASE DE CONOCIMIENTO
informacion_conocimiento = """
GUÍA DE SEGURIDAD ANTE INUNDACIONES EN ZAPOPAN:

- Si vives cerca de zonas de riesgo, ten siempre lista una mochila con documentos importantes, linterna, radio de pilas y agua embotellada.
- No intentes cruzar calles o arroyos con corriente de agua; la fuerza del agua puede arrastrar un vehículo fácilmente.
- Si el agua entra a tu casa, corta el suministro eléctrico inmediatamente para evitar cortocircuitos.
- Mantente al tanto de las redes oficiales de Protección Civil Zapopan y del ayuntamiento.
- Si la situación es crítica, dirígete a las zonas altas identificadas por las autoridades.
- El número de emergencia local es el 911.
"""

# INTERFAZ
st.title("🌊 Asistente de Seguridad: Inundaciones")

st.sidebar.title("Información del Proyecto")
st.sidebar.text("""
• Carlos Yoel
• Pablo Medina
• Alex Ayala
• Ponce de León
""")

@st.cache_resource
def inicializar_bot():
    docs = [Document(page_content=informacion_conocimiento)]
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs_divididos = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        docs_divididos,
        OpenAIEmbeddings()
    )

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa

# Inicializar bot
try:
    qa = inicializar_bot()
except Exception as e:
    st.error(f"Error al inicializar el bot: {e}")
    st.stop()

# HISTORIAL DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# CHAT INPUT
if prompt := st.chat_input("¿Qué duda tienes sobre seguridad?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        resultado = qa.invoke({"query": prompt})
        respuesta = resultado["result"]
        st.markdown(respuesta)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})

