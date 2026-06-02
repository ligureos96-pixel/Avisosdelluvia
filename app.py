import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
# --- CONFIGURACIÓN ---
# Si usas Streamlit Cloud, configura la clave en Settings > Secrets
os.environ["OPENAI_API_KEY"] = "TU_API_KEY_AQUI"

st.set_page_config(page_title="Bot de Seguridad", page_icon="🌊")

# --- CONOCIMIENTO ---
informacion_conocimiento = """
GUÍA DE SEGURIDAD ANTE INUNDACIONES EN ZAPOPAN:
- Si vives cerca de zonas de riesgo, ten siempre lista una mochila con documentos importantes, linterna, radio de pilas y agua embotellada.
- No intentes cruzar calles o arroyos con corriente de agua; la fuerza del agua puede arrastrar un vehículo fácilmente.
- Si el agua entra a tu casa, corta el suministro eléctrico inmediatamente para evitar cortocircuitos.
- Mantente al tanto de las redes oficiales de Protección Civil Zapopan y del ayuntamiento.
- Si la situación es crítica, dirígete a las zonas altas identificadas por las autoridades.
- El número de emergencia local es el 911.
"""

# --- UI ---
st.title("🌊 Asistente de Seguridad: Inundaciones")
st.sidebar.title("Información del Proyecto")
st.sidebar.text("• Carlos Yoel\n• Pablo Medina\n• Alex Ayala\n• Ponce de León")

@st.cache_resource
def inicializar_bot():
    docs = [Document(page_content=informacion_conocimiento)]
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs_divididos = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(docs_divididos, OpenAIEmbeddings())
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o"), 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever()
    )

qa = inicializar_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué duda tienes sobre seguridad?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        resultado = qa.invoke({"query": prompt})
        respuesta = resultado["result"]
        st.markdown(respuesta)
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
