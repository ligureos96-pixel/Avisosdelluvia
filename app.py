import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# CONFIGURACIÓN
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Bot de Seguridad", page_icon="🌊")

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

st.title("🌊 Asistente de Seguridad: Inundaciones")

@st.cache_resource
def inicializar_bot():
    docs = [Document(page_content=informacion_conocimiento)]
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs_divididos = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(docs_divididos, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt_template = ChatPromptTemplate.from_template("""
    Responde basándote solo en este contexto: {context}
    Pregunta: {input}
    """)
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, combine_docs_chain)

qa_chain = inicializar_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué duda tienes?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = qa_chain.invoke({"input": prompt})
        respuesta = response["answer"]
        st.markdown(respuesta)
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
