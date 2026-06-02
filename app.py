import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
    # Creamos el documento directamente sin splitter para evitar más imports
    docs = [Document(page_content=informacion_conocimiento)]
    
    vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    prompt_template = ChatPromptTemplate.from_template("""
    Eres un asistente especializado en seguridad ante inundaciones en Zapopan.
    
    CONTEXTO DISPONIBLE:
    {context}
    
    INSTRUCCIONES:
    - Usa PRIMERO la información del contexto proporcionado.
    - Si el contexto NO tiene la información exacta, puedes usar tu conocimiento general sobre seguridad, pero siempre aclara que es información general.
    - Responde de manera clara, directa y útil.
    - Si la pregunta no tiene nada que ver con seguridad o inundaciones, indícalo amablemente.
    
    PREGUNTA: {input}
    
    RESPUESTA:
    """)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

qa_chain = inicializar_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué duda tienes sobre seguridad o inundaciones?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            respuesta = qa_chain.invoke(prompt)
            st.markdown(respuesta)
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
