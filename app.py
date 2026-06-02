import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Bot de Seguridad", page_icon="🌊")

informacion_conocimiento = """
GUIA DE SEGURIDAD ANTE INUNDACIONES EN ZAPOPAN:
- Si vives cerca de zonas de riesgo, ten siempre lista una mochila con documentos importantes, linterna, radio de pilas y agua embotellada.
- No intentes cruzar calles o arroyos con corriente de agua.
- Si el agua entra a tu casa, corta el suministro electrico inmediatamente.
- Mantente al tanto de las redes oficiales de Proteccion Civil Zapopan.
- Si la situacion es critica, dirigete a las zonas altas.
- El numero de emergencia local es el 911.
"""

st.title("🌊 Asistente de Seguridad: Inundaciones")

@st.cache_resource
def inicializar_bot():
    docs = [Document(page_content=informacion_conocimiento)]
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    prompt_template = ChatPromptTemplate.from_template("""
    Eres un asistente especializado en seguridad ante inundaciones en Zapopan.
    CONTEXTO: {context}
    PREGUNTA: {input}
    Responde de manera clara y util basandote en el contexto.
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

if prompt := st.chat_input("¿Que duda tienes sobre seguridad o inundaciones?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            respuesta = qa_chain.invoke(prompt)
            st.markdown(respuesta)
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
