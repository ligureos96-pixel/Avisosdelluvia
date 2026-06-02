import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# CONFIGURACIÓN
# Asegúrate de tener GOOGLE_API_KEY en tus Secrets de Streamlit
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("No se encontró la GOOGLE_API_KEY en los Secrets.")

st.set_page_config(page_title="Bot de Seguridad", page_icon="🌊")

# BASE DE CONOCIMIENTO
informacion_conocimiento = """
GUIA DE SEGURIDAD ANTE INUNDACIONES EN ZAPOPAN:
- Si vives cerca de zonas de riesgo, ten siempre lista una mochila con documentos importantes, linterna, radio de pilas y agua embotellada.
- No intentes cruzar calles o arroyos con corriente de agua.
- Si el agua entra a tu casa, corta el suministro eléctrico inmediatamente para evitar cortocircuitos.
- Mantente al tanto de las redes oficiales de Protección Civil Zapopan y del ayuntamiento.
- Si la situación es crítica, dirígete a las zonas altas identificadas por las autoridades.
- El número de emergencia local es el 911.
"""

st.title("🌊 Asistente de Seguridad: Inundaciones")

@st.cache_resource
def inicializar_bot():
    # Usamos gemini-1.5-flash porque es el más compatible
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    prompt_template = ChatPromptTemplate.from_template("""
    Eres un asistente de seguridad. Usa EXCLUSIVAMENTE esta guía para responder:
    {context}
    
    PREGUNTA: {input}
    RESPUESTA:
    """)
    
    rag_chain = (
        {"context": lambda x: informacion_conocimiento, "input": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain

qa_chain = inicializar_bot()

# GESTIÓN DEL CHAT
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
            try:
                respuesta = qa_chain.invoke(prompt)
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
            except Exception as e:
                st.error(f"Error al consultar: {e}")
