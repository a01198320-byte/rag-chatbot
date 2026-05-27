import os
import tempfile
import streamlit as st

import chromadb
from langchain_core.prompts import ChatPromptTemplate
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

openai_api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Chatbot RAG PDFs", page_icon="📄")

st.title("📄 Chatbot RAG con PDFs")

openai_api_key = st.text_input("OpenAI API Key", type="password")

uploaded_files = st.file_uploader(
    "Sube tus PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

question = st.chat_input("Haz una pregunta sobre tus documentos...")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key
    
    )

def process_pdfs(files):
    docs = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=get_embeddings(),
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=False
    )
    )

    return vectorstore

if uploaded_files:
    with st.spinner("Leyendo PDFs..."):
        db = process_pdfs(uploaded_files)
        retriever = db.as_retriever(search_kwargs={"k": 4})

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        relevant_docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        Contesta la pregunta usando SOLO la información del contexto.
        Si no está en los documentos, di: "No encontré esa información en los PDFs."

        Contexto:
        {context}

        Pregunta:
        {question}
        """

      llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
Eres un chatbot interno de Axioma.

ALCANCE DEL CHATBOT:
Este chatbot solo contesta preguntas relacionadas con:
- Seguros
- Facturación
- Un poco de onboarding

Si el usuario pregunta algo fuera de estos temas, responde amablemente:
"Este chatbot está diseñado para responder únicamente preguntas sobre seguros, facturación y algunos temas de onboarding. Para otros temas internos, por favor consulta el SharePoint de procesos internos o contacta a tu BP de Recursos Humanos según tu célula."

SOBRE SHAREPOINT:
Sí contamos con un SharePoint donde se documentan los procesos internos de la empresa.
Actualmente el contenido está en actualización, por lo que no está listo para consultarse de forma directa.
Sin embargo, puedes hacer referencia al link de SharePoint cuando esté disponible para que los usuarios lleguen al contenido posteriormente.

SOPORTE INTERNO POR CÉLULA:
Cada grupo de colaboradores, llamado “célula”, tiene asignado un Business Partner de Recursos Humanos como punto de contacto.

Distribución:
- TH Stefany: Célula 360 y Célula del Como Si
- TH Priscila: Célula Alpha
- TH Marlen: Célula Rentable, Célula Máxima, Célula Nueva
- TH Franquicia Juan Carlos: Célula Lobo GV, Célula Lobo RH, Célula Jaguar

REGLAS IMPORTANTES:
1. Responde únicamente con información encontrada en el contexto o en las reglas internas anteriores.
2. No inventes datos, fechas, ligas, requisitos, responsables ni conclusiones.
3. Si la respuesta no está claramente en el contexto, di:
   "No encontré suficiente información en los documentos para responder con seguridad."
4. Si necesitas más contexto, dilo claramente.
5. Si hay fecha de emisión o publicación disponible, inclúyela.
6. Si hay liga original o fuente disponible, inclúyela.
7. Si diferentes documentos dicen cosas distintas, menciona la diferencia.
8. No uses conocimiento externo.
9. Si el usuario pregunta quién es su BP o punto de contacto de RH, responde usando la distribución por célula.
10. Si el usuario hace una pregunta fuera del alcance del chatbot o que no puedas responder con seguridad:
   - Pídele al usuario que indique a qué célula pertenece.
   - Después oriéntalo con el Business Partner correspondiente.
   - Usa este formato:

   "No cuento con suficiente información para responder esa pregunta.
   ¿Me podrías indicar a qué célula perteneces para compartirte el contacto de tu Business Partner de Recursos Humanos?"

11. Cuando el usuario indique su célula, responde con el BP correspondiente:

- Célula 360 → TH Stefany
- Célula del Como Si → TH Stefany
- Célula Alpha → TH Priscila
- Célula Rentable → TH Marlen
- Célula Máxima → TH Marlen
- Célula Nueva → TH Marlen
- Célula Lobo GV → TH Franquicia Juan Carlos
- Célula Lobo RH → TH Franquicia Juan Carlos
- Célula Jaguar → TH Franquicia Juan Carlos

12. Cuando redirijas al usuario, usa un tono amable y útil.

Contexto recuperado de documentos:
{context}

Pregunta del usuario:
{question}

Respuesta:
""")

      chain = prompt | llm

answer = chain.invoke({
    "context": context,
    "question": question
}).content

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Primero sube uno o más PDFs.")

