import os
import tempfile
import streamlit as st

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
        embedding=get_embeddings()
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

        answer = llm.invoke(prompt).content

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Primero sube uno o más PDFs.")

