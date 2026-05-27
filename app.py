import tempfile
import streamlit as st

from chromadb.config import Settings

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Axioma Internal Chatbot",
    page_icon="📄"
)
st.markdown("""
<style>

/* ===== MAIN BACKGROUND ===== */

.main {
    background-color: #f4f7fb;
}

/* ===== PAGE WIDTH ===== */

.block-container {
    max-width: 950px;
    padding-top: 1.5rem;
}

/* ===== TOP BANNER ===== */

.banner {
    background: linear-gradient(
        90deg,
        #1e3a8a,
        #2563eb
    );

    padding: 1.2rem 1.5rem;

    border-radius: 20px;

    margin-bottom: 1.5rem;

    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
}

.banner h1 {
    color: white;
    margin: 0;
    font-size: 2rem;
}

.banner p {
    color: rgba(255,255,255,0.85);
    margin-top: 6px;
}

/* ===== FILE UPLOADER ===== */

.stFileUploader {
    background: white;

    padding: 18px;

    border-radius: 18px;

    border: 2px solid #dbeafe;

    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* ===== CHAT BUBBLES ===== */

[data-testid="stChatMessage"] {

    background: white;

    border-radius: 18px;

    padding: 14px;

    margin-bottom: 14px;

    border: 2px solid #c7d2fe;

    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}

/* ===== USER MESSAGE ===== */

[data-testid="stChatMessage"]:has(.stChatMessageContent-user) {

    border: 2px solid #93c5fd;

    background-color: #eff6ff;
}

/* ===== ASSISTANT MESSAGE ===== */

[data-testid="stChatMessage"]:has(.stChatMessageContent-assistant) {

    border: 2px solid #c4b5fd;

    background-color: #faf5ff;
}

/* ===== CHAT INPUT ===== */

.stChatInputContainer {

    border-radius: 18px;
}

/* ===== BUTTONS ===== */

div.stButton > button {

    border-radius: 14px;

    border: none;

    background: #2563eb;

    color: white;

    font-weight: 600;

    padding: 0.6rem 1rem;
}

/* ===== SIDEBAR ===== */

[data-testid="stSidebar"] {

    background-color: #111827;
}

[data-testid="stSidebar"] * {

    color: white;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
# 🤖 Axioma Assistant  
Consulta información interna sobre **seguros, facturación y onboarding** a partir de tus PDFs.
""")

# =====================================
# OPENAI KEY
# =====================================

openai_api_key = st.secrets["OPENAI_API_KEY"]

# =====================================
# SESSION MEMORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =====================================
# FILE UPLOADER
# =====================================

uploaded_files = st.file_uploader(
    "Sube tus PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

question = st.chat_input(
    "Haz una pregunta sobre los documentos..."
)

# =====================================
# EMBEDDINGS
# =====================================

@st.cache_resource
def get_embeddings():

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key
    )

# =====================================
# PDF PROCESSING
# =====================================

def process_pdfs(files):

    docs = []

    for file in files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

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

# =====================================
# MAIN CHAT
# =====================================

if uploaded_files:

    with st.spinner("Leyendo PDFs..."):

        db = process_pdfs(uploaded_files)

        retriever = db.as_retriever(
            search_kwargs={"k": 4}
        )

    if question:

        # USER MESSAGE

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        # RETRIEVE DOCUMENTS

        relevant_docs = retriever.invoke(question)

        context = "\n\n".join([
            doc.page_content
            for doc in relevant_docs
        ])

        # =====================================
        # LLM
        # =====================================

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0
        )

        # =====================================
        # PROMPT
        # =====================================

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

SOPORTE INTERNO POR CÉLULA:

- TH Stefany:
  - Célula 360
  - Célula del Como Si

- TH Priscila:
  - Célula Alpha

- TH Marlen:
  - Célula Rentable
  - Célula Máxima
  - Célula Nueva

- TH Franquicia Juan Carlos:
  - Célula Lobo GV
  - Célula Lobo RH
  - Célula Jaguar

REGLAS IMPORTANTES:

1. Responde únicamente usando el contexto recuperado.

2. No inventes información.

3. Si la respuesta no está claramente en el contexto, responde:

"No encontré suficiente información en los documentos para responder con seguridad."

4. No uses conocimiento externo.

5. Si el usuario pregunta algo fuera del alcance o que no puedas responder:
   - pídele su célula
   - oriéntalo con su BP correspondiente

6. Cuando redirijas al usuario, usa un tono amable y útil.

Contexto:
{context}

Pregunta:
{question}

Respuesta:
""")

        # =====================================
        # CHAIN
        # =====================================

        chain = prompt | llm

        answer = chain.invoke({
            "context": context,
            "question": question
        }).content

        # =====================================
        # ASSISTANT MESSAGE
        # =====================================

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:

    st.info("Primero sube uno o más PDFs.")

