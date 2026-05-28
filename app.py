import tempfile
import streamlit as st
import pandas as pd

from datetime import datetime
from chromadb.config import Settings

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Axcess",
    page_icon="🤖",
    layout="centered"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* ===== MAIN BACKGROUND ===== */

.main {
    background-color: #fffdf7;
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
        #facc15,
        #f59e0b
    );

    padding: 1.2rem 1.5rem;

    border-radius: 20px;

    margin-bottom: 1.5rem;

    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
}

.banner h1 {
    color: #1f2937;
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
}

.banner p {
    color: #374151;
    margin-top: 6px;
}

/* ===== FILE UPLOADER ===== */

.stFileUploader {

    background: white;

    padding: 18px;

    border-radius: 18px;

    border: 2px solid #fde68a;

    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* ===== ALL CHAT BUBBLES ===== */

[data-testid="stChatMessage"] {

    background-color: #fffdf5;

    border: 2px solid #facc15;

    border-radius: 18px;

    padding: 14px;

    margin-bottom: 14px;

    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}

/* ===== BUTTONS ===== */

div.stButton > button {

    border-radius: 14px;

    border: none;

    background: #facc15;

    color: #1f2937;

    font-weight: 700;

    padding: 0.6rem 1rem;
}

/* ===== USER ICON ===== */

[data-testid="stChatMessageAvatarUser"] {

    background-color: #9ca3af !important;
}

/* ===== ASSISTANT ICON ===== */

[data-testid="stChatMessageAvatarAssistant"] {

    background-color: #facc15 !important;
}

/* ===== SIDEBAR ===== */

[data-testid="stSidebar"] {

    background-color: #1f2937;
}

[data-testid="stSidebar"] * {

    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="banner">
    <h1>🤖 Axcess</h1>
    <p>
        ¡Hola! Soy Axcess 👋  
        Tu asistente interno de Axioma para ayudarte con dudas sobre
        RH, onboarding, seguros, facturación y procesos internos.
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# OPENAI KEY
# ==================================================

openai_api_key = st.secrets["OPENAI_API_KEY"]

# ==================================================
# SESSION MEMORY
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_files = st.file_uploader(
    "📎 Sube documentos PDF",
    type=["pdf"],
    accept_multiple_files=True
)

question = st.chat_input(
    "¿En qué puedo ayudarte hoy?"
)

# ==================================================
# EMBEDDINGS
# ==================================================

@st.cache_resource
def get_embeddings():

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key
    )

# ==================================================
# PDF PROCESSING
# ==================================================

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

# ==================================================
# FEEDBACK SAVE
# ==================================================

def save_feedback(question, answer, feedback):

    row = {
        "timestamp": datetime.now(),
        "question": question,
        "answer": answer,
        "feedback": feedback
    }

    df = pd.DataFrame([row])

    try:

        existing = pd.read_csv("questions_log.csv")

        updated = pd.concat([existing, df])

        updated.to_csv(
            "questions_log.csv",
            index=False
        )

    except:

        df.to_csv(
            "questions_log.csv",
            index=False
        )

# ==================================================
# MAIN CHAT
# ==================================================

if uploaded_files:

    with st.spinner("Leyendo documentos..."):

        db = process_pdfs(uploaded_files)

        retriever = db.as_retriever(
            search_kwargs={"k": 4}
        )

    if question:

        # =========================
        # USER MESSAGE
        # =========================

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        # =========================
        # RETRIEVE DOCS
        # =========================

        relevant_docs = retriever.invoke(question)

        context = "\n\n".join([
            doc.page_content
            for doc in relevant_docs
        ])

        # =========================
        # LLM
        # =========================

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0
        )

        # =========================
        # PROMPT
        # =========================

        prompt = ChatPromptTemplate.from_template("""
Eres Axcess, un chatbot interno de Axioma.

ALCANCE:
Responde preguntas sobre:
- Seguros
- Facturación
- Onboarding
- Procesos internos documentados

REGLAS:
1. Responde únicamente usando el contexto proporcionado.
2. No inventes información.
3. Si no encuentras suficiente información, responde:
"No encontré suficiente información en los documentos para responder con seguridad."
4. No uses conocimiento externo.
5. Si la pregunta está fuera del alcance, pide al usuario contactar RH o revisar documentación interna.

Contexto:
{context}

Pregunta:
{question}

Respuesta:
""")

        # =========================
        # CHAIN
        # =========================

        chain = prompt | llm

        answer = chain.invoke({
            "context": context,
            "question": question
        }).content

        # =========================
        # ASSISTANT MESSAGE
        # =========================

        with st.chat_message("assistant"):

            st.write(answer)

            col1, col2 = st.columns(2)

            with col1:

                if st.button("👍 Sí me ayudó"):

                    save_feedback(
                        question,
                        answer,
                        "helpful"
                    )

                    st.success(
                        "¡Gracias por tu feedback!"
                    )

            with col2:

                if st.button("👎 No ayudó"):

                    save_feedback(
                        question,
                        answer,
                        "not_helpful"
                    )

                    st.warning(
                        "Tu pregunta será revisada para mejorar la documentación interna."
                    )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:

    st.info(
        "Sube uno o más PDFs para comenzar."
    )
# =====================================
# ADMIN PANEL
# =====================================

st.divider()

admin_password = st.text_input(
    "Admin Access",
    type="password"
)

if admin_password == "axiomaadmin":

    st.subheader("📊 Feedback Dashboard")

    try:

        feedback_df = pd.read_csv(
            "questions_log.csv"
        )

        st.dataframe(feedback_df)

        st.subheader("❌ Preguntas no resueltas")

        unresolved = feedback_df[
            feedback_df["feedback"] == "not_helpful"
        ]

        st.dataframe(unresolved)

    except:

        st.info(
            "Todavía no hay feedback registrado."
        )
