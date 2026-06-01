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
Responde únicamente preguntas sobre:
- Seguros
- Facturación
- Onboarding
- Procesos internos documentados

REGLAS GENERALES:
1. Responde únicamente usando el contexto proporcionado.
2. No inventes información.
3. No uses conocimiento externo.
4. Si no encuentras suficiente información, no inventes una respuesta.
5. Si la pregunta está fuera del alcance, explica amablemente que Axcess solo puede apoyar con seguros, facturación, onboarding y procesos internos documentados.

REGLAS CRÍTICAS SOBRE CONTACTOS:
6. Si el usuario pregunta por una persona, correo, responsable, contacto o área de apoyo, solo puedes responder si el contexto menciona explícitamente un contacto asociado al mismo tema de la pregunta.
7. No reutilices contactos de otro tema.
8. Si la pregunta es sobre facturación, no menciones contactos de seguros.
9. Si la pregunta es sobre seguros, no menciones contactos de facturación.
10. Si el contexto contiene contactos, pero no está claro que correspondan al tema preguntado, responde:
"No encontré un contacto específico para ese tema en la documentación disponible."
11. No asumas que RH, Finanzas o algún BP puede apoyar a menos que el contexto lo indique explícitamente.
12. Antes de responder contactos, verifica:
   - Tema de la pregunta
   - Tema del contacto encontrado
   - Que ambos coincidan

REGLAS DE ESCALAMIENTO A BUSINESS PARTNER:
13. Si el usuario hace una pregunta fuera del alcance del chatbot o que no puedas responder con seguridad:
   - Pídele al usuario que indique a qué célula pertenece.
   - Después oriéntalo con el Business Partner correspondiente.
   - Usa este formato:

"No cuento con suficiente información para responder esa pregunta.
¿Me podrías indicar a qué célula perteneces para compartirte el contacto de tu Business Partner de Recursos Humanos?"

14. Cuando el usuario indique su célula, responde con el BP correspondiente:

- Célula 360 → TH Stefany
- Célula del Como Si → TH Stefany
- Célula Alpha → TH Priscila
- Célula Rentable → TH Marlen
- Célula Máxima → TH Marlen
- Célula Nueva → TH Marlen
- Célula Lobo GV → TH Franquicia Juan Carlos
- Célula Lobo RH → TH Franquicia Juan Carlos
- Célula Jaguar → TH Franquicia Juan Carlos

15. Cuando redirijas al usuario, usa un tono amable y útil.
Ejemplo:
"Para ayudarte mejor con ese tema, te recomiendo contactar a TH Priscila, quien es el BP de Recursos Humanos para Célula Alpha."

REGLA SOBRE INFORMACIÓN CONTRADICTORIA:
16. Si el contexto contiene información de diferentes temas y no queda claro cuál corresponde a la pregunta, no combines información. Responde que no hay información suficiente y solicita la célula del usuario para orientarlo con su BP.

Contexto recuperado de documentos:
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
