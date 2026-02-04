
import os
import warnings
import logging
import streamlit as st



from groq import Groq

groq_api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=groq_api_key)

# =========================
# Page Config (TOP)
# =========================
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🇮🇳",
    layout="centered",
)

# =========================
# Disable warnings/logs
# =========================
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# =========================
# LangChain imports
# =========================
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain

# =========================
# Title & Header
# =========================
st.markdown(
    """
    <h1 style="text-align:center;">RAG Chatbot  🤖📚</h1>
    <p style="text-align:center; color: gray;">
    Ask questions based on <b>India’s Struggle for Independence</b>
    </p>
    """,
    unsafe_allow_html=True
)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📘 About")
    st.write(
        "This chatbot answers questions using the book "
        "**India's Struggle for Independence**."
    )

    st.divider()

    st.header("⚙️ Settings")
    top_k = st.slider("Number of sources to retrieve", 1, 5, 3)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# Display Chat History
# =========================
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    st.chat_message(message["role"], avatar=avatar).markdown(message["content"])

# =========================
# Vectorstore (cached)
# =========================
@st.cache_resource
def get_vectorstore(pdf_path="./Indias_Struggle.pdf"):
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found: {pdf_path}")
        return None

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# =========================
# Chat Input
# =========================
prompt = st.chat_input("Ask a question about India’s independence…")

if prompt:
    # Show user message
    st.chat_message("user", avatar="🧑‍💻").markdown(prompt)
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # =========================
    # Get API Key
    # ========================


    if not groq_api_key:
        st.error("Please set your GROQ_API_KEY environment variable")
        st.stop()

    # =========================
    # Create Groq LLM
    # =========================
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    # =========================
    # RAG Prompt
    # =========================
    groq_sys_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant.
Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{input}

Answer:"""
    )

    # =========================
    # Run Retrieval Chain
    # =========================
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.stop()

        document_chain = create_stuff_documents_chain(
            llm=groq_chat,
            prompt=groq_sys_prompt
        )

        retrieval_chain = create_retrieval_chain(
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": top_k}
            ),
            combine_docs_chain=document_chain
        )

        with st.spinner("📚 Searching the book and thinking..."):
            result = retrieval_chain.invoke({"input": prompt})
            response = result["answer"]

        # =========================
        # Show Assistant Response
        # =========================
        st.chat_message("assistant", avatar="🤖").markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        # =========================
        # Show Sources
        # =========================
        sources = result.get("context", [])

        with st.expander("📖 Sources used"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}:**")
                st.write(doc.page_content[:300] + "...")

    except Exception as e:
        st.error(f"Error: {e}")

# =========================
# Styling (BOTTOM)


# =========================
st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
