An interactive Retrieval-Augmented Generation (RAG) chatbot built using LangChain 1.x, Groq LLM, FAISS, and Streamlit.
The chatbot answers questions grounded in the book India’s Struggle for Independence by retrieving relevant context from a PDF and generating accurate responses.

✨ Features

📘 PDF-based Question Answering (RAG)

🔍 Semantic search using FAISS vector store

🤖 Fast LLM inference using Groq (LLaMA 3.1)

💬 Interactive chat-style UI

📖 Source document visibility for transparency

⚙️ Adjustable retrieval depth (top-k)

🧹 Clear chat with one click

🚀 Built on LangChain 1.x (LCEL-compatible)

🏗️ Tech Stack
Layer	Technology
UI	Streamlit
LLM	Groq (LLaMA-3.1-8B-Instant)
Framework	LangChain 1.x
Embeddings	SentenceTransformers
Vector DB	FAISS
Document Loader	PyPDF
Language	Python

📂 Project Structure
RAG Chatbot/
│
├── app.py                     # Main Streamlit application
├── Indias_Struggle.pdf        # Source document
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── .streamlit/
    └── secrets.toml           # API key (recommended)

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/milan-7417/RAG-chatbot-using-LangChain.git
cd india-independence-rag-chatbot

2️⃣ Create and activate environment (recommended)
conda create -n rag-chatbot python=3.10 -y
conda activate rag-chatbot

3️⃣ Install dependencies
pip install -r requirements.txt

🔑 Setting up GROQ API Key
✅ Recommended (Streamlit Secrets)

Create the file:

.streamlit/secrets.toml


Add:

GROQ_API_KEY = "your_groq_api_key_here"

⚠️ Alternative (Environment Variable)

Windows (PowerShell):

$env:GROQ_API_KEY="your_groq_api_key_here"


Linux / macOS:

export GROQ_API_KEY="your_groq_api_key_here"

▶️ Running the Application

Always run Streamlit using:

python -m streamlit run app.py


Then open in browser:

http://localhost:8501
