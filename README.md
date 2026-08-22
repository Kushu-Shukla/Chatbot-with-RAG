# 🤖 AI Chatbot with RAG (Retrieval-Augmented Generation)

Welcome to the **AI Chatbot with RAG**! This project is an intelligent document assistant that reads, understands, and answers questions based on your provided PDF documents using state-of-the-art Large Language Models (LLMs).

## 🚀 Features
- **Intelligent RAG Pipeline**: Accurately retrieves context from PDFs before answering to eliminate AI hallucinations.
- **Lightning Fast Inference**: Powered by **Groq** and the **Llama-3 (8B)** model for incredibly fast responses.
- **Conversational Memory**: Remembers your chat history for fluid, natural conversations.
- **Clean UI**: Built with **Streamlit** for a beautiful, responsive, and easy-to-use chat interface.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **LLM Engine**: Groq API (Llama3-8b)
- **RAG Framework**: LangChain
- **Embeddings**: HuggingFace (`all-MiniLM-L12-v2`)
- **Document Processing**: PyPDF

## 📂 Project Structure
- `phase_1.py`: The basic Streamlit Chat UI with session memory.
- `phase_2.py`: Integration with Groq API and LangChain for basic AI responses.
- `phase_3.py`: The complete application featuring the full RAG pipeline (PDF parsing, chunking, embeddings, and vector search).

## ⚙️ How to Run Locally

### 1. Install Dependencies
Make sure you have Python 3.11+ installed. Then install the required packages:
```bash
pipenv install
```
*(Or install manually via `pip install streamlit langchain langchain-groq pypdf sentence-transformers`)*

### 2. Add Your API Key
You will need a free API key from [Groq](https://console.groq.com/keys). Set it as an environment variable in your terminal:
```bash
# Windows
set GROQ_API_KEY=your_api_key_here

# Mac/Linux
export GROQ_API_KEY="your_api_key_here"
```

### 3. Run the App
Launch the Streamlit server using the complete phase 3 file:
```bash
streamlit run phase_3.py
```

## ☁️ Deployment
This project is perfectly configured to be deployed for free on **Streamlit Community Cloud**.
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Connect this GitHub repository.
3. Select `phase_3.py` as the main file path.
4. Add your `GROQ_API_KEY` in the Advanced Settings (Secrets) section.
5. Deploy!
