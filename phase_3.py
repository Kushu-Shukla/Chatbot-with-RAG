import os
import warnings
import logging
import streamlit as st

# LangChain and Groq Imports
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Suppress warnings for a cleaner terminal output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------
# UI Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="My RAG Assistant", page_icon="🤖")
st.title('🤖 My Intelligent Document Assistant')
st.markdown("Ask me any questions based on the uploaded documents!")

# Sidebar for controls
with st.sidebar:
    st.header("Chat Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Render previous messages
for message in st.session_state.chat_history:
    st.chat_message(message['role']).markdown(message['content'])


# ---------------------------------------------------------
# Core Application Logic
# ---------------------------------------------------------
@st.cache_resource
def initialize_vector_database():
    """Loads a PDF and creates a searchable vector database."""
    target_document = "./reflexion.pdf"
    
    # Load and split the document into smaller chunks
    document_loader = [PyPDFLoader(target_document)]
    
    vector_index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(document_loader)
    
    return vector_index.vectorstore

# Capture user input
user_question = st.chat_input('Ask me anything about your documents...')

if user_question:
    # 1. Display user message
    st.chat_message('user').markdown(user_question)
    st.session_state.chat_history.append({'role': 'user', 'content': user_question})
    
    # 2. Setup the AI Model (Groq Llama 3)
    llm_model = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"), 
        model_name="llama3-8b-8192"
    )

    try:
        # 3. Retrieve context and generate answer
        db = initialize_vector_database()
        if db is None:
            st.error("Document failed to load into the vector database.")
      
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )
       
        # Execute query
        generation_result = qa_chain({"query": user_question})
        ai_response = generation_result["result"] 
        
        # 4. Display AI response
        st.chat_message('assistant').markdown(ai_response)
        st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
        
    except Exception as error:
        st.error(f"Something went wrong: {str(error)}")
