import os
import warnings
import logging
import streamlit as st

# Suppress logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Modern Langchain imports
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="My RAG Assistant", page_icon="🤖")
st.title('🤖 My Intelligent Document Assistant')
st.markdown("Ask me any questions based on the uploaded documents!")

with st.sidebar:
    st.header("Chat Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    pdf_name = "./reflexion.pdf"
    if not os.path.exists(pdf_name):
        return None
        
    loader = PyPDFLoader(pdf_name)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
    vectorstore = Chroma.from_documents(docs, embeddings)
    
    return vectorstore

prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content': prompt})
    
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best, 
                                            the most accurate and most precise answers. 
                                            Answer the user's question based ONLY on the context below:
                                            
                                            <context>
                                            {context}
                                            </context>
                                            
                                            Question: {input}
                                            
                                            Start the answer directly. No small talk please""")

    model="llama3-8b-8192"

    groq_chat = ChatGroq(
            groq_api_key=os.environ.get("GROQ_API_KEY"), 
            model_name=model
    )

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Document 'reflexion.pdf' not found. Please add it to the repository.")
        else:
            # Modern Langchain retrieval
            document_chain = create_stuff_documents_chain(groq_chat, groq_sys_prompt)
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
           
            result = retrieval_chain.invoke({"input": prompt})
            response = result["answer"]
            
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role':'assistant', 'content':response})
    except Exception as e:
        st.error(f"Error: {str(e)}")
