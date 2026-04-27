import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- UI Setup ---
st.set_page_config(page_title="Universal RAG Dashboard", page_icon="📊")
st.title("📊 Universal AI Financial Analyst")
st.markdown("Upload any PDF document and your Groq API key to instantly extract insights.")

# --- User Inputs ---
api_key = st.text_input("Enter your Groq API Key:", type="password")
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# --- Cache the Database Load ---
@st.cache_resource
def load_and_embed_data(file_name, file_hash): # FIX 1: File hash forces the cache to reset correctly
    loader = PyPDFLoader(file_name)
    pages = loader.load()
    # FIX 2: Optimal size so the MiniLM embedding model doesn't truncate our text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_db

# --- Main App Logic ---
if api_key and uploaded_file:
    os.environ["GROQ_API_KEY"] = api_key
    
    temp_pdf_path = "temp_uploaded.pdf"
    file_bytes = uploaded_file.getvalue()
    with open(temp_pdf_path, "wb") as f:
        f.write(file_bytes)
        
    with st.spinner("Reading PDF and building AI Brain..."):
        # Pass the file size as a hash to automatically reset the cache when a new file is uploaded
        vector_db = load_and_embed_data(temp_pdf_path, len(file_bytes))

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # FIX 3: Standard Similarity Search to prevent the "MMR Penalty" from hiding the right page
    retriever = vector_db.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 10} 
    )
    
    custom_prompt_template = """You are an expert financial analyst. Use the following pieces of context to answer the user's question. 
    If the information is inside a table, read the rows and columns very carefully before answering. 
    If you cannot find the exact answer in the provided context, politely say "I cannot find the exact data in this document." Do NOT make up numbers.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    CUSTOM_PROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )

    # User Question Area
    st.divider()
    query = st.text_input("Ask a question about your uploaded document:")
    
    if st.button("Submit"):
        if query:
            with st.spinner("Analyzing document..."):
                result = qa_chain.invoke({"query": query})
                
                st.subheader("Answer:")
                st.write(result["result"])
                
                st.subheader("Sources:")
                unique_pages = set([doc.metadata.get('page', 'N/A') for doc in result["source_documents"]])
                for page in sorted(unique_pages):
                    st.caption(f"- Page {page}")
        else:
            st.error("Please enter a question.")
elif not api_key:
    st.warning("Please enter your Groq API key.")
elif not uploaded_file:
    st.info("Please upload a PDF document to begin.")