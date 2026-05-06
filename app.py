import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Initialize Memory (Session State) ---
# This ensures the app remembers previous questions even when it reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Step 1: Setup & API Keys ---
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

st.title("My Custom RAG Pipeline ")

if not groq_key:
    st.error("Cannot find the GROQ_API_KEY. Check your .env file.")
    st.stop()

# --- Step 2: Document Ingestion ---
st.header("1. Upload your PDFs")
uploaded_files = st.file_uploader("Choose PDF documents", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents and building database..."):
        
        all_docs = []
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            
            loader = PyPDFLoader(temp_file_path)
            all_docs.extend(loader.load())
            os.remove(temp_file_path)
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_docs)
        
        # --- Step 3: Embeddings & Vector Store (FAISS) ---
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
    st.success(f"Database built successfully from {len(uploaded_files)} file(s)! We are ready to search.")
    
    # --- Step 4 & 5: The Modern Brain (LCEL) ---
    st.header("2. Ask Questions")
    llm = ChatGroq(api_key=groq_key, model_name="llama-3.3-70b-versatile")
    
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the following question based ONLY on the provided context. 
        If the answer is not in the context, say "I cannot find the answer in the document."
        
        Context: {context}
        
        Question: {input}
        """
    )
    
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_input("Ask a question about your PDF(s):")
        submitted = st.form_submit_button("Ask")
    
    if submitted and user_question:
        with st.spinner("Thinking..."):
            
            # Search FAISS
            results_with_scores = vector_store.similarity_search_with_score(user_question, k=3)
            retrieved_chunks = [result[0] for result in results_with_scores]
            context_text = "\n\n".join(doc.page_content for doc in retrieved_chunks)
            
            # Generate Answer
            pipeline = prompt_template | llm | StrOutputParser()
            answer = pipeline.invoke({"context": context_text, "input": user_question})
            
            # Display Current Answer
            st.write("### Current Answer:")
            st.info(answer)
            
            with st.expander("Where did I find this? (Source Chunks & Scores)"):
                for i, (chunk, score) in enumerate(results_with_scores):
                    st.write(f"**Source {i+1} | Similarity Score (L2): {score:.4f}**")
                    st.write(chunk.page_content)
                    st.divider()

            # Save this Q&A to our memory bank!
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer
            })

    # --- Step 6: Search History ---
    # Only show this section if the user has actually asked a question
    if len(st.session_state.chat_history) > 0:
        st.divider()
        st.header("3. Search History")
        st.write("Here are the questions you have asked during this session:")
        
        # We use reversed() so the newest history items show up at the top of the list!
        for past_chat in reversed(st.session_state.chat_history):
            with st.expander(f"🗣️ **Q:** {past_chat['question']}"):
                st.write(f"🤖 **A:** {past_chat['answer']}")