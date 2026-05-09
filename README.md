# 🚀 Document Question Answering System using RAG

**Live Demo:** [Click here to view the live app!](https://rag-pdf-explorer.streamlit.app) 

## 📌 Project Overview
This project is a fully functional Retrieval-Augmented Generation (RAG) pipeline designed to ingest, process, and answer questions based on uploaded PDF documents. It acts as an intelligent assistant, allowing users to chat directly with their documents.

## 🛠️ Tech Stack
* **Frontend UI:** Streamlit
* **Orchestration:** LangChain (LCEL)
* **Large Language Model (LLM):** Llama-3 (via Groq API)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS

## ✨ Key Features
* **Multi-Document Support:** Upload and process multiple PDFs simultaneously.
* **Semantic Search:** Retrieves the top 3 most relevant context chunks using FAISS similarity search.
* **Source Transparency:** Displays the exact document chunks and L2 similarity scores used to generate the answer.
* **Session Memory:** Maintains a conversational history log during the active session.

## ⚙️ How It Works
1.  **Ingestion:** PDF text is extracted and split into manageable chunks using `RecursiveCharacterTextSplitter`.
2.  **Embedding:** Text chunks are converted into dense vector representations.
3.  **Storage & Retrieval:** Vectors are stored locally in FAISS. When a user asks a question, the system retrieves the most semantically similar chunks.
4.  **Generation:** The Llama-3 model receives the user's question alongside the retrieved context to generate a precise, hallucination-free answer.
