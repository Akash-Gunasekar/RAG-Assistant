# Gemini RAG Assistant
A Streamlit app that leverages Google Generative AI (Gemini) along with ChromaDB vector database to create a Retrieval-Augmented Generation (RAG) chatbot.
Upload documents (PDF, DOCX, TXT), which are chunked and embedded using Gemini embeddings, then queried in a conversational interface.

# Features
* Upload multiple documents (PDF, DOCX, TXT)
* Extracts and chunks text intelligently to preserve sentence integrity
* Embeds chunks with Google Generative AI embedding model (text-embedding-004)
* Stores embeddings in a persistent ChromaDB vector database
* Chat interface that queries ChromaDB for relevant document chunks
* Generates responses using Google Gemini models (gemini-2.0-flash by default)
* Document management UI to upload, view, and delete files
* Supports selecting which documents to query
* Stores chat history in Streamlit session state
* Configurable Google API key and chat model via sidebar

# Getting Started
* Prerequisites
  * Python 3.9+
  * Google Generative AI API key
  * Install ChromaDB and other dependencies
