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
    
# Installation
```bash
git clone https://github.com/Akash-Gunasekar/RAG-Assistant.git
cd RAG-Assistant
pip install -r requirements.txt
```
# Running the App
```bash
streamlit run app.py
```
The app opens in your browser.

# Usage
1. Configure API Key and Model
   Enter your Google Generative AI API key in the sidebar and select a chat model (default: gemini-2.0-flash). Click "Save Configuration".
2. Upload Documents
   Upload PDF, DOCX, or TXT files. The app will extract text, chunk it, embed the chunks, and store them in ChromaDB.
3. Chat with Documents
   Select which uploaded documents to query. Ask questions in the chat box. The assistant retrieves relevant context and generates answers.
4. Manage Documents
   View uploaded documents, delete unwanted files and their embeddings from the database.
5. Clear Chat History
   Clear previous conversations to start fresh.

# File Formats Supported
* PDF (.pdf)
* Microsoft Word (.docx)
* Plain Text (.txt)

# Code Structure Highlights
1. Text Extraction: Uses PyMuPDF (fitz) for PDFs, python-docx for DOCX, and plain reading for TXT.
2. Text Chunking: Splits text into 500 character chunks with 100 character overlap, avoiding splitting sentences mid-way.
3. Embedding: Uses google.generativeai to generate semantic embeddings.
4. Vector DB: Persistent ChromaDB stores embeddings and metadata.
5. Chat Interface: Uses Google Gemini models for natural language generation.

# Troubleshooting
1. Ensure API key is valid and has permissions for Google Generative AI.
2. Verify ChromaDB is installed and accessible.
3. Streamlit errors usually indicate missing dependencies or configuration issues.

# Future Enhancements
1. Support more document formats (e.g., HTML, Markdown).
2. Add user authentication and document access control.
3. Enhance chunking algorithm with NLP sentence segmentation.
4. Support multi-turn conversation context.
