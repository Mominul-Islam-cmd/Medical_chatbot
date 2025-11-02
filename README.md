# MedGPT-RAG-Chatbot

A **Retrieval-Augmented Generation (RAG) Medical Chatbot** built with **Python, LangChain, FAISS, and an LLM**.  
This chatbot allows users to query medical documents and receive context-aware, AI-generated responses.  

> **Note:** This project is for educational and informational purposes only. It **does not provide medical advice**.

---

## Features

- **Document Retrieval:** Efficiently searches medical documents using embeddings stored in FAISS.  
- **RAG-Powered Responses:** Combines retrieval with LLM generation for accurate answers.  
- **Supports Multiple Formats:** PDFs, TXT, and other document types.  
- **Interactive Chat:** Ask questions in natural language and get informative responses.  
- **Modular Architecture:** Easy to extend or integrate with web apps.  

---

## Technologies Used

- **Python** – Core language  
- **LangChain** – For RAG pipeline construction  
- **FAISS** – Vector database for storing embeddings  
- **LLM** – OpenAI GPT, Gemini, or other language models  
- **Document Parsing** – Handles PDFs, TXT, DOCX  
- Optional: Web interface (Streamlit/Gradio/Flask/Django)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mominul-Islam-cmd/MedGPT-RAG-Chatbot.git
cd MedGPT-RAG-Chatbot
