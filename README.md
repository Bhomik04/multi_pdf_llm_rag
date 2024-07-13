## Geospat Chat Application

This project is a Streamlit-based web application that allows users to upload PDF documents, process them, and chat with a conversational AI model trained on the content of the PDFs.

## Features

- Upload and process multiple PDF documents.
- Extract text from PDFs and break it into manageable chunks.
- Store vectorized data using FAISS for efficient retrieval.
- Chat with an AI model to retrieve information from the uploaded documents.
- Display chat history with a user-friendly interface.

## Requirements

- Python 3.8 or higher
- Streamlit
- PyPDF
- LangChain
- HuggingFace Transformers
- FAISS
- PyTorch
- Sentence Transformers
- dotenv

## Installation

1. Clone this repository:

```sh
git clone https://github.com/your-username/geospat-chat-app.git
cd geospat-chat-app
