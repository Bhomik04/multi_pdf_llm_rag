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



## Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

## Install the required dependencies:

pip install -r requirements.txt

## Usage
Place your .env file in the project root directory. This file should contain your API keys and other environment variables.

## Run the Streamlit application:

streamlit run new_main.py
Open your web browser and go to http://localhost:8501 to view the application.

## Upload your PDF documents using the sidebar, click "Process" to extract and vectorize the text, and then start chatting with the AI model.

## File Overview
new_main.py: The Streamlit application's main script.
htmlTemplates.py: Contains HTML templates for the chat UI.
requirements.txt: Lists the required Python packages.
.env: (not included) Should contain your environment variables.
## Project Structure
.
├── .env
├── htmlTemplates.py
├── new_main.py
├── README.md
├── requirements.txt
└── venv/
## Troubleshooting
Please make sure you have all the necessary environment variables set in your .env file.
Make sure you have the correct versions of the dependencies installed as specified in requirements.txt.
If you encounter issues with FAISS, ensure you have installed the faiss-cpu package correctly.
Contributing
Feel free to open issues or submit pull requests for improvements or bug fixes. Contributions are welcome!

License
This project is licensed under the MIT License.

1. Clone this repository:

```sh
git clone https://github.com/your-username/geospat-chat-app.git
cd geospatial-chat-app
