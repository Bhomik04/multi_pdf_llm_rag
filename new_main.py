import streamlit as st  # to create a UI
from dotenv import load_dotenv
from langchain.chains import \
    ConversationalRetrievalChain  # which allows us to chat our vectorstore and adds memory to it
from langchain.embeddings import HuggingFaceInstructEmbeddings  # open AI embedding is paid so use hugging face
from langchain.llms import HuggingFaceHub  # importing Ollama for the chat model
from langchain.memory import ConversationBufferMemory  # this will give memory to the bot
from langchain.text_splitter import \
    CharacterTextSplitter  # the above lib is used to break the data into small chunks for processing
from langchain.vectorstores import FAISS  # a store that allows you to store a vectorized form of data
from pypdf import PdfReader  # used to read the given PDF files

from htmlTemplates import css, bot_template, user_template


# FAISS stores data locally, so it will be erased after the application is closed
# Reminder: update the code so that it also stores the data...

# This function gets PDF from UI, breaks it into chunks, and reads page by page to extract information
def get_pdf_text(pdf_docs):
    text = ""  # Initialize a variable called text where we will be storing all the text from the PDF
    for pdf in pdf_docs:  # Start looping through the PDFs
        pdf_reader = PdfReader(pdf)  # PDF reader object to read each page
        for page in pdf_reader.pages:  # Loops through the pages to read each page
            text += page.extract_text()  # Extract text from each page
    return text


# This function is used to divide the PDF data after submission in the UI into small chunks for processing
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(  # splits the text into chunks based on parameters as shown
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# This function vectorizes the chunk data & stores it in FAISS that is running locally
# The data will be lost when the program closes. Please refer to the reminder.
# def get_vectorstore(text_chunks):
# model_name = "hkunlp/instructor-xl"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name).to("cuda")

# def embed(texts):
#  inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
# outputs = model(**inputs)
# return outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy()

# embeddings = HuggingFaceInstructEmbeddings(model=embed)
# vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
# return vectorstore

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# This function sets up the conversational retrieval chain using a locally loaded model
def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    retries = 3  # Number of retries
    for attempt in range(retries):
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response.get('chat_history', [])

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
            break  # Exit loop if successful
        except ValueError as e:
            if attempt < retries - 1:
                st.warning(f"Attempt {attempt + 1} failed. Retrying...")
            else:
                st.error(f"All attempts failed. Error: {e}")


# Ensure to initialize session_state variables and other setup as per your previous code


# In def main() we are creating web UI using Streamlit lib in line 1
def main():
    load_dotenv()
    # Below this, the code is used for web UI. If you want to change anything in the UI, edit the code below
    st.set_page_config(page_title='Chat with "Geospat"', page_icon=":books:")  # Page config

    st.write(css, unsafe_allow_html=True)  # html file executed here

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with "Geospat" :books:')  # Header title
    user_question = st.text_input("Ask me what you want to know:")  # Asks for input
    if user_question:  # to handle user question
        if st.session_state.conversation is not None:
            handle_userinput(user_question)
        else:
            st.write("Please upload and process your documents first.")

    st.write(user_template.replace("{{MSG}}", "Hello Geospat"), unsafe_allow_html=True)  # html file
    st.write(bot_template.replace("{{MSG}}", "Hello Sir, How may I help you"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader('Your document')
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'",
                                    accept_multiple_files=True)  # Lets you upload your file & accept multiple files to upload and process multiple files
        if st.button("Process"):
            with st.spinner("Processing"):  # Shows the user a spinning bar to indicate that the process is taking place
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)  # Because of the get_pdf_text function, we will get the output in
                # a single string that will come in the variable called raw_text
                # st.write(raw_text)

                # Get text chunk
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)  # this will show chunks in the UI after being split

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create a conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)  # takes history of convo and return the next convo


if __name__ == '__main__':
    main()
