import streamlit as st

import openai

import fitz  # PyMuPDF for PDFs

import docx

import os

import hashlib

import logging

from sentence_transformers import SentenceTransformer, util



# Configure logging

logging.basicConfig(level=logging.INFO, filename='api_calls.log',

                    format='%(asctime)s - %(levelname)s - %(message)s')



# Set OpenAI API Key (Replace with your key or use secrets)
 openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load the embedding model

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")



# Function to extract text from PDFs

def extract_text_from_pdf(file):

    try:

        doc = fitz.open(stream=bytes(file.read()), filetype="pdf")

        return "\n".join([page.get_text("text") for page in doc])

    except Exception as e:

        logging.error(f"Error extracting PDF text: {e}")

        return ""



# Function to extract text from DOCX

def extract_text_from_docx(file):

    try:

        doc = docx.Document(file)

        return "\n".join([para.text for para in doc.paragraphs])

    except Exception as e:

        logging.error(f"Error extracting DOCX text: {e}")

        return ""



# Function to extract text from TXT

def extract_text_from_txt(file):

    try:

        return file.read().decode("utf-8")

    except Exception as e:

        logging.error(f"Error extracting TXT text: {e}")

        return ""



# Function to get OpenAI response

def ask_chatgpt(question, document_text):

    logging.info(f"Received question: {question}")

    cache_key = hashlib.sha256((question + document_text).encode()).hexdigest()

    if cache_key in st.session_state["response_cache"]:

        logging.info("Answer retrieved from cache.")

        return st.session_state["response_cache"][cache_key]

   

    logging.info("Calling OpenAI API.")

    try:

        response = openai.ChatCompletion.create(

            model="gpt-4",

            messages=[{"role": "system", "content": "You are an assistant."},

                      {"role": "user", "content": f"{document_text}\n\n{question}"}]

        )

        answer = response["choices"][0]["message"]["content"]

        st.session_state["response_cache"][cache_key] = answer  # Cache response

        return answer

    except Exception as e:

        logging.error(f"Error calling OpenAI API: {e}")

        return "Error in retrieving the answer. Please try again."



# Streamlit UI

st.title("📄 Multi-File Document Chatbot")



# Initialize session state

if "documents" not in st.session_state:

    st.session_state["documents"] = ""

if "response_cache" not in st.session_state:

    st.session_state["response_cache"] = {}



# File uploader

uploaded_files = st.file_uploader("Upload one or more documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:

    all_text = ""

    for file in uploaded_files:

        if file.type == "application/pdf":

            all_text += extract_text_from_pdf(file) + "\n"

        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":

            all_text += extract_text_from_docx(file) + "\n"

        elif file.type == "text/plain":

            all_text += extract_text_from_txt(file) + "\n"

    st.session_state["documents"] = all_text

    st.success("Documents processed successfully!")



# Question input

question = st.text_input("Ask a question about the uploaded documents:")

if question:

    if not st.session_state["documents"]:

        st.warning("Please upload documents first.")

    else:

        # Retrieve relevant text using embeddings

        document_sentences = st.session_state["documents"].split(". ")

        question_embedding = embedding_model.encode(question, convert_to_tensor=True)

        doc_embeddings = embedding_model.encode(document_sentences, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(question_embedding, doc_embeddings)[0]

        top_sentences = [document_sentences[i] for i in similarities.argsort(descending=True)[:5]]

        document_text = " ".join(top_sentences)

       

        # Get AI response

        answer = ask_chatgpt(question, document_text)

        st.write("### 🤖 Answer:", answer)
