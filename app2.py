import streamlit as st
import openai
import fitz  # PyMuPDF for PDFs
import docx
import os
import hashlib
import logging
from sentence_transformers import SentenceTransformer, util
# Configure logging
logging.basicConfig(level=logging.INFO, filename='api_calls.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Set OpenAI API Key
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# Streamlit UI
st.title("📄 Multi-File Document Chatbot")

# Initialize session state
if "documents" not in st.session_state:
    st.session_state["documents"] = ""

# File uploader
uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    documents_text = ""
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                documents_text += page.get_text("text") + "\n"
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                documents_text += para.text + "\n"
        elif uploaded_file.type == "text/plain":
            documents_text += uploaded_file.read().decode("utf-8") + "\n"
    
    st.session_state["documents"] = documents_text
    st.success("Documents uploaded and processed!")

# Function to get OpenAI response
def ask_chatgpt(question, document_text):
    cache_key = hashlib.sha256((document_text).encode()).hexdigest()
    if cache_key in st.session_state.get("response_cache", {}):
        logging.info("Answer retrieved from cache.")
        return st.session_state["response_cache"][cache_key]
    
    logging.info("Calling OpenAI API.")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that answers questions based on the provided document."},
                {"role": "user", "content": f"{document_text}\n\n{question}"}
            ],
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        if "response_cache" not in st.session_state:
            st.session_state["response_cache"] = {}
        st.session_state["response_cache"][cache_key] = answer
        return answer
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API Error: {e}")
        st.error(f"OpenAI API Error: {e}")
        return "An error occurred with the OpenAI API. Please try again later."
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        st.error("An unexpected error occurred. Please check the logs for details.")
        return "An unexpected error occurred. Please try again later."

# Question input
question = st.text_input("Ask a question about the uploaded documents:")
if question:
    if not st.session_state["documents"]:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Thinking..."):
            document_text = st.session_state["documents"]
            answer = ask_chatgpt(question, document_text)
            st.write("### 🤖 Answer:", answer)
